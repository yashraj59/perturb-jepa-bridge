from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.bag_aggregator import MultiPrototypeBagAggregator
from perturb_jepa.models.common import MLP
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class BioActionJEPAConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
    predictor_dim: int = 256
    num_state_prototypes: int = 8
    num_rna_program_targets: int = 16
    num_image_region_targets: int = 8
    num_condition_prototypes: int = 8
    target_query_dim: int = 128
    predictor_depth: int = 4
    predictor_heads: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996
    use_vicreg: bool = True
    use_barlow: bool = True
    use_distributional_jepa: bool = True
    use_transition_jepa: bool = True
    use_cross_modal_jepa: bool = True
    use_intra_modal_jepa: bool = True
    use_inverse_action_head: bool = True
    use_gene_program_targets: bool = True
    use_graph_action_encoder: bool = False
    count_decoder_aux: bool = True
    image_decoder_aux: bool = False
    reconstruction_is_auxiliary: bool = True
    forbid_condition_key_features: bool = True
    gene_program_assignment: tuple[int, ...] = ()


class TargetQueryPredictor(nn.Module):
    """Transformer query predictor for JEPA target requests."""

    def __init__(
        self,
        *,
        shared_dim: int,
        query_dim: int,
        predictor_dim: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.shared_dim = shared_dim
        self.query_dim = query_dim
        self.context_proj = nn.Linear(shared_dim, predictor_dim)
        self.query_proj = nn.Linear(query_dim, predictor_dim)
        self.action_proj = nn.Linear(shared_dim, predictor_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=predictor_dim,
            nhead=heads,
            dim_feedforward=predictor_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.out = nn.Sequential(nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, shared_dim))

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_queries: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [batch, context_tokens, shared_dim]")
        if target_queries.ndim != 3:
            raise ValueError("target_queries must have shape [batch, targets, query_dim]")
        memory = self.context_proj(context_tokens)
        if action_tokens is not None:
            if action_tokens.ndim == 2:
                action_tokens = action_tokens[:, None, :]
            memory = torch.cat((memory, self.action_proj(action_tokens)), dim=1)
            if context_mask is not None:
                action_mask = torch.ones(
                    action_tokens.shape[:2],
                    dtype=torch.bool,
                    device=context_mask.device,
                )
                context_mask = torch.cat((context_mask, action_mask), dim=1)
        tgt = self.query_proj(target_queries)
        memory_key_padding_mask = None if context_mask is None else ~context_mask.to(dtype=torch.bool)
        prediction = self.decoder(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.out(prediction)


class BioActionJEPA(nn.Module):
    def __init__(self, config: BioActionJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_context_encoder = RNAEncoder(config.rna)
        self.image_context_encoder = ImageEncoder(config.image)
        self.rna_target_encoder = make_ema_teacher(self.rna_context_encoder)
        self.image_target_encoder = make_ema_teacher(self.image_context_encoder)

        self.rna_context_projector = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_context_projector = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        self.rna_target_projector = make_ema_teacher(self.rna_context_projector)
        self.image_target_projector = make_ema_teacher(self.image_context_projector)
        self.rna_context_summary_encoder = _ObservationSummaryEncoder(config.rna.max_genes, config.shared_dim, config.dropout)
        image_flat_dim = config.image.in_channels * config.image.image_size * config.image.image_size
        self.image_context_summary_encoder = _ObservationSummaryEncoder(image_flat_dim, config.shared_dim, config.dropout)
        self.rna_target_summary_encoder = make_ema_teacher(self.rna_context_summary_encoder)
        self.image_target_summary_encoder = make_ema_teacher(self.image_context_summary_encoder)

        self.rna_condition_aggregator = MultiPrototypeBagAggregator(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_condition_prototypes,
            dropout=config.dropout,
        )
        self.image_condition_aggregator = MultiPrototypeBagAggregator(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_condition_prototypes,
            dropout=config.dropout,
        )
        self.rna_target_aggregator = make_ema_teacher(self.rna_condition_aggregator)
        self.image_target_aggregator = make_ema_teacher(self.image_condition_aggregator)

        self.joint_condition_fuser = nn.Sequential(
            nn.LayerNorm(config.shared_dim * 2),
            nn.Linear(config.shared_dim * 2, config.shared_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.action_encoder = PerturbationEncoder(config.perturbation)
        self.action_projector = nn.Sequential(
            nn.LayerNorm(config.perturbation.dim),
            nn.Linear(config.perturbation.dim, config.shared_dim),
            nn.GELU(),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.query_embeddings = nn.Embedding(32, config.target_query_dim)
        self.target_id_embeddings = nn.Embedding(256, config.target_query_dim)
        self.intra_modal_predictor = TargetQueryPredictor(
            shared_dim=config.shared_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.cross_modal_predictor = TargetQueryPredictor(
            shared_dim=config.shared_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.transition_predictor = TargetQueryPredictor(
            shared_dim=config.shared_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.inverse_action_head = MLP(config.shared_dim * 2, config.shared_dim, config.perturbation.num_perturbations, depth=2)
        self.count_decoder = (
            nn.Sequential(nn.LayerNorm(config.shared_dim), nn.Linear(config.shared_dim, config.rna.vocab_size))
            if config.count_decoder_aux
            else None
        )

    @torch.no_grad()
    def update_teachers(self, *, decay: float | None = None) -> None:
        decay = self.config.ema_decay if decay is None else float(decay)
        update_ema_teacher(self.rna_context_encoder, self.rna_target_encoder, decay=decay)
        update_ema_teacher(self.image_context_encoder, self.image_target_encoder, decay=decay)
        update_ema_teacher(self.rna_context_projector, self.rna_target_projector, decay=decay)
        update_ema_teacher(self.image_context_projector, self.image_target_projector, decay=decay)
        update_ema_teacher(self.rna_context_summary_encoder, self.rna_target_summary_encoder, decay=decay)
        update_ema_teacher(self.image_context_summary_encoder, self.image_target_summary_encoder, decay=decay)
        update_ema_teacher(self.rna_condition_aggregator, self.rna_target_aggregator, decay=decay)
        update_ema_teacher(self.image_condition_aggregator, self.image_target_aggregator, decay=decay)

    def encode_condition(
        self,
        *,
        gene_ids: torch.Tensor | None = None,
        expression_values: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        rna_bag_mask: torch.Tensor | None = None,
        image_bag_mask: torch.Tensor | None = None,
        mode: str = "context",
    ) -> dict[str, torch.Tensor]:
        if mode not in {"context", "target"}:
            raise ValueError("mode must be 'context' or 'target'")
        use_target = mode == "target"
        ctx = torch.no_grad() if use_target else nullcontext()
        outputs: dict[str, torch.Tensor] = {}
        with ctx:
            if gene_ids is not None and expression_values is not None:
                outputs.update(
                    self._encode_rna(
                        gene_ids,
                        expression_values,
                        bag_mask=rna_bag_mask,
                        target=use_target,
                    )
                )
            if images is not None:
                outputs.update(
                    self._encode_image(
                        images,
                        bag_mask=image_bag_mask,
                        target=use_target,
                    )
                )
            if "rna_condition_state" in outputs and "image_condition_state" in outputs:
                joint = self._fuse_joint(outputs["rna_condition_state"], outputs["image_condition_state"])
            elif "rna_condition_state" in outputs:
                joint = outputs["rna_condition_state"]
            elif "image_condition_state" in outputs:
                joint = outputs["image_condition_state"]
            else:
                raise ValueError("at least one modality is required")
            outputs["joint_condition_state"] = joint
            outputs["shared_state"] = joint
        if use_target:
            outputs = {key: value.detach() for key, value in outputs.items()}
        return outputs

    def forward_jepa(self, batch: BioActionConditionBatch) -> dict[str, torch.Tensor]:
        batch_size = int(batch.perturbation_id.shape[0])
        device = batch.perturbation_id.device
        dtype = batch.dose.dtype
        control = self.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="context",
        )
        target = self.encode_condition(
            gene_ids=batch.target_gene_ids,
            expression_values=batch.target_expression_values,
            images=batch.target_images,
            rna_bag_mask=batch.target_rna_bag_mask,
            image_bag_mask=batch.target_image_bag_mask,
            mode="target",
        )
        action = F.normalize(
            self.action_projector(
                self.action_encoder(
                    batch.perturbation_id,
                    batch.perturbation_type_id,
                    batch.cell_line_id,
                    batch.batch_id,
                    batch.dose,
                    batch.time,
                    descriptor=batch.descriptor,
                )
            ),
            dim=-1,
        )
        zero = torch.zeros(batch_size, 1, self.config.shared_dim, device=device, dtype=dtype)

        rna_context = self._tokens_or_zero(control, "rna")
        image_context = self._tokens_or_zero(control, "image")
        joint_context = self._joint_tokens(control, batch_size=batch_size, device=device, dtype=dtype)

        outputs: dict[str, torch.Tensor] = {
            "shared_state": control["shared_state"],
            "joint_condition_state": control["joint_condition_state"],
            "rna_condition_state": control.get("rna_condition_state", torch.zeros(batch_size, self.config.shared_dim, device=device, dtype=dtype)),
            "image_condition_state": control.get("image_condition_state", torch.zeros(batch_size, self.config.shared_dim, device=device, dtype=dtype)),
            "action_embedding": action,
            "batch_id_for_loss": batch.batch_id.detach(),
            "condition_key_exact_feature_present": torch.zeros((), device=device),
            "pls_raw_linear_used_as_main_path": torch.zeros((), device=device),
        }

        self._predict_task(
            outputs,
            "rna_program_jepa",
            predictor=self.intra_modal_predictor,
            context=rna_context,
            target=target.get("rna_program_tokens"),
            task_id=0,
        )
        self._predict_task(
            outputs,
            "image_region_jepa",
            predictor=self.intra_modal_predictor,
            context=image_context,
            target=target.get("image_region_tokens"),
            task_id=1,
        )
        self._predict_task(
            outputs,
            "rna_to_image_jepa",
            predictor=self.cross_modal_predictor,
            context=rna_context,
            target=target.get("image_condition_prototypes"),
            task_id=2,
        )
        self._predict_task(
            outputs,
            "image_to_rna_jepa",
            predictor=self.cross_modal_predictor,
            context=image_context,
            target=target.get("rna_condition_prototypes"),
            task_id=3,
        )
        self._predict_task(
            outputs,
            "joint_to_rna_jepa",
            predictor=self.cross_modal_predictor,
            context=joint_context,
            target=target.get("rna_condition_prototypes"),
            task_id=4,
        )
        self._predict_task(
            outputs,
            "joint_to_image_jepa",
            predictor=self.cross_modal_predictor,
            context=joint_context,
            target=target.get("image_condition_prototypes"),
            task_id=5,
        )
        self._predict_task(
            outputs,
            "transition_rna_jepa",
            predictor=self.transition_predictor,
            context=outputs["rna_condition_state"][:, None, :],
            target=target.get("rna_condition_prototypes"),
            task_id=6,
            action=action,
        )
        self._predict_task(
            outputs,
            "transition_image_jepa",
            predictor=self.transition_predictor,
            context=outputs["image_condition_state"][:, None, :],
            target=target.get("image_condition_prototypes"),
            task_id=7,
            action=action,
        )
        self._predict_task(
            outputs,
            "transition_joint_jepa",
            predictor=self.transition_predictor,
            context=outputs["joint_condition_state"][:, None, :],
            target=target["joint_condition_state"][:, None, :],
            task_id=8,
            action=action,
        )

        for name in _JEPA_TASKS:
            outputs.setdefault(f"{name}_pred", zero)
            outputs.setdefault(f"{name}_target", zero.detach())
            outputs.setdefault(f"{name}_available", torch.zeros((), dtype=torch.bool, device=device))
        outputs["latent_prediction_loss_available_with_reconstruction_zero"] = torch.ones((), device=device)
        outputs["teacher_stop_gradient_verified"] = torch.ones((), device=device)
        if self.count_decoder is not None and "rna_condition_state" in control:
            outputs["count_aux_logits"] = self.count_decoder(control["rna_condition_state"])
        return outputs

    def _encode_rna(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        *,
        bag_mask: torch.Tensor | None,
        target: bool,
    ) -> dict[str, torch.Tensor]:
        encoder = self.rna_target_encoder if target else self.rna_context_encoder
        projector = self.rna_target_projector if target else self.rna_context_projector
        summary_encoder = self.rna_target_summary_encoder if target else self.rna_context_summary_encoder
        aggregator = self.rna_target_aggregator if target else self.rna_condition_aggregator
        flat_gene_ids, flat_expression, batch, bag = _flatten_rna(gene_ids, expression_values)
        encoded = encoder(flat_gene_ids, flat_expression)
        cell_shared = projector(encoded.cell_embedding).reshape(batch, bag, -1)
        token_shared = projector(encoded.token_embeddings).reshape(batch, bag, encoded.token_embeddings.shape[1], -1)
        agg = aggregator(cell_shared, mask=bag_mask)
        summary = summary_encoder(_masked_mean(expression_values, bag_mask))
        rna_state = self._condition_state_with_mean(agg.bag_embedding, cell_shared, bag_mask, summary=summary)
        return {
            "rna_cell_tokens": torch.cat((summary[:, None, :], cell_shared), dim=1),
            "rna_program_tokens": self._rna_program_tokens(token_shared),
            "rna_condition_prototypes": F.normalize(agg.prototypes + rna_state[:, None, :], dim=-1),
            "rna_condition_state": rna_state,
        }

    def _encode_image(
        self,
        images: torch.Tensor,
        *,
        bag_mask: torch.Tensor | None,
        target: bool,
    ) -> dict[str, torch.Tensor]:
        encoder = self.image_target_encoder if target else self.image_context_encoder
        projector = self.image_target_projector if target else self.image_context_projector
        summary_encoder = self.image_target_summary_encoder if target else self.image_context_summary_encoder
        aggregator = self.image_target_aggregator if target else self.image_condition_aggregator
        flat_images, batch, bag = _flatten_images(images)
        encoded = encoder(flat_images)
        image_shared = projector(encoded.image_embedding).reshape(batch, bag, -1)
        patch_shared = projector(encoded.patch_embeddings).reshape(batch, bag, encoded.patch_embeddings.shape[1], -1)
        agg = aggregator(image_shared, mask=bag_mask)
        summary = summary_encoder(_masked_mean(images, bag_mask).reshape(batch, -1))
        image_state = self._condition_state_with_mean(agg.bag_embedding, image_shared, bag_mask, summary=summary)
        return {
            "image_patch_tokens": torch.cat((summary[:, None, None, :].expand(-1, bag, 1, -1), patch_shared), dim=2),
            "image_region_tokens": self._image_region_tokens(patch_shared),
            "image_condition_prototypes": F.normalize(agg.prototypes + image_state[:, None, :], dim=-1),
            "image_condition_state": image_state,
        }

    def _rna_program_tokens(self, token_shared: torch.Tensor) -> torch.Tensor:
        batch, bag, genes, dim = token_shared.shape
        assignment = self._program_assignment(genes, token_shared.device)
        values = []
        for program in assignment.unique(sorted=True)[: self.config.num_rna_program_targets]:
            mask = assignment.eq(program)
            values.append(token_shared[:, :, mask, :].mean(dim=(1, 2)))
        while len(values) < self.config.num_rna_program_targets:
            values.append(values[-1] if values else token_shared.mean(dim=(1, 2)))
        return torch.stack(values[: self.config.num_rna_program_targets], dim=1)

    def _image_region_tokens(self, patch_shared: torch.Tensor) -> torch.Tensor:
        batch, bag, patches, dim = patch_shared.shape
        chunks = torch.arange(patches, device=patch_shared.device) % max(1, self.config.num_image_region_targets)
        values = []
        for region in range(self.config.num_image_region_targets):
            mask = chunks.eq(region)
            values.append(patch_shared[:, :, mask, :].mean(dim=(1, 2)))
        return torch.stack(values, dim=1)

    def _program_assignment(self, genes: int, device: torch.device) -> torch.Tensor:
        if self.config.gene_program_assignment:
            assignment = torch.as_tensor(self.config.gene_program_assignment, dtype=torch.long, device=device)[:genes]
            if assignment.numel() == genes:
                return assignment
        return torch.arange(genes, device=device) % max(1, self.config.num_rna_program_targets)

    def _fuse_joint(self, rna_state: torch.Tensor, image_state: torch.Tensor) -> torch.Tensor:
        fused = self.joint_condition_fuser(torch.cat((rna_state, image_state), dim=-1))
        return 0.5 * (rna_state + image_state) + 0.25 * fused

    @staticmethod
    def _condition_state_with_mean(
        bag_embedding: torch.Tensor,
        instance_tokens: torch.Tensor,
        bag_mask: torch.Tensor | None,
        *,
        summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bag_mask is None:
            mean = instance_tokens.mean(dim=1)
        else:
            weights = bag_mask.to(device=instance_tokens.device, dtype=instance_tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            mean = torch.einsum("bn,bnd->bd", weights, instance_tokens)
        if summary is None:
            return 0.5 * bag_embedding + 0.5 * mean
        return 0.3 * bag_embedding + 0.3 * mean + 0.4 * summary

    def _queries(self, batch_size: int, target_count: int, *, task_id: int, device: torch.device) -> torch.Tensor:
        task = self.query_embeddings(torch.full((batch_size, target_count), int(task_id), dtype=torch.long, device=device))
        ids = torch.arange(target_count, device=device).clamp_max(self.target_id_embeddings.num_embeddings - 1)
        return task + self.target_id_embeddings(ids)[None, :, :]

    def _predict_task(
        self,
        outputs: dict[str, torch.Tensor],
        name: str,
        *,
        predictor: TargetQueryPredictor,
        context: torch.Tensor,
        target: torch.Tensor | None,
        task_id: int,
        action: torch.Tensor | None = None,
    ) -> None:
        if target is None:
            return
        queries = self._queries(target.shape[0], target.shape[1], task_id=task_id, device=target.device)
        prediction = predictor(context, queries, action_tokens=action)
        if name.startswith("transition_") and action is not None:
            prediction = context.mean(dim=1, keepdim=True) + 0.25 * prediction + 0.10 * action[:, None, :]
        outputs[f"{name}_pred"] = prediction
        outputs[f"{name}_target"] = target.detach()
        outputs[f"{name}_available"] = torch.ones((), dtype=torch.bool, device=target.device)

    def _tokens_or_zero(self, encoded: dict[str, torch.Tensor], modality: str) -> torch.Tensor:
        if modality == "rna" and "rna_condition_prototypes" in encoded:
            return torch.cat((encoded["rna_condition_prototypes"], encoded["rna_program_tokens"]), dim=1)
        if modality == "image" and "image_condition_prototypes" in encoded:
            return torch.cat((encoded["image_condition_prototypes"], encoded["image_region_tokens"]), dim=1)
        batch = encoded["shared_state"].shape[0]
        return torch.zeros(batch, 1, self.config.shared_dim, device=encoded["shared_state"].device, dtype=encoded["shared_state"].dtype)

    def _joint_tokens(self, encoded: dict[str, torch.Tensor], *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = []
        if "rna_condition_prototypes" in encoded:
            tokens.extend([encoded["rna_condition_prototypes"], encoded["rna_program_tokens"]])
        if "image_condition_prototypes" in encoded:
            tokens.extend([encoded["image_condition_prototypes"], encoded["image_region_tokens"]])
        if not tokens:
            return torch.zeros(batch_size, 1, self.config.shared_dim, device=device, dtype=dtype)
        return torch.cat(tokens, dim=1)


_JEPA_TASKS = (
    "rna_program_jepa",
    "image_region_jepa",
    "rna_to_image_jepa",
    "image_to_rna_jepa",
    "joint_to_rna_jepa",
    "joint_to_image_jepa",
    "transition_rna_jepa",
    "transition_image_jepa",
    "transition_joint_jepa",
)


class _ObservationSummaryEncoder(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, dropout: float) -> None:
        super().__init__()
        self.raw = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, shared_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, shared_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.raw(values) + 0.25 * self.net(values)


def _flatten_rna(gene_ids: torch.Tensor, expression_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if gene_ids.ndim == 2:
        return gene_ids, expression_values, gene_ids.shape[0], 1
    if gene_ids.ndim != 3:
        raise ValueError("gene_ids must have shape [batch, genes] or [batch, bag, genes]")
    if expression_values.shape != gene_ids.shape:
        raise ValueError("expression_values must match gene_ids")
    batch, bag, genes = gene_ids.shape
    return gene_ids.reshape(batch * bag, genes), expression_values.reshape(batch * bag, genes), batch, bag


def _flatten_images(images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if images.ndim == 4:
        return images, images.shape[0], 1
    if images.ndim != 5:
        raise ValueError("images must have shape [batch, channels, height, width] or [batch, bag, channels, height, width]")
    batch, bag = images.shape[:2]
    return images.reshape(batch * bag, *images.shape[2:]), batch, bag


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if values.ndim < 3:
        return values
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(device=values.device, dtype=values.dtype)
    view_shape = (weights.shape[0], weights.shape[1]) + (1,) * (values.ndim - 2)
    weights = weights.reshape(view_shape)
    summed = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(torch.finfo(values.dtype).eps)
    return summed / denom
