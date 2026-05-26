from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.bag_aggregator import MultiPrototypeBagAggregator
from perturb_jepa.models.bioaction_jepa import (
    TargetQueryPredictor,
    _ObservationSummaryEncoder,
    _flatten_images,
    _flatten_rna,
    _masked_mean,
)
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class BioTechJEPAConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 64
    bio_dim: int = 48
    tech_dim: int = 16
    predictor_dim: int = 128
    target_query_dim: int = 64
    predictor_depth: int = 1
    predictor_heads: int = 4
    num_condition_prototypes: int = 4
    num_rna_program_targets: int = 8
    num_image_region_targets: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996
    count_decoder_aux: bool = False
    forbid_condition_key_features: bool = True
    gene_program_assignment: tuple[int, ...] = ()


class BioTechJEPA(nn.Module):
    """Factorized BioTech-JEPA with biological and technical latent branches."""

    def __init__(self, config: BioTechJEPAConfig) -> None:
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

        image_flat_dim = config.image.in_channels * config.image.image_size * config.image.image_size
        self.rna_context_summary_encoder = _ObservationSummaryEncoder(config.rna.max_genes, config.shared_dim, config.dropout)
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
        self.target_joint_condition_fuser = make_ema_teacher(self.joint_condition_fuser)

        self.rna_bio_head = _branch_head(config.shared_dim, config.bio_dim, config.dropout)
        self.image_bio_head = _branch_head(config.shared_dim, config.bio_dim, config.dropout)
        self.joint_bio_head = _branch_head(config.shared_dim, config.bio_dim, config.dropout)
        self.rna_tech_head = _branch_head(config.shared_dim, config.tech_dim, config.dropout)
        self.image_tech_head = _branch_head(config.shared_dim, config.tech_dim, config.dropout)
        self.joint_tech_head = _branch_head(config.shared_dim, config.tech_dim, config.dropout)
        self.rna_target_bio_head = make_ema_teacher(self.rna_bio_head)
        self.image_target_bio_head = make_ema_teacher(self.image_bio_head)
        self.joint_target_bio_head = make_ema_teacher(self.joint_bio_head)
        self.rna_target_tech_head = make_ema_teacher(self.rna_tech_head)
        self.image_target_tech_head = make_ema_teacher(self.image_tech_head)
        self.joint_target_tech_head = make_ema_teacher(self.joint_tech_head)

        self.action_encoder = PerturbationEncoder(config.perturbation)
        self.action_projector = nn.Sequential(
            nn.LayerNorm(config.perturbation.dim),
            nn.Linear(config.perturbation.dim, config.bio_dim),
            nn.GELU(),
            nn.Linear(config.bio_dim, config.bio_dim),
        )

        self.query_embeddings = nn.Embedding(32, config.target_query_dim)
        self.target_id_embeddings = nn.Embedding(256, config.target_query_dim)
        self.intra_modal_predictor = TargetQueryPredictor(
            shared_dim=config.bio_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.cross_modal_predictor = TargetQueryPredictor(
            shared_dim=config.bio_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.transition_predictor = TargetQueryPredictor(
            shared_dim=config.bio_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.batch_probe_from_z_bio = nn.Linear(config.bio_dim, config.perturbation.num_batches)
        self.batch_head_from_z_tech = nn.Linear(config.tech_dim, config.perturbation.num_batches)
        self.perturbation_probe_from_z_bio = nn.Linear(config.bio_dim, config.perturbation.num_perturbations)
        self.perturbation_probe_from_z_tech = nn.Linear(config.tech_dim, config.perturbation.num_perturbations)
        self.count_decoder = (
            nn.Sequential(nn.LayerNorm(config.bio_dim), nn.Linear(config.bio_dim, config.rna.vocab_size))
            if config.count_decoder_aux
            else None
        )

    @torch.no_grad()
    def update_teachers(self, *, decay: float | None = None) -> None:
        decay = self.config.ema_decay if decay is None else float(decay)
        for online, target in (
            (self.rna_context_encoder, self.rna_target_encoder),
            (self.image_context_encoder, self.image_target_encoder),
            (self.rna_context_projector, self.rna_target_projector),
            (self.image_context_projector, self.image_target_projector),
            (self.rna_context_summary_encoder, self.rna_target_summary_encoder),
            (self.image_context_summary_encoder, self.image_target_summary_encoder),
            (self.rna_condition_aggregator, self.rna_target_aggregator),
            (self.image_condition_aggregator, self.image_target_aggregator),
            (self.joint_condition_fuser, self.target_joint_condition_fuser),
            (self.rna_bio_head, self.rna_target_bio_head),
            (self.image_bio_head, self.image_target_bio_head),
            (self.joint_bio_head, self.joint_target_bio_head),
            (self.rna_tech_head, self.rna_target_tech_head),
            (self.image_tech_head, self.image_target_tech_head),
            (self.joint_tech_head, self.joint_target_tech_head),
        ):
            update_ema_teacher(online, target, decay=decay)

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
                outputs.update(self._encode_image(images, bag_mask=image_bag_mask, target=use_target))
            if "rna_shared_state" in outputs and "image_shared_state" in outputs:
                joint_shared = self._fuse_joint(outputs["rna_shared_state"], outputs["image_shared_state"], target=use_target)
                joint_z_bio = _normalize(self._head("joint_bio", use_target)(joint_shared))
                joint_z_tech = self._head("joint_tech", use_target)(joint_shared)
            elif "rna_shared_state" in outputs:
                joint_shared = outputs["rna_shared_state"]
                joint_z_bio = outputs["rna_z_bio"]
                joint_z_tech = outputs["rna_z_tech"]
            elif "image_shared_state" in outputs:
                joint_shared = outputs["image_shared_state"]
                joint_z_bio = outputs["image_z_bio"]
                joint_z_tech = outputs["image_z_tech"]
            else:
                raise ValueError("at least one modality is required")
            outputs["joint_shared_state"] = joint_shared
            outputs["joint_z_bio"] = joint_z_bio
            outputs["joint_z_tech"] = joint_z_tech
            outputs["shared_state"] = joint_z_bio
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
        target_context = self.encode_condition(
            gene_ids=batch.target_gene_ids,
            expression_values=batch.target_expression_values,
            images=batch.target_images,
            rna_bag_mask=batch.target_rna_bag_mask,
            image_bag_mask=batch.target_image_bag_mask,
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
        action = _normalize(
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
            )
        )
        zero = torch.zeros(batch_size, 1, self.config.bio_dim, device=device, dtype=dtype)
        outputs: dict[str, torch.Tensor] = {
            "joint_z_bio": control["joint_z_bio"],
            "joint_z_tech": control["joint_z_tech"],
            "target_joint_z_bio": target["joint_z_bio"],
            "target_joint_z_tech": target["joint_z_tech"],
            "shared_state": control["joint_z_bio"],
            "joint_condition_state": control["joint_z_bio"],
            "action_embedding": action,
            "batch_id_for_loss": batch.batch_id.detach(),
            "condition_key_exact_feature_present": torch.zeros((), device=device),
            "pls_raw_linear_used_as_main_path": torch.zeros((), device=device),
            "teacher_stop_gradient_verified": torch.ones((), device=device),
            "latent_prediction_loss_available_with_reconstruction_zero": torch.ones((), device=device),
            "batch_logits_from_z_bio_for_probe_only": self.batch_probe_from_z_bio(control["joint_z_bio"].detach()),
            "batch_logits_from_z_tech": self.batch_head_from_z_tech(control["joint_z_tech"]),
            "perturbation_logits_from_z_bio_for_probe_only": self.perturbation_probe_from_z_bio(control["joint_z_bio"].detach()),
            "perturbation_logits_from_z_tech_for_probe_only": self.perturbation_probe_from_z_tech(control["joint_z_tech"].detach()),
            "z_bio_variance": control["joint_z_bio"].var(dim=0, unbiased=False).mean(),
            "z_tech_variance": control["joint_z_tech"].var(dim=0, unbiased=False).mean(),
            "z_bio_z_tech_cross_covariance_abs_mean": _cross_covariance_abs_mean(control["joint_z_bio"], control["joint_z_tech"]),
        }
        if "rna_z_bio" in control:
            outputs["rna_z_bio"] = control["rna_z_bio"]
            outputs["rna_z_tech"] = control["rna_z_tech"]
            outputs["rna_condition_state"] = control["rna_z_bio"]
        if "image_z_bio" in control:
            outputs["image_z_bio"] = control["image_z_bio"]
            outputs["image_z_tech"] = control["image_z_tech"]
            outputs["image_condition_state"] = control["image_z_bio"]

        self._predict_task(
            outputs,
            "rna_program_jepa",
            predictor=self.intra_modal_predictor,
            context=target_context.get("rna_bio_tokens"),
            target=target.get("rna_program_bio_tokens"),
            task_id=0,
        )
        self._predict_task(
            outputs,
            "image_region_jepa",
            predictor=self.intra_modal_predictor,
            context=target_context.get("image_bio_tokens"),
            target=target.get("image_region_bio_tokens"),
            task_id=1,
        )
        self._predict_task(
            outputs,
            "rna_to_image_jepa",
            predictor=self.cross_modal_predictor,
            context=target_context.get("rna_bio_tokens"),
            target=target.get("image_condition_bio_prototypes"),
            task_id=2,
        )
        self._predict_task(
            outputs,
            "image_to_rna_jepa",
            predictor=self.cross_modal_predictor,
            context=target_context.get("image_bio_tokens"),
            target=target.get("rna_condition_bio_prototypes"),
            task_id=3,
        )
        self._predict_task(
            outputs,
            "transition_bio_jepa",
            predictor=self.transition_predictor,
            context=control["joint_z_bio"][:, None, :],
            target=target["joint_z_bio"][:, None, :],
            task_id=4,
            action=action,
            residual_source=control["joint_z_bio"][:, None, :],
        )

        for name in _BIOTECH_JEPA_TASKS:
            outputs.setdefault(f"{name}_pred", zero)
            outputs.setdefault(f"{name}_target", zero.detach())
            outputs.setdefault(f"{name}_available", torch.zeros((), dtype=torch.bool, device=device))
        if self.count_decoder is not None and batch.target_expression_values is not None:
            outputs["count_aux_logits"] = self.count_decoder(control["rna_z_bio"])
            outputs["count_aux_target"] = _masked_mean(batch.target_expression_values, batch.target_rna_bag_mask).detach()
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
        bio_head = self._head("rna_bio", target)
        tech_head = self._head("rna_tech", target)
        flat_gene_ids, flat_expression, batch, bag = _flatten_rna(gene_ids, expression_values)
        encoded = encoder(flat_gene_ids, flat_expression)
        cell_shared = projector(encoded.cell_embedding).reshape(batch, bag, -1)
        token_shared = projector(encoded.token_embeddings).reshape(batch, bag, encoded.token_embeddings.shape[1], -1)
        agg = aggregator(cell_shared, mask=bag_mask)
        summary = summary_encoder(_masked_mean(expression_values, bag_mask))
        rna_shared = _condition_state_with_mean(agg.bag_embedding, cell_shared, bag_mask, summary=summary)
        token_bio = _normalize(bio_head(token_shared))
        condition_prototypes = _normalize(bio_head(agg.prototypes + rna_shared[:, None, :]))
        return {
            "rna_shared_state": rna_shared,
            "rna_z_bio": _normalize(bio_head(rna_shared)),
            "rna_z_tech": tech_head(rna_shared),
            "rna_bio_tokens": torch.cat((_normalize(bio_head(rna_shared))[:, None, :], condition_prototypes), dim=1),
            "rna_program_bio_tokens": self._rna_program_tokens(token_bio),
            "rna_condition_bio_prototypes": condition_prototypes,
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
        bio_head = self._head("image_bio", target)
        tech_head = self._head("image_tech", target)
        flat_images, batch, bag = _flatten_images(images)
        encoded = encoder(flat_images)
        image_shared = projector(encoded.image_embedding).reshape(batch, bag, -1)
        patch_shared = projector(encoded.patch_embeddings).reshape(batch, bag, encoded.patch_embeddings.shape[1], -1)
        agg = aggregator(image_shared, mask=bag_mask)
        summary = summary_encoder(_masked_mean(images, bag_mask).reshape(batch, -1))
        image_shared_state = _condition_state_with_mean(agg.bag_embedding, image_shared, bag_mask, summary=summary)
        patch_bio = _normalize(bio_head(patch_shared))
        condition_prototypes = _normalize(bio_head(agg.prototypes + image_shared_state[:, None, :]))
        return {
            "image_shared_state": image_shared_state,
            "image_z_bio": _normalize(bio_head(image_shared_state)),
            "image_z_tech": tech_head(image_shared_state),
            "image_bio_tokens": torch.cat((_normalize(bio_head(image_shared_state))[:, None, :], condition_prototypes), dim=1),
            "image_region_bio_tokens": self._image_region_tokens(patch_bio),
            "image_condition_bio_prototypes": condition_prototypes,
        }

    def _head(self, name: str, target: bool) -> nn.Module:
        if not target:
            return getattr(self, f"{name}_head")
        return getattr(self, f"{name.replace('_bio', '_target_bio').replace('_tech', '_target_tech')}_head")

    def _fuse_joint(self, rna_state: torch.Tensor, image_state: torch.Tensor, *, target: bool) -> torch.Tensor:
        fuser = self.target_joint_condition_fuser if target else self.joint_condition_fuser
        fused = fuser(torch.cat((rna_state, image_state), dim=-1))
        return 0.5 * (rna_state + image_state) + 0.25 * fused

    def _rna_program_tokens(self, token_bio: torch.Tensor) -> torch.Tensor:
        _, _, genes, _ = token_bio.shape
        assignment = self._program_assignment(genes, token_bio.device)
        values = []
        for program in assignment.unique(sorted=True)[: self.config.num_rna_program_targets]:
            mask = assignment.eq(program)
            values.append(token_bio[:, :, mask, :].mean(dim=(1, 2)))
        while len(values) < self.config.num_rna_program_targets:
            values.append(values[-1] if values else token_bio.mean(dim=(1, 2)))
        return torch.stack(values[: self.config.num_rna_program_targets], dim=1)

    def _image_region_tokens(self, patch_bio: torch.Tensor) -> torch.Tensor:
        _, _, patches, _ = patch_bio.shape
        chunks = torch.arange(patches, device=patch_bio.device) % max(1, self.config.num_image_region_targets)
        values = []
        for region in range(self.config.num_image_region_targets):
            mask = chunks.eq(region)
            values.append(patch_bio[:, :, mask, :].mean(dim=(1, 2)))
        return torch.stack(values, dim=1)

    def _program_assignment(self, genes: int, device: torch.device) -> torch.Tensor:
        if self.config.gene_program_assignment:
            assignment = torch.as_tensor(self.config.gene_program_assignment, dtype=torch.long, device=device)[:genes]
            if assignment.numel() == genes:
                return assignment
        return torch.arange(genes, device=device) % max(1, self.config.num_rna_program_targets)

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
        context: torch.Tensor | None,
        target: torch.Tensor | None,
        task_id: int,
        action: torch.Tensor | None = None,
        residual_source: torch.Tensor | None = None,
    ) -> None:
        if context is None or target is None:
            return
        queries = self._queries(target.shape[0], target.shape[1], task_id=task_id, device=target.device)
        prediction = predictor(context, queries, action_tokens=action)
        if residual_source is not None and action is not None:
            prediction = residual_source + 0.25 * prediction + 0.10 * action[:, None, :]
        outputs[f"{name}_pred"] = prediction
        outputs[f"{name}_target"] = target.detach()
        outputs[f"{name}_available"] = torch.ones((), dtype=torch.bool, device=target.device)


_BIOTECH_JEPA_TASKS = (
    "rna_program_jepa",
    "image_region_jepa",
    "rna_to_image_jepa",
    "image_to_rna_jepa",
    "transition_bio_jepa",
)


def _branch_head(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(output_dim),
        nn.Linear(output_dim, output_dim),
    )


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


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1)


def _cross_covariance_abs_mean(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if left.shape[0] < 2:
        return torch.zeros((), device=left.device, dtype=left.dtype)
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_centered = left_centered / torch.sqrt(left_centered.var(dim=0, unbiased=False, keepdim=True) + eps)
    right_centered = right_centered / torch.sqrt(right_centered.var(dim=0, unbiased=False, keepdim=True) + eps)
    return (left_centered.T @ right_centered / left.shape[0]).abs().mean()
