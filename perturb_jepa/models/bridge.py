from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.models.common import MLP, gradient_reverse
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.adversary import BatchAdversary
from perturb_jepa.models.bag_aggregator import MeanBagAggregator, MultiPrototypeBagAggregator
from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig


@dataclass(frozen=True)
class PerturbJEPABridgeConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
    num_bag_prototypes: int = 8
    dropout: float = 0.1
    adversary_scale: float = 1.0
    bag_aggregator: str = "attention"
    rna_condition_readout: str = "encoder"
    rna_pseudobulk_normalize: bool = True
    image_condition_readout: str = "encoder"
    image_raw_normalize: bool = True
    counterfactual_rna_residual: bool = False
    counterfactual_rna_program_factorized: bool = False
    counterfactual_rna_num_programs: int = 0
    counterfactual_rna_program_assignment: tuple[int, ...] = ()
    counterfactual_rna_within_program_residual: bool = False
    counterfactual_rna_program_conditioned: bool = False
    counterfactual_rna_program_metadata_context: bool = False
    counterfactual_rna_program_decoder_depth: int = 2


class PerturbJEPABridge(nn.Module):
    def __init__(self, config: PerturbJEPABridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_encoder = RNAEncoder(config.rna)
        self.image_encoder = ImageEncoder(config.image)
        self.rna_teacher = make_ema_teacher(self.rna_encoder)
        self.image_teacher = make_ema_teacher(self.image_encoder)
        self.perturbation_encoder = PerturbationEncoder(config.perturbation)

        self.rna_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        image_raw_dim = config.image.in_channels * config.image.image_size * config.image.image_size
        self.image_raw_projection = nn.Sequential(
            nn.LayerNorm(image_raw_dim),
            nn.Linear(image_raw_dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.image_raw_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.dim),
            nn.Linear(config.rna.dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.max_genes),
            nn.Linear(config.rna.max_genes, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_teacher_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_teacher_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        self.rna_teacher_projection.load_state_dict(self.rna_projection.state_dict())
        self.image_teacher_projection.load_state_dict(self.image_projection.state_dict())
        for module in (self.rna_teacher_projection, self.image_teacher_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        aggregator_cls = self._bag_aggregator_cls(config.bag_aggregator)
        self.rna_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.image_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.rna_teacher_bag_aggregator = make_ema_teacher(self.rna_bag_aggregator)
        self.image_teacher_bag_aggregator = make_ema_teacher(self.image_bag_aggregator)
        self.rna_jepa_predictor = MLP(config.rna.dim, config.rna.dim, config.rna.dim, depth=2, dropout=config.dropout)
        self.image_jepa_predictor = MLP(
            config.image.dim,
            config.image.dim,
            config.image.dim,
            depth=2,
            dropout=config.dropout,
        )
        self.state_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.response_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.perturbation_classifier = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.state_perturbation_adversary = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.batch_adversary = BatchAdversary(
            config.shared_dim,
            config.perturbation.num_batches,
            hidden_dim=config.shared_dim,
            dropout=config.dropout,
            scale=config.adversary_scale,
        )
        self.delta_gate = MLP(
            config.shared_dim + config.perturbation.dim,
            config.shared_dim,
            config.shared_dim,
            depth=3,
            dropout=config.dropout,
        )
        self.delta_effect = nn.Linear(config.perturbation.dim, config.shared_dim)
        self.rna_distribution_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.rna.vocab_size,
            depth=2,
            dropout=config.dropout,
        )
        if config.counterfactual_rna_program_factorized:
            if config.counterfactual_rna_num_programs <= 0:
                raise ValueError("counterfactual_rna_num_programs must be positive for program-factorized decoding")
            assignment = torch.as_tensor(config.counterfactual_rna_program_assignment, dtype=torch.long)
            if assignment.numel() != config.rna.vocab_size:
                raise ValueError("counterfactual_rna_program_assignment length must match RNA vocab size")
            if int(assignment.min().item()) < 0 or int(assignment.max().item()) >= config.counterfactual_rna_num_programs:
                raise ValueError("counterfactual_rna_program_assignment contains an out-of-range program id")
            program_decoder_input_dim = config.shared_dim
            if config.counterfactual_rna_program_conditioned:
                program_decoder_input_dim += config.counterfactual_rna_num_programs
            if config.counterfactual_rna_program_metadata_context:
                program_decoder_input_dim += config.perturbation.num_perturbations + config.perturbation.num_cell_lines + 2
            self.counterfactual_program_decoder = MLP(
                program_decoder_input_dim,
                config.shared_dim,
                config.counterfactual_rna_num_programs,
                depth=config.counterfactual_rna_program_decoder_depth,
                dropout=config.dropout,
            )
            self.register_buffer("counterfactual_rna_program_assignment", assignment, persistent=True)
        self.image_prototype_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.shared_dim,
            depth=2,
            dropout=config.dropout,
        )

    @staticmethod
    def _bag_aggregator_cls(name: str):
        if name == "attention":
            return MultiPrototypeBagAggregator
        if name == "mean":
            return MeanBagAggregator
        raise ValueError(f"unsupported bag_aggregator: {name}")

    @torch.no_grad()
    def update_teachers(self, *, decay: float = 0.996) -> None:
        update_ema_teacher(self.rna_encoder, self.rna_teacher, decay=decay)
        update_ema_teacher(self.image_encoder, self.image_teacher, decay=decay)
        update_ema_teacher(self.rna_projection, self.rna_teacher_projection, decay=decay)
        update_ema_teacher(self.image_projection, self.image_teacher_projection, decay=decay)
        update_ema_teacher(self.rna_bag_aggregator, self.rna_teacher_bag_aggregator, decay=decay)
        update_ema_teacher(self.image_bag_aggregator, self.image_teacher_bag_aggregator, decay=decay)

    @staticmethod
    def _set_eval_temporarily(modules: tuple[nn.Module, ...]):
        class _TeacherEvalContext:
            def __init__(self, modules: tuple[nn.Module, ...]) -> None:
                self.modules = modules
                self.states = [module.training for module in modules]

            def __enter__(self) -> None:
                for module in self.modules:
                    module.eval()

            def __exit__(self, exc_type, exc, tb) -> None:
                for module, state in zip(self.modules, self.states, strict=True):
                    module.train(state)

        return _TeacherEvalContext(modules)

    def encode_perturbation(
        self,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        *,
        descriptor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.perturbation_encoder(
            perturbation_id,
            perturbation_type_id,
            cell_line_id,
            batch_id,
            dose,
            time,
            descriptor=descriptor,
        )

    def predict_delta(self, z_state: torch.Tensor, perturbation_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.delta_gate(torch.cat((z_state, perturbation_embedding), dim=-1)))
        base_delta = self.delta_effect(perturbation_embedding)
        delta = gate * base_delta
        return delta, gate, base_delta

    def forward(
        self,
        *,
        gene_ids: torch.Tensor | None = None,
        expression_values: torch.Tensor | None = None,
        rna_token_mask: torch.Tensor | None = None,
        rna_bag_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_patch_mask: torch.Tensor | None = None,
        image_bag_mask: torch.Tensor | None = None,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        descriptor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        perturbation = self.encode_perturbation(
            perturbation_id,
            perturbation_type_id,
            cell_line_id,
            batch_id,
            dose,
            time,
            descriptor=descriptor,
        )
        outputs["perturbation_embedding"] = perturbation

        shared_for_state: torch.Tensor | None = None
        rna_condition_for_counterfactual: torch.Tensor | None = None
        if gene_ids is not None and expression_values is not None:
            rna_batch_shape = gene_ids.shape[:-1]
            if gene_ids.ndim == 2:
                flat_gene_ids = gene_ids
                flat_expression_values = expression_values
                flat_token_mask = rna_token_mask
                rna_bag_shape = (gene_ids.shape[0], 1)
            elif gene_ids.ndim == 3:
                if expression_values.shape != gene_ids.shape:
                    raise ValueError("bagged gene_ids and expression_values must have matching shapes")
                rna_bag_shape = gene_ids.shape[:2]
                flat_gene_ids = gene_ids.reshape(-1, gene_ids.shape[-1])
                flat_expression_values = expression_values.reshape(-1, expression_values.shape[-1])
                flat_token_mask = None if rna_token_mask is None else rna_token_mask.reshape(-1, rna_token_mask.shape[-1])
            else:
                raise ValueError("gene_ids must have shape [batch, genes] or [batch, bag, genes]")

            rna = self.rna_encoder(flat_gene_ids, flat_expression_values, token_mask=flat_token_mask)
            rna_instance_shared = self.rna_projection(rna.cell_embedding).reshape(*rna_bag_shape, -1)
            rna_aggregated = self.rna_bag_aggregator(rna_instance_shared, mask=rna_bag_mask)
            rna_shared = rna_aggregated.bag_embedding
            rna_pseudobulk_shared = self._rna_pseudobulk_shared(gene_ids, expression_values)
            rna_raw_pseudobulk_shared = self._rna_raw_pseudobulk_shared(expression_values)
            rna_raw_linear_shared = self._rna_raw_linear_pseudobulk_shared(expression_values)
            rna_distilled_linear_shared = self._rna_distilled_linear_pseudobulk_shared(expression_values)
            rna_distilled_residual_shared = rna_distilled_linear_shared + (
                self.rna_distilled_residual_scale * rna_raw_pseudobulk_shared
            )
            rna_condition_for_counterfactual = self._rna_condition_values(expression_values)
            if self.config.rna_condition_readout == "encoder":
                pass
            elif self.config.rna_condition_readout == "pseudobulk":
                rna_shared = rna_pseudobulk_shared
            elif self.config.rna_condition_readout == "encoder_plus_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_pseudobulk_shared, dim=-1)
            elif self.config.rna_condition_readout == "raw_pseudobulk":
                rna_shared = rna_raw_pseudobulk_shared
            elif self.config.rna_condition_readout == "encoder_plus_raw_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_raw_pseudobulk_shared, dim=-1)
            elif self.config.rna_condition_readout == "raw_linear_pseudobulk":
                rna_shared = rna_raw_linear_shared
            elif self.config.rna_condition_readout == "encoder_plus_raw_linear_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_raw_linear_shared, dim=-1)
            else:
                raise ValueError(f"unsupported rna_condition_readout: {self.config.rna_condition_readout}")
            rna_state = self.state_head(rna_shared)
            rna_response = self.response_head(rna_shared)
            with torch.no_grad(), self._set_eval_temporarily(
                (self.rna_teacher, self.rna_teacher_projection, self.rna_teacher_bag_aggregator)
            ):
                rna_teacher = self.rna_teacher(flat_gene_ids, flat_expression_values, token_mask=None)
                rna_teacher_instances = self.rna_teacher_projection(rna_teacher.cell_embedding).reshape(*rna_bag_shape, -1)
                rna_teacher_aggregated = self.rna_teacher_bag_aggregator(rna_teacher_instances, mask=rna_bag_mask)
                rna_teacher_shared = rna_teacher_aggregated.bag_embedding.detach()
            rna_tokens = rna.token_embeddings.reshape(*rna_batch_shape, *rna.token_embeddings.shape[1:])
            rna_teacher_tokens = rna_teacher.token_embeddings.reshape(*rna_batch_shape, *rna_teacher.token_embeddings.shape[1:])
            rna_token_prediction = self.rna_jepa_predictor(rna.token_embeddings).reshape(
                *rna_batch_shape,
                *rna.token_embeddings.shape[1:],
            )
            rna_reconstruction = rna.reconstruction.reshape(*rna_batch_shape, rna.reconstruction.shape[-1])
            outputs.update(
                {
                    "rna_tokens": rna_tokens,
                    "rna_teacher_tokens": rna_teacher_tokens.detach(),
                    "rna_token_prediction": rna_token_prediction,
                    "rna_reconstruction": rna_reconstruction,
                    "rna_instance_shared": rna_instance_shared,
                    "rna_prototypes": rna_aggregated.prototypes,
                    "rna_attention": rna_aggregated.attention,
                    "rna_pseudobulk_shared": rna_pseudobulk_shared,
                    "rna_raw_pseudobulk_shared": rna_raw_pseudobulk_shared,
                    "rna_raw_linear_shared": rna_raw_linear_shared,
                    "rna_distilled_shared": rna_raw_pseudobulk_shared,
                    "rna_distilled_linear_shared": rna_distilled_linear_shared,
                    "rna_distilled_residual_shared": rna_distilled_residual_shared,
                    "rna_shared": rna_shared,
                    "rna_retrieval": rna_shared,
                    "rna_teacher_shared": rna_teacher_shared,
                    "rna_state": rna_state,
                    "rna_response": rna_response,
                    "rna_perturbation_logits": self.perturbation_classifier(rna_response),
                    "rna_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(rna_state, scale=self.config.adversary_scale)
                    ),
                    "rna_batch_logits": self.batch_adversary(rna_shared, scale=self.config.adversary_scale),
                }
            )
            shared_for_state = rna_shared

        if images is not None:
            if images.ndim == 4:
                image_is_bagged = False
                image_bag_shape = (images.shape[0], 1)
                flat_images = images
                flat_patch_mask = image_patch_mask
            elif images.ndim == 5:
                image_is_bagged = True
                image_bag_shape = images.shape[:2]
                flat_images = images.reshape(-1, *images.shape[-3:])
                flat_patch_mask = None if image_patch_mask is None else image_patch_mask.reshape(-1, image_patch_mask.shape[-1])
            else:
                raise ValueError("images must have shape [batch, channels, height, width] or [batch, bag, channels, height, width]")

            image = self.image_encoder(flat_images, patch_mask=flat_patch_mask)
            image_instance_shared = self.image_projection(image.image_embedding).reshape(*image_bag_shape, -1)
            image_aggregated = self.image_bag_aggregator(image_instance_shared, mask=image_bag_mask)
            image_shared = image_aggregated.bag_embedding
            image_raw_shared = self._image_raw_shared(images)
            image_raw_linear_shared = self._image_raw_linear_shared(images)
            image_distilled_linear_shared = self._image_distilled_linear_shared(images)
            image_distilled_residual_shared = image_distilled_linear_shared + (
                self.image_distilled_residual_scale * image_raw_shared
            )
            if self.config.image_condition_readout == "encoder":
                pass
            elif self.config.image_condition_readout == "raw_pooled":
                image_shared = image_raw_shared
            elif self.config.image_condition_readout == "encoder_plus_raw_pooled":
                image_shared = F.normalize(image_shared + image_raw_shared, dim=-1)
            elif self.config.image_condition_readout == "raw_linear_pooled":
                image_shared = image_raw_linear_shared
            elif self.config.image_condition_readout == "encoder_plus_raw_linear_pooled":
                image_shared = F.normalize(image_shared + image_raw_linear_shared, dim=-1)
            else:
                raise ValueError(f"unsupported image_condition_readout: {self.config.image_condition_readout}")
            image_state = self.state_head(image_shared)
            image_response = self.response_head(image_shared)
            with torch.no_grad(), self._set_eval_temporarily(
                (self.image_teacher, self.image_teacher_projection, self.image_teacher_bag_aggregator)
            ):
                image_teacher = self.image_teacher(flat_images, patch_mask=None)
                image_teacher_instances = self.image_teacher_projection(image_teacher.image_embedding).reshape(*image_bag_shape, -1)
                image_teacher_aggregated = self.image_teacher_bag_aggregator(image_teacher_instances, mask=image_bag_mask)
                image_teacher_shared = image_teacher_aggregated.bag_embedding.detach()
            image_patch_prediction_flat = self.image_jepa_predictor(image.patch_embeddings)
            if image_is_bagged:
                image_patches = image.patch_embeddings.reshape(*image_bag_shape, *image.patch_embeddings.shape[1:])
                image_teacher_patches = image_teacher.patch_embeddings.reshape(
                    *image_bag_shape,
                    *image_teacher.patch_embeddings.shape[1:],
                )
                image_patch_prediction = image_patch_prediction_flat.reshape(
                    *image_bag_shape,
                    *image_patch_prediction_flat.shape[1:],
                )
                image_reconstruction = image.patch_reconstruction.reshape(
                    *image_bag_shape,
                    *image.patch_reconstruction.shape[1:],
                )
            else:
                image_patches = image.patch_embeddings
                image_teacher_patches = image_teacher.patch_embeddings
                image_patch_prediction = image_patch_prediction_flat
                image_reconstruction = image.patch_reconstruction
            outputs.update(
                {
                    "image_patches": image_patches,
                    "image_teacher_patches": image_teacher_patches.detach(),
                    "image_patch_prediction": image_patch_prediction,
                    "image_patch_reconstruction": image_reconstruction,
                    "image_instance_shared": image_instance_shared,
                    "image_prototypes": image_aggregated.prototypes,
                    "image_attention": image_aggregated.attention,
                    "image_raw_shared": image_raw_shared,
                    "image_raw_linear_shared": image_raw_linear_shared,
                    "image_distilled_shared": image_raw_shared,
                    "image_distilled_linear_shared": image_distilled_linear_shared,
                    "image_distilled_residual_shared": image_distilled_residual_shared,
                    "image_shared": image_shared,
                    "image_retrieval": image_shared,
                    "image_teacher_shared": image_teacher_shared,
                    "image_state": image_state,
                    "image_response": image_response,
                    "image_perturbation_logits": self.perturbation_classifier(image_response),
                    "image_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(image_state, scale=self.config.adversary_scale)
                    ),
                    "image_batch_logits": self.batch_adversary(image_shared, scale=self.config.adversary_scale),
                }
            )
            shared_for_state = image_shared if shared_for_state is None else 0.5 * (shared_for_state + image_shared)

        if shared_for_state is not None:
            z_state = self.state_head(shared_for_state)
            z_response = self.response_head(shared_for_state)
            delta, delta_gate, delta_base = self.predict_delta(z_state, perturbation)
            z_counterfactual = z_state + delta
            reverse_delta, reverse_gate, reverse_base = self.predict_delta(z_counterfactual, -perturbation)
            z_cycle = z_counterfactual + reverse_delta
            counterfactual_rna_delta, counterfactual_rna_aux = self._counterfactual_rna_delta(
                z_counterfactual,
                rna_condition_for_counterfactual,
                perturbation_id=perturbation_id,
                cell_line_id=cell_line_id,
                dose=dose,
                time=time,
            )
            counterfactual_rna = counterfactual_rna_delta
            if self.config.counterfactual_rna_residual and rna_condition_for_counterfactual is not None:
                counterfactual_rna = _match_last_dim(rna_condition_for_counterfactual, counterfactual_rna_delta.shape[-1])
                counterfactual_rna = counterfactual_rna + counterfactual_rna_delta
            outputs.update(
                {
                    "z_state": z_state,
                    "z_response": z_response,
                    "z_counterfactual": z_counterfactual,
                    "counterfactual_delta": delta,
                    "counterfactual_gate": delta_gate,
                    "counterfactual_base_delta": delta_base,
                    "cycle_reconstruction": z_cycle,
                    "cycle_delta": reverse_delta,
                    "cycle_gate": reverse_gate,
                    "cycle_base_delta": reverse_base,
                    "counterfactual_rna_delta": counterfactual_rna_delta,
                    "counterfactual_rna": counterfactual_rna,
                    "counterfactual_image": self.image_prototype_decoder(z_counterfactual),
                    **counterfactual_rna_aux,
                }
            )

        return outputs

    def _rna_pseudobulk_shared(self, gene_ids: torch.Tensor, expression_values: torch.Tensor) -> torch.Tensor:
        if gene_ids.ndim == 2 and expression_values.ndim == 2:
            condition_gene_ids = gene_ids
            condition_values = expression_values
        elif gene_ids.ndim == 3 and expression_values.ndim == 3:
            condition_gene_ids = gene_ids[:, 0, :]
            condition_values = expression_values.mean(dim=1)
        else:
            raise ValueError("gene_ids and expression_values must both be 2D or 3D")
        gene_embedding = self.rna_encoder.gene_embedding(condition_gene_ids)
        value_embedding = self.rna_encoder.value_embedding(condition_values.unsqueeze(-1))
        projected = self.rna_pseudobulk_projection((gene_embedding + value_embedding).mean(dim=1))
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_raw_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_raw_pseudobulk_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_raw_linear_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_raw_linear_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_distilled_linear_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_distilled_linear_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_condition_values(self, expression_values: torch.Tensor) -> torch.Tensor:
        if expression_values.ndim == 2:
            condition_values = expression_values
        elif expression_values.ndim == 3:
            condition_values = expression_values.mean(dim=1)
        else:
            raise ValueError("expression_values must be 2D or 3D")
        expected_genes = self.config.rna.max_genes
        if condition_values.shape[-1] > expected_genes:
            raise ValueError(f"raw pseudobulk input has {condition_values.shape[-1]} genes, expected at most {expected_genes}")
        if condition_values.shape[-1] < expected_genes:
            condition_values = F.pad(condition_values, (0, expected_genes - condition_values.shape[-1]))
        return condition_values

    def _image_raw_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_raw_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_raw_linear_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_raw_linear_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_distilled_linear_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_distilled_linear_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_condition_values(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            condition_images = images
        elif images.ndim == 5:
            condition_images = images.mean(dim=1)
        else:
            raise ValueError("images must be 4D or 5D")
        return condition_images.reshape(condition_images.shape[0], -1)

    def _counterfactual_rna_delta(
        self,
        z_counterfactual: torch.Tensor,
        source_rna: torch.Tensor | None,
        perturbation_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.counterfactual_rna_program_factorized:
            return self.rna_distribution_decoder(z_counterfactual), {}
        assignment = self.counterfactual_rna_program_assignment
        decoder_input = z_counterfactual
        source_program_context = torch.zeros(
            z_counterfactual.shape[0],
            self.config.counterfactual_rna_num_programs,
            device=z_counterfactual.device,
            dtype=z_counterfactual.dtype,
        )
        if self.config.counterfactual_rna_program_conditioned:
            if source_rna is not None:
                source_program_context = _program_means(
                    _match_last_dim(source_rna, assignment.numel()),
                    assignment,
                    num_programs=self.config.counterfactual_rna_num_programs,
                )
            decoder_input = torch.cat((z_counterfactual, source_program_context), dim=-1)
        metadata_context = torch.zeros(
            z_counterfactual.shape[0],
            self.config.perturbation.num_perturbations + self.config.perturbation.num_cell_lines + 2,
            device=z_counterfactual.device,
            dtype=z_counterfactual.dtype,
        )
        if self.config.counterfactual_rna_program_metadata_context:
            metadata_context = torch.cat(
                (
                    F.one_hot(perturbation_id, num_classes=self.config.perturbation.num_perturbations).to(
                        dtype=z_counterfactual.dtype
                    ),
                    F.one_hot(cell_line_id, num_classes=self.config.perturbation.num_cell_lines).to(
                        dtype=z_counterfactual.dtype
                    ),
                    dose.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
                    time.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
            decoder_input = torch.cat((decoder_input, metadata_context), dim=-1)
        program_delta = self.counterfactual_program_decoder(decoder_input)
        program_gene_delta = program_delta.index_select(dim=1, index=assignment)
        within_program_residual = torch.zeros_like(program_gene_delta)
        if self.config.counterfactual_rna_within_program_residual:
            raw_residual = _match_last_dim(self.rna_distribution_decoder(z_counterfactual), program_gene_delta.shape[-1])
            within_program_residual = raw_residual - _program_gene_means(raw_residual, assignment)
        delta = program_gene_delta + within_program_residual
        return delta, {
            "counterfactual_rna_program_delta": program_delta,
            "counterfactual_rna_program_gene_delta": program_gene_delta,
            "counterfactual_rna_within_program_residual": within_program_residual,
            "counterfactual_rna_source_program_context": source_program_context,
            "counterfactual_rna_metadata_context": metadata_context,
            "counterfactual_rna_program_decoder_input": decoder_input,
        }


def _match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    if values.shape[-1] == target_dim:
        return values
    if values.shape[-1] > target_dim:
        return values[..., :target_dim]
    return F.pad(values, (0, target_dim - values.shape[-1]))


def _program_means(values: torch.Tensor, assignment: torch.Tensor, *, num_programs: int | None = None) -> torch.Tensor:
    if values.shape[-1] != assignment.numel():
        raise ValueError("program assignment length must match values last dimension")
    num_programs = int(num_programs or (int(assignment.max().item()) + 1))
    membership = F.one_hot(assignment, num_classes=num_programs).to(dtype=values.dtype, device=values.device)
    counts = membership.sum(dim=0).clamp_min(1.0)
    return values @ membership / counts


def _program_gene_means(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    program_means = _program_means(values, assignment)
    return program_means.index_select(dim=1, index=assignment)
