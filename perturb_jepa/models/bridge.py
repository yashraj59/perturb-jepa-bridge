from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from perturb_jepa.models.common import MLP, gradient_reverse
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.adversary import BatchAdversary
from perturb_jepa.models.bag_aggregator import MultiPrototypeBagAggregator
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
        self.rna_teacher_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_teacher_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        self.rna_teacher_projection.load_state_dict(self.rna_projection.state_dict())
        self.image_teacher_projection.load_state_dict(self.image_projection.state_dict())
        for module in (self.rna_teacher_projection, self.image_teacher_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.rna_bag_aggregator = MultiPrototypeBagAggregator(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.image_bag_aggregator = MultiPrototypeBagAggregator(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
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
        self.image_prototype_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.shared_dim,
            depth=2,
            dropout=config.dropout,
        )

    @torch.no_grad()
    def update_teachers(self, *, decay: float = 0.996) -> None:
        update_ema_teacher(self.rna_encoder, self.rna_teacher, decay=decay)
        update_ema_teacher(self.image_encoder, self.image_teacher, decay=decay)
        update_ema_teacher(self.rna_projection, self.rna_teacher_projection, decay=decay)
        update_ema_teacher(self.image_projection, self.image_teacher_projection, decay=decay)

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
            rna_state = self.state_head(rna_shared)
            rna_response = self.response_head(rna_shared)
            with torch.no_grad():
                rna_teacher = self.rna_teacher(flat_gene_ids, flat_expression_values, token_mask=None)
                rna_teacher_instances = self.rna_teacher_projection(rna_teacher.cell_embedding).reshape(*rna_bag_shape, -1)
                rna_teacher_aggregated = self.rna_bag_aggregator(rna_teacher_instances, mask=rna_bag_mask)
                rna_teacher_shared = rna_teacher_aggregated.bag_embedding
            rna_tokens = rna.token_embeddings.reshape(*rna_batch_shape, *rna.token_embeddings.shape[1:])
            rna_reconstruction = rna.reconstruction.reshape(*rna_batch_shape, rna.reconstruction.shape[-1])
            outputs.update(
                {
                    "rna_tokens": rna_tokens,
                    "rna_reconstruction": rna_reconstruction,
                    "rna_instance_shared": rna_instance_shared,
                    "rna_prototypes": rna_aggregated.prototypes,
                    "rna_attention": rna_aggregated.attention,
                    "rna_shared": rna_shared,
                    "rna_teacher_shared": rna_teacher_shared,
                    "rna_state": rna_state,
                    "rna_response": rna_response,
                    "rna_perturbation_logits": self.perturbation_classifier(rna_response),
                    "rna_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(rna_state, scale=self.config.adversary_scale)
                    ),
                    "rna_batch_logits": self.batch_adversary(rna_state, scale=self.config.adversary_scale),
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
            image_state = self.state_head(image_shared)
            image_response = self.response_head(image_shared)
            with torch.no_grad():
                image_teacher = self.image_teacher(flat_images, patch_mask=None)
                image_teacher_instances = self.image_teacher_projection(image_teacher.image_embedding).reshape(*image_bag_shape, -1)
                image_teacher_aggregated = self.image_bag_aggregator(image_teacher_instances, mask=image_bag_mask)
                image_teacher_shared = image_teacher_aggregated.bag_embedding
            if image_is_bagged:
                image_patches = image.patch_embeddings.reshape(*image_bag_shape, *image.patch_embeddings.shape[1:])
                image_reconstruction = image.patch_reconstruction.reshape(
                    *image_bag_shape,
                    *image.patch_reconstruction.shape[1:],
                )
            else:
                image_patches = image.patch_embeddings
                image_reconstruction = image.patch_reconstruction
            outputs.update(
                {
                    "image_patches": image_patches,
                    "image_patch_reconstruction": image_reconstruction,
                    "image_instance_shared": image_instance_shared,
                    "image_prototypes": image_aggregated.prototypes,
                    "image_attention": image_aggregated.attention,
                    "image_shared": image_shared,
                    "image_teacher_shared": image_teacher_shared,
                    "image_state": image_state,
                    "image_response": image_response,
                    "image_perturbation_logits": self.perturbation_classifier(image_response),
                    "image_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(image_state, scale=self.config.adversary_scale)
                    ),
                    "image_batch_logits": self.batch_adversary(image_state, scale=self.config.adversary_scale),
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
                    "counterfactual_rna": self.rna_distribution_decoder(z_counterfactual),
                    "counterfactual_image": self.image_prototype_decoder(z_counterfactual),
                }
            )

        return outputs
