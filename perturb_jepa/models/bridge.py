from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from perturb_jepa.models.common import MLP, gradient_reverse
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig


@dataclass(frozen=True)
class PerturbJEPABridgeConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
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

        self.rna_projection = MLP(config.rna.dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.image_projection = MLP(config.image.dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.rna_teacher_projection = MLP(config.rna.dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.image_teacher_projection = MLP(config.image.dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.rna_teacher_projection.load_state_dict(self.rna_projection.state_dict())
        self.image_teacher_projection.load_state_dict(self.image_projection.state_dict())
        for module in (self.rna_teacher_projection, self.image_teacher_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.state_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.response_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.perturbation_classifier = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.batch_adversary = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_batches,
            depth=2,
            dropout=config.dropout,
        )
        self.counterfactual_delta = MLP(
            config.shared_dim + config.perturbation.dim,
            config.shared_dim,
            config.shared_dim,
            depth=3,
            dropout=config.dropout,
        )
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

    def forward(
        self,
        *,
        gene_ids: torch.Tensor | None = None,
        expression_values: torch.Tensor | None = None,
        rna_token_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_patch_mask: torch.Tensor | None = None,
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
            rna = self.rna_encoder(gene_ids, expression_values, token_mask=rna_token_mask)
            rna_shared = self.rna_projection(rna.cell_embedding)
            rna_state = self.state_head(rna_shared)
            rna_response = self.response_head(rna_shared)
            with torch.no_grad():
                rna_teacher = self.rna_teacher(gene_ids, expression_values, token_mask=None)
                rna_teacher_shared = self.rna_teacher_projection(rna_teacher.cell_embedding)
            outputs.update(
                {
                    "rna_tokens": rna.token_embeddings,
                    "rna_reconstruction": rna.reconstruction,
                    "rna_shared": rna_shared,
                    "rna_teacher_shared": rna_teacher_shared,
                    "rna_state": rna_state,
                    "rna_response": rna_response,
                    "rna_perturbation_logits": self.perturbation_classifier(rna_response),
                    "rna_batch_logits": self.batch_adversary(
                        gradient_reverse(rna_state, scale=self.config.adversary_scale)
                    ),
                }
            )
            shared_for_state = rna_shared

        if images is not None:
            image = self.image_encoder(images, patch_mask=image_patch_mask)
            image_shared = self.image_projection(image.image_embedding)
            image_state = self.state_head(image_shared)
            image_response = self.response_head(image_shared)
            with torch.no_grad():
                image_teacher = self.image_teacher(images, patch_mask=None)
                image_teacher_shared = self.image_teacher_projection(image_teacher.image_embedding)
            outputs.update(
                {
                    "image_patches": image.patch_embeddings,
                    "image_patch_reconstruction": image.patch_reconstruction,
                    "image_shared": image_shared,
                    "image_teacher_shared": image_teacher_shared,
                    "image_state": image_state,
                    "image_response": image_response,
                    "image_perturbation_logits": self.perturbation_classifier(image_response),
                    "image_batch_logits": self.batch_adversary(
                        gradient_reverse(image_state, scale=self.config.adversary_scale)
                    ),
                }
            )
            shared_for_state = image_shared if shared_for_state is None else 0.5 * (shared_for_state + image_shared)

        if shared_for_state is not None:
            z_state = self.state_head(shared_for_state)
            z_response = self.response_head(shared_for_state)
            delta = self.counterfactual_delta(torch.cat((z_state, perturbation), dim=-1))
            z_counterfactual = z_state + delta
            outputs.update(
                {
                    "z_state": z_state,
                    "z_response": z_response,
                    "z_counterfactual": z_counterfactual,
                    "counterfactual_delta": delta,
                    "counterfactual_rna": self.rna_distribution_decoder(z_counterfactual),
                    "counterfactual_image": self.image_prototype_decoder(z_counterfactual),
                }
            )

        return outputs
