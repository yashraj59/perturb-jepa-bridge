from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from perturb_jepa.models.common import MLP


@dataclass(frozen=True)
class PerturbationEncoderConfig:
    num_perturbations: int
    num_types: int
    num_cell_lines: int
    num_batches: int
    dim: int = 128
    descriptor_dim: int = 0
    perturbation_feature_mode: str = "lookup"
    perturbation_feature_scale_init: float = 0.0
    dropout: float = 0.1


class PerturbationEncoder(nn.Module):
    def __init__(self, config: PerturbationEncoderConfig) -> None:
        super().__init__()
        if config.perturbation_feature_mode not in {"lookup", "residual", "feature_only"}:
            raise ValueError("perturbation_feature_mode must be 'lookup', 'residual', or 'feature_only'")
        if config.perturbation_feature_mode != "lookup" and config.descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive when using feature-conditioned perturbation modes")
        self.config = config
        self.perturbation_embedding = nn.Embedding(config.num_perturbations, config.dim)
        self.perturbation_feature_mlp = (
            MLP(config.descriptor_dim, config.dim, config.dim, depth=2, dropout=config.dropout)
            if config.descriptor_dim > 0
            else None
        )
        self.perturbation_feature_scale = nn.Parameter(torch.tensor(float(config.perturbation_feature_scale_init)))
        self.type_embedding = nn.Embedding(config.num_types, config.dim)
        self.cell_line_embedding = nn.Embedding(config.num_cell_lines, config.dim)
        numeric_dim = 2 + config.descriptor_dim
        self.numeric_mlp = MLP(numeric_dim, config.dim, config.dim, depth=2, dropout=config.dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.dim * 4),
            nn.Linear(config.dim * 4, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )

    def forward(
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
        if descriptor is None:
            descriptor = torch.zeros(
                perturbation_id.shape[0],
                self.config.descriptor_dim,
                device=perturbation_id.device,
                dtype=dose.dtype,
            )
        numeric = torch.cat((dose.unsqueeze(-1), time.unsqueeze(-1), descriptor), dim=-1)
        pieces = (
            self._perturbation_identity(perturbation_id, descriptor),
            self.type_embedding(perturbation_type_id),
            self.cell_line_embedding(cell_line_id),
            self.numeric_mlp(numeric),
        )
        return self.fusion(torch.cat(pieces, dim=-1))

    def _perturbation_identity(self, perturbation_id: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        lookup = self.perturbation_embedding(perturbation_id)
        if self.config.perturbation_feature_mode == "lookup":
            return lookup
        if self.perturbation_feature_mlp is None:
            raise RuntimeError("feature-conditioned perturbation mode requires perturbation_feature_mlp")
        feature = self.perturbation_feature_mlp(descriptor)
        if self.config.perturbation_feature_mode == "residual":
            return lookup + self.perturbation_feature_scale * feature
        return feature
