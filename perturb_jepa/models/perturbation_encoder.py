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
    dropout: float = 0.1


class PerturbationEncoder(nn.Module):
    def __init__(self, config: PerturbationEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.perturbation_embedding = nn.Embedding(config.num_perturbations, config.dim)
        self.type_embedding = nn.Embedding(config.num_types, config.dim)
        self.cell_line_embedding = nn.Embedding(config.num_cell_lines, config.dim)
        self.batch_embedding = nn.Embedding(config.num_batches, config.dim)
        numeric_dim = 2 + config.descriptor_dim
        self.numeric_mlp = MLP(numeric_dim, config.dim, config.dim, depth=2, dropout=config.dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.dim * 5),
            nn.Linear(config.dim * 5, config.dim),
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
            self.perturbation_embedding(perturbation_id),
            self.type_embedding(perturbation_type_id),
            self.cell_line_embedding(cell_line_id),
            self.batch_embedding(batch_id),
            self.numeric_mlp(numeric),
        )
        return self.fusion(torch.cat(pieces, dim=-1))
