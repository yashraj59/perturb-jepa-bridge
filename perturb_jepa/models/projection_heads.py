from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class RNAProjectionHead(_ProjectionHead):
    """Projection head for RNA embeddings; intentionally accepts no metadata."""


class ImageProjectionHead(_ProjectionHead):
    """Projection head for image embeddings; intentionally accepts no metadata."""
