from __future__ import annotations

import math

import torch
from torch import nn

from perturb_jepa.models.common import MLP


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        return _GradientReverse.apply(x, self.scale if scale is None else float(scale))


def gradient_reversal_ramp(step: int, max_steps: int, *, max_scale: float = 1.0) -> float:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
    return float(max_scale) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


class BatchAdversary(nn.Module):
    """Predict caller-provided technical labels through gradient reversal."""

    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        *,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        hidden_dim = hidden_dim or input_dim
        self.reversal = GradientReversalLayer(scale=scale)
        self.classifier = MLP(input_dim, hidden_dim, num_batches, depth=depth, dropout=dropout)

    def forward(self, x: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
        return self.classifier(self.reversal(x, scale=scale))
