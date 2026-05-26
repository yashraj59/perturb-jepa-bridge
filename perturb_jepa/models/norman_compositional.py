from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.models.common import MLP


@dataclass(frozen=True)
class NormanFeatureBridgeConfig:
    gene_dim: int
    feature_dim: int
    hidden_dim: int = 64
    operator_rank: int = 0
    dropout: float = 0.0


class NormanFeatureBridge(nn.Module):
    """Lightweight RNA-only state + perturbation bridge for Norman Tier 1 screening."""

    def __init__(self, config: NormanFeatureBridgeConfig) -> None:
        super().__init__()
        if config.operator_rank < 0:
            raise ValueError("operator_rank must be non-negative")
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(config.gene_dim),
            nn.Linear(config.gene_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.perturbation_encoder = MLP(
            config.feature_dim,
            config.hidden_dim,
            config.hidden_dim,
            depth=2,
            dropout=config.dropout,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.gene_dim),
        )
        if config.operator_rank > 0:
            self.operator_u = nn.Linear(config.hidden_dim, config.operator_rank * config.hidden_dim)
            self.operator_v = nn.Linear(config.hidden_dim, config.operator_rank * config.hidden_dim)
            self.operator_out = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.gene_dim),
            )
            self.operator_gate = nn.Linear(config.hidden_dim * 2, config.gene_dim)

    def forward(self, control_expression: torch.Tensor, perturbation_features: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.state_encoder(control_expression)
        perturbation = self.perturbation_encoder(perturbation_features)
        additive_delta = self.delta_head(perturbation)
        operator_delta = torch.zeros_like(additive_delta)
        gate = torch.ones_like(additive_delta)
        if self.config.operator_rank > 0:
            batch = state.shape[0]
            rank = self.config.operator_rank
            u = self.operator_u(perturbation).reshape(batch, rank, self.config.hidden_dim)
            v = self.operator_v(perturbation).reshape(batch, rank, self.config.hidden_dim)
            scores = (v * state.unsqueeze(1)).sum(dim=-1)
            operator_hidden = (u * scores.unsqueeze(-1)).sum(dim=1)
            raw_operator = self.operator_out(operator_hidden)
            gate = torch.sigmoid(self.operator_gate(torch.cat((state, perturbation), dim=-1)))
            operator_delta = gate * raw_operator
        delta = additive_delta + operator_delta
        return {
            "prediction": control_expression + delta,
            "delta": delta,
            "additive_delta": additive_delta,
            "operator_delta": operator_delta,
            "operator_gate": gate,
            "state": state,
            "perturbation": perturbation,
        }


@dataclass(frozen=True)
class NormanInteractionConfig:
    gene_dim: int
    feature_dim: int
    hidden_dim: int = 64
    rank: int = 0
    dropout: float = 0.0


class NormanAdditiveInteraction(nn.Module):
    """Bounded residual interaction on top of a frozen single-perturbation additive delta."""

    def __init__(self, config: NormanInteractionConfig) -> None:
        super().__init__()
        if config.rank < 0:
            raise ValueError("rank must be non-negative")
        self.config = config
        if config.rank > 0:
            self.left = nn.Linear(config.feature_dim, config.rank, bias=False)
            self.right = nn.Linear(config.feature_dim, config.rank, bias=False)
            input_dim = config.feature_dim + config.rank
        else:
            input_dim = config.feature_dim
        self.interaction = MLP(
            input_dim,
            config.hidden_dim,
            config.gene_dim,
            depth=3,
            dropout=config.dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        additive_delta: torch.Tensor,
        perturbation_features: torch.Tensor,
        left_features: torch.Tensor | None = None,
        right_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = perturbation_features
        if self.config.rank > 0:
            if left_features is None or right_features is None:
                raise ValueError("ranked interaction requires left_features and right_features")
            bilinear = self.left(left_features) * self.right(right_features)
            features = torch.cat((perturbation_features, bilinear), dim=-1)
        raw_interaction = self.interaction(features)
        interaction = torch.sigmoid(self.logit_scale) * raw_interaction
        return {
            "delta": additive_delta + interaction,
            "additive_delta": additive_delta,
            "interaction_delta": interaction,
            "interaction_scale": torch.sigmoid(self.logit_scale),
        }


def norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return numerator.norm(dim=-1) / denominator.norm(dim=-1).clamp_min(eps)


def interaction_ratio_loss(interaction: torch.Tensor, additive: torch.Tensor, *, max_ratio: float = 0.5) -> torch.Tensor:
    return F.relu(norm_ratio(interaction, additive) - float(max_ratio)).pow(2).mean()
