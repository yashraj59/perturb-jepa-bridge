from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.distribution_losses import mmd_rbf_loss
from perturb_jepa.models.common import MLP


@dataclass
class CounterfactualResponseOutput:
    delta_mu: torch.Tensor
    delta_logvar: torch.Tensor
    treated_mu: torch.Tensor
    treated_logvar: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return ("delta_mu", "delta_logvar", "treated_mu", "treated_logvar")

    def items(self):
        return ((key, getattr(self, key)) for key in self.keys())


class PerturbationConditionEncoder(nn.Module):
    """Encode perturbation condition labels and optional descriptor features."""

    def __init__(
        self,
        *,
        num_conditions: int | None = None,
        num_perturbations: int | None = None,
        num_cell_lines: int | None = None,
        num_time_bins: int | None = None,
        num_moas: int | None = None,
        num_pathways: int | None = None,
        feature_dim: int = 0,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            num_conditions is None
            and num_perturbations is None
            and num_cell_lines is None
            and num_time_bins is None
            and num_moas is None
            and num_pathways is None
            and feature_dim <= 0
        ):
            raise ValueError("provide categorical vocab sizes, feature_dim, or both")
        self.num_conditions = num_conditions
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(num_conditions, output_dim) if num_conditions is not None else None
        self.perturbation_embedding = (
            nn.Embedding(num_perturbations, output_dim) if num_perturbations is not None else None
        )
        self.cell_line_embedding = nn.Embedding(num_cell_lines, output_dim) if num_cell_lines is not None else None
        self.time_embedding = nn.Embedding(num_time_bins, output_dim) if num_time_bins is not None else None
        self.moa_embedding = nn.Embedding(num_moas, output_dim) if num_moas is not None else None
        self.pathway_embedding = nn.Embedding(num_pathways, output_dim) if num_pathways is not None else None
        categorical_count = sum(
            module is not None
            for module in (
                self.embedding,
                self.perturbation_embedding,
                self.cell_line_embedding,
                self.time_embedding,
                self.moa_embedding,
                self.pathway_embedding,
            )
        )
        self.numeric_dim = feature_dim + 2
        mlp_input_dim = self.numeric_dim + categorical_count * output_dim
        self.projection = MLP(mlp_input_dim, hidden_dim, output_dim, depth=2, dropout=dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        condition_id: torch.Tensor | None = None,
        condition_features: torch.Tensor | None = None,
        *,
        perturbation_id: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        time_id: torch.Tensor | None = None,
        cell_line_id: torch.Tensor | None = None,
        moa_id: torch.Tensor | None = None,
        pathway_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if self.embedding is not None:
            if condition_id is None:
                raise ValueError("condition_id is required when num_conditions is configured")
            pieces.append(self.embedding(condition_id))
        if self.perturbation_embedding is not None:
            if perturbation_id is None:
                raise ValueError("perturbation_id is required when num_perturbations is configured")
            pieces.append(self.perturbation_embedding(perturbation_id))
        if self.cell_line_embedding is not None:
            if cell_line_id is None:
                raise ValueError("cell_line_id is required when num_cell_lines is configured")
            pieces.append(self.cell_line_embedding(cell_line_id))
        if self.time_embedding is not None:
            if time_id is None:
                raise ValueError("time_id is required when num_time_bins is configured")
            pieces.append(self.time_embedding(time_id))
        if self.moa_embedding is not None:
            if moa_id is None:
                raise ValueError("moa_id is required when num_moas is configured")
            pieces.append(self.moa_embedding(moa_id))
        if self.pathway_embedding is not None:
            if pathway_id is None:
                raise ValueError("pathway_id is required when num_pathways is configured")
            pieces.append(self.pathway_embedding(pathway_id))

        batch_size, device, dtype = self._infer_batch_device_dtype(
            pieces,
            condition_id,
            perturbation_id,
            cell_line_id,
            time_id,
            moa_id,
            pathway_id,
            condition_features,
            dose,
            time,
        )
        if condition_features is None:
            condition_features = torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype)
        else:
            if condition_features.shape[-1] != self.feature_dim:
                raise ValueError(f"expected condition feature dim {self.feature_dim}")
            condition_features = condition_features.to(device=device, dtype=dtype)
        log_dose = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if dose is not None:
            log_dose = torch.log1p(dose.to(device=device, dtype=dtype).reshape(batch_size, 1).clamp_min(0.0))
        continuous_time = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if time is not None:
            continuous_time = time.to(device=device, dtype=dtype).reshape(batch_size, 1)
        pieces.append(torch.cat((log_dose, continuous_time, condition_features), dim=-1))
        return self.norm(self.projection(torch.cat(pieces, dim=-1)))

    def _infer_batch_device_dtype(self, pieces: list[torch.Tensor], *values: torch.Tensor | None) -> tuple[int, torch.device, torch.dtype]:
        for piece in pieces:
            return piece.shape[0], piece.device, piece.dtype
        for value in values:
            if value is not None:
                dtype = value.dtype if value.is_floating_point() else torch.float32
                return value.shape[0], value.device, dtype
        raise ValueError("at least one condition tensor is required")


class CounterfactualResponsePredictor(nn.Module):
    def __init__(
        self,
        prototype_dim: int,
        condition_dim: int,
        *,
        hidden_dim: int | None = None,
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(prototype_dim, condition_dim)
        self.prototype_dim = prototype_dim
        self.condition_dim = condition_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.net = nn.Sequential(
            nn.Linear(prototype_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * prototype_dim),
        )

    def forward(
        self,
        control_prototypes: torch.Tensor,
        condition_embedding: torch.Tensor,
    ) -> CounterfactualResponseOutput:
        if control_prototypes.shape[-1] != self.prototype_dim:
            raise ValueError(f"expected prototype dim {self.prototype_dim}")
        if condition_embedding.ndim != 2 or condition_embedding.shape[-1] != self.condition_dim:
            raise ValueError(f"condition_embedding must have shape [batch, {self.condition_dim}]")
        if control_prototypes.shape[0] != condition_embedding.shape[0]:
            raise ValueError("control_prototypes and condition_embedding batch sizes must match")

        condition = condition_embedding
        while condition.ndim < control_prototypes.ndim:
            condition = condition.unsqueeze(1)
        condition = condition.expand(*control_prototypes.shape[:-1], self.condition_dim)
        delta_params = self.net(torch.cat((control_prototypes, condition), dim=-1))
        delta_mu, delta_logvar = delta_params.chunk(2, dim=-1)
        delta_logvar = delta_logvar.clamp(self.min_logvar, self.max_logvar)
        treated_mu = control_prototypes + delta_mu
        treated_logvar = delta_logvar
        return CounterfactualResponseOutput(
            delta_mu=delta_mu,
            delta_logvar=delta_logvar,
            treated_mu=treated_mu,
            treated_logvar=treated_logvar,
        )


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mu.shape != logvar.shape or mu.shape != target.shape:
        raise ValueError("mu, logvar, and target must have matching shapes")
    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar)).mean()


def counterfactual_mmd_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mmd_rbf_loss(predicted, target)


def counterfactual_cosine_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shapes")
    return 1.0 - F.cosine_similarity(predicted.flatten(0, -2), target.flatten(0, -2), dim=-1).mean()
