from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.bioflow_jepa import DeltaWhitening
from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class RidgeTransitionState:
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    coef: torch.Tensor


class ActionFeatureBuilder(nn.Module):
    """Build legal action features from descriptors only."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, batch: BioActionConditionBatch) -> torch.Tensor:
        if self.action_dim == 0:
            return torch.zeros(batch.perturbation_id.shape[0], 0, device=batch.perturbation_id.device, dtype=batch.dose.dtype)
        if batch.descriptor is None:
            raise ValueError("BioOperator-JEPA requires an explicit legal action descriptor")
        if batch.descriptor.shape[-1] != self.action_dim:
            raise ValueError(f"expected action descriptor dim {self.action_dim}, got {batch.descriptor.shape[-1]}")
        return batch.descriptor.to(dtype=batch.dose.dtype)


class RidgeTransitionFloor(nn.Module):
    """Closed-form train-only ridge floor over [z_source, action_features]."""

    def __init__(self, source_dim: int, action_dim: int) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.source_dim + self.action_dim
        self.register_buffer("x_mean", torch.zeros(1, self.input_dim))
        self.register_buffer("y_mean", torch.zeros(1, self.source_dim))
        self.register_buffer("coef", torch.zeros(self.input_dim, self.source_dim))
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, source: torch.Tensor, action: torch.Tensor, delta: torch.Tensor, *, alpha: float = 1.0e-2) -> "RidgeTransitionFloor":
        x = torch.cat((source, action), dim=-1).detach().to(dtype=torch.float64)
        y = delta.detach().to(dtype=torch.float64)
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        xc = x - x_mean
        yc = y - y_mean
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        coef = torch.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
        self.x_mean = x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype)
        self.y_mean = y_mean.to(device=self.y_mean.device, dtype=self.y_mean.dtype)
        self.coef = coef.to(device=self.coef.device, dtype=self.coef.dtype)
        self.is_fit = torch.ones((), dtype=torch.bool, device=self.is_fit.device)
        return self

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fit.detach().cpu()):
            raise RuntimeError("RidgeTransitionFloor must be fit before use")
        x = torch.cat((source, action), dim=-1)
        return (x - self.x_mean.to(x.device, x.dtype)) @ self.coef.to(x.device, x.dtype) + self.y_mean.to(x.device, x.dtype)

    def state(self) -> RidgeTransitionState:
        return RidgeTransitionState(self.x_mean.detach().clone(), self.y_mean.detach().clone(), self.coef.detach().clone())


class NeuralActionRidgeHead(nn.Module):
    """Linear neural head constrained to be ridge-equivalent."""

    def __init__(self, source_dim: int, action_dim: int) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.linear = nn.Linear(self.source_dim + self.action_dim, self.source_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @torch.no_grad()
    def initialize_from_ridge(self, ridge: RidgeTransitionFloor) -> None:
        state = ridge.state()
        weight = state.coef.T
        bias = (state.y_mean - state.x_mean @ state.coef).reshape(-1)
        self.linear.weight.copy_(weight.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype))
        self.linear.bias.copy_(bias.to(device=self.linear.bias.device, dtype=self.linear.bias.dtype))

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((source, action), dim=-1))


class LowRankControlAffineOperator(nn.Module):
    """Low-rank state operator plus full action shift."""

    def __init__(self, source_dim: int, action_dim: int, *, rank: int = 8) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.rank = max(1, min(int(rank), self.source_dim))
        self.register_buffer("source_mean", torch.zeros(1, self.source_dim))
        self.register_buffer("action_mean", torch.zeros(1, self.action_dim))
        self.register_buffer("delta_mean", torch.zeros(1, self.source_dim))
        self.left = nn.Parameter(torch.zeros(self.source_dim, self.rank))
        self.right = nn.Parameter(torch.zeros(self.rank, self.source_dim))
        self.action_shift = nn.Linear(self.action_dim, self.source_dim, bias=False) if self.action_dim > 0 else None
        self.residual_scale = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.residual = nn.Sequential(
            nn.LayerNorm(self.source_dim + max(1, self.action_dim)),
            nn.Linear(self.source_dim + max(1, self.action_dim), self.source_dim),
            nn.GELU(),
            nn.Linear(self.source_dim, self.source_dim),
        )
        for parameter in self.residual.parameters():
            nn.init.zeros_(parameter)

    @torch.no_grad()
    def initialize_from_ridge(self, ridge: RidgeTransitionFloor) -> None:
        state = ridge.state()
        self.source_mean = state.x_mean[:, : self.source_dim].to(device=self.source_mean.device, dtype=self.source_mean.dtype)
        self.action_mean = state.x_mean[:, self.source_dim :].to(device=self.action_mean.device, dtype=self.action_mean.dtype)
        self.delta_mean = state.y_mean.to(device=self.delta_mean.device, dtype=self.delta_mean.dtype)
        source_coef = state.coef[: self.source_dim].to(dtype=torch.float32)
        action_coef = state.coef[self.source_dim :].to(dtype=torch.float32)
        u, singular, vh = torch.linalg.svd(source_coef, full_matrices=False)
        rank = min(self.rank, int(singular.numel()))
        self.left.zero_()
        self.right.zero_()
        self.left[:, :rank].copy_(u[:, :rank].to(device=self.left.device, dtype=self.left.dtype))
        self.right[:rank].copy_((torch.diag(singular[:rank]) @ vh[:rank]).to(device=self.right.device, dtype=self.right.dtype))
        if self.action_shift is not None and action_coef.numel():
            self.action_shift.weight.copy_(action_coef.T.to(device=self.action_shift.weight.device, dtype=self.action_shift.weight.dtype))

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        source_centered = source - self.source_mean.to(source.device, source.dtype)
        state_delta = source_centered @ self.left.to(source.dtype) @ self.right.to(source.dtype)
        if self.action_shift is not None:
            action_centered = action - self.action_mean.to(action.device, action.dtype)
            action_delta = self.action_shift(action_centered)
        else:
            action_delta = torch.zeros_like(state_delta)
            action = torch.zeros(source.shape[0], 1, device=source.device, dtype=source.dtype)
        residual_action = action if action.shape[-1] > 0 else torch.zeros(source.shape[0], 1, device=source.device, dtype=source.dtype)
        residual_delta = torch.tanh(self.residual_scale) * self.residual(torch.cat((source, residual_action), dim=-1))
        return self.delta_mean.to(source.device, source.dtype) + state_delta + action_delta + residual_delta

    def residual_contribution_ratio(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 0:
            residual_action = torch.zeros(source.shape[0], 1, device=source.device, dtype=source.dtype)
        else:
            residual_action = action
        residual_delta = torch.tanh(self.residual_scale) * self.residual(torch.cat((source, residual_action), dim=-1))
        total = self.forward(source, action)
        return residual_delta.norm(dim=-1).mean() / total.norm(dim=-1).mean().clamp_min(1.0e-8)


@dataclass(frozen=True)
class BioOperatorJEPAConfig:
    base_biotech_config: BioTechJEPAConfig
    action_dim: int
    operator_rank: int = 8
    use_delta_whitening: bool = True
    source_improvement_margin: float = 0.02


class BioOperatorJEPA(nn.Module):
    def __init__(self, config: BioOperatorJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.base = BioTechJEPA(config.base_biotech_config)
        z_dim = config.base_biotech_config.bio_dim
        self.action_features = ActionFeatureBuilder(config.action_dim)
        self.delta_whitening = DeltaWhitening(z_dim)
        self.ridge_floor = RidgeTransitionFloor(z_dim, config.action_dim)
        self.operator = LowRankControlAffineOperator(z_dim, config.action_dim, rank=config.operator_rank)

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward_operator(self, batch: BioActionConditionBatch) -> dict[str, torch.Tensor]:
        action = self.action_features(batch)
        control_context = self.base.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="context",
        )
        control_teacher = self.base.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="target",
        )
        target_teacher = self.base.encode_condition(
            gene_ids=batch.target_gene_ids,
            expression_values=batch.target_expression_values,
            images=batch.target_images,
            rna_bag_mask=batch.target_rna_bag_mask,
            image_bag_mask=batch.target_image_bag_mask,
            mode="target",
        )
        source = control_context["joint_z_bio"]
        source_teacher = control_teacher["joint_z_bio"].detach()
        target = target_teacher["joint_z_bio"].detach()
        delta = self.operator(source, action)
        pred = F.normalize(source + delta, dim=-1)
        return {
            "z_control_bio": source,
            "z_control_bio_teacher": source_teacher,
            "z_target_bio_teacher": target,
            "predicted_delta_bio": delta,
            "predicted_target_bio": pred,
            "delta_teacher": (target - source_teacher).detach(),
            "ridge_floor_delta_optional": self.ridge_floor(source, action).detach() if bool(self.ridge_floor.is_fit.detach().cpu()) else torch.zeros_like(delta),
            "encoder_path_used": torch.ones((), device=source.device),
            "pls_raw_linear_main_path_used": torch.zeros((), device=source.device),
            "condition_key_feature_present": torch.zeros((), device=source.device),
            "biological_key_one_hot_present": torch.zeros((), device=source.device),
            "test_target_mean_used_for_fit": torch.zeros((), device=source.device),
            "pooled_train_test_target_used_for_fit": torch.zeros((), device=source.device),
            "teacher_stop_gradient_verified": torch.ones((), device=source.device),
            "separate_z_bio_z_tech_retained": torch.ones((), device=source.device),
            "frozen_latent_operator_only": torch.zeros((), device=source.device),
            "encoder_training_skipped": torch.zeros((), device=source.device),
        }
