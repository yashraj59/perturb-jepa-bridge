from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProgramBootstrapJEPAConfig:
    genes: int
    image_dim: int
    action_dim: int
    z_dim: int = 24
    hidden_dim: int = 96
    ema_decay: float = 0.99
    pca_anchor_weight: float = 0.2
    transition_weight: float = 1.0
    cross_modal_weight: float = 1.0
    action_negative_weight: float = 0.1
    delta_direction_weight: float = 0.0
    source_improvement_weight: float = 0.0
    variance_weight: float = 0.05


@dataclass(frozen=True)
class FloorInitializedTransitionHeadConfig:
    z_dim: int
    action_dim: int
    hidden_dim: int = 96
    residual_scale_init: float = 1.0


class ProgramBootstrapJEPA(nn.Module):
    """Small real-JEPA probe using non-exact program actions.

    The model is intentionally narrow for total-autonomy diagnostics:
    RNA/image online encoders are paired with EMA target encoders, latent
    teacher targets are stop-gradient, and query-conditioned predictors solve
    RNA->image, image->RNA, and control+program-action->target-RNA tasks.
    """

    def __init__(self, config: ProgramBootstrapJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_online = _mlp(config.genes, config.hidden_dim, config.z_dim)
        self.image_online = _mlp(config.image_dim, config.hidden_dim, config.z_dim)
        self.rna_target = _mlp(config.genes, config.hidden_dim, config.z_dim)
        self.image_target = _mlp(config.image_dim, config.hidden_dim, config.z_dim)
        self.rna_target.load_state_dict(self.rna_online.state_dict())
        self.image_target.load_state_dict(self.image_online.state_dict())
        for module in (self.rna_target, self.image_target):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.transition_query = nn.Parameter(torch.zeros(config.z_dim))
        self.rna_to_image_query = nn.Parameter(torch.zeros(config.z_dim))
        self.image_to_rna_query = nn.Parameter(torch.zeros(config.z_dim))
        self.transition_predictor = _mlp(config.z_dim + config.action_dim + config.z_dim, config.hidden_dim, config.z_dim)
        self.rna_to_image_predictor = _mlp(config.z_dim + config.z_dim, config.hidden_dim, config.z_dim)
        self.image_to_rna_predictor = _mlp(config.z_dim + config.z_dim, config.hidden_dim, config.z_dim)

    def encode_rna(self, expression_values: torch.Tensor) -> torch.Tensor:
        return self.rna_online(expression_values)

    def encode_image(self, images_flat: torch.Tensor) -> torch.Tensor:
        return self.image_online(images_flat)

    def predict_transition(self, control_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        query = self.transition_query.expand(control_z.shape[0], -1)
        return self.transition_predictor(torch.cat((control_z, action.to(control_z), query), dim=-1))

    def predict_rna_to_image(self, rna_z: torch.Tensor) -> torch.Tensor:
        query = self.rna_to_image_query.expand(rna_z.shape[0], -1)
        return self.rna_to_image_predictor(torch.cat((rna_z, query), dim=-1))

    def predict_image_to_rna(self, image_z: torch.Tensor) -> torch.Tensor:
        query = self.image_to_rna_query.expand(image_z.shape[0], -1)
        return self.image_to_rna_predictor(torch.cat((image_z, query), dim=-1))

    def forward_loss(
        self,
        *,
        control_expression: torch.Tensor,
        target_expression: torch.Tensor,
        target_image_flat: torch.Tensor,
        action: torch.Tensor,
        pca_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        control_z = self.rna_online(control_expression)
        target_rna_z = self.rna_online(target_expression)
        target_image_z = self.image_online(target_image_flat)
        with torch.no_grad():
            control_teacher = self.rna_target(control_expression).detach()
            target_rna_teacher = self.rna_target(target_expression).detach()
            target_image_teacher = self.image_target(target_image_flat).detach()

        transition_pred = self.predict_transition(control_z, action)
        rna_to_image_pred = self.predict_rna_to_image(target_rna_z)
        image_to_rna_pred = self.predict_image_to_rna(target_image_z)
        shuffled_action = action[torch.randperm(action.shape[0])]
        negative_transition = self.predict_transition(control_z, shuffled_action)

        transition_loss = _cosine_loss(transition_pred, target_rna_teacher) + 0.05 * F.mse_loss(transition_pred, target_rna_teacher)
        pred_delta = transition_pred - control_z.detach()
        teacher_delta = target_rna_teacher - control_teacher
        delta_direction_loss = _cosine_loss(pred_delta, teacher_delta)
        source_target_cos = F.cosine_similarity(control_z.detach(), target_rna_teacher, dim=-1)
        transition_target_cos = F.cosine_similarity(transition_pred, target_rna_teacher, dim=-1)
        source_improvement_hinge = F.relu(source_target_cos - transition_target_cos + 0.02).mean()
        rna_to_image_loss = _cosine_loss(rna_to_image_pred, target_image_teacher) + 0.05 * F.mse_loss(rna_to_image_pred, target_image_teacher)
        image_to_rna_loss = _cosine_loss(image_to_rna_pred, target_rna_teacher) + 0.05 * F.mse_loss(image_to_rna_pred, target_rna_teacher)
        pca_anchor = _cosine_loss(target_rna_z, pca_target) + 0.05 * F.mse_loss(target_rna_z, pca_target)
        positive_cos = F.cosine_similarity(transition_pred, target_rna_teacher, dim=-1)
        negative_cos = F.cosine_similarity(negative_transition, target_rna_teacher, dim=-1)
        action_negative = F.relu(negative_cos - positive_cos + 0.05).mean()
        variance = _variance_floor(target_rna_z) + _variance_floor(target_image_z)
        loss = (
            self.config.transition_weight * transition_loss
            + self.config.delta_direction_weight * delta_direction_loss
            + self.config.source_improvement_weight * source_improvement_hinge
            + self.config.cross_modal_weight * (rna_to_image_loss + image_to_rna_loss)
            + self.config.pca_anchor_weight * pca_anchor
            + self.config.action_negative_weight * action_negative
            + self.config.variance_weight * variance
        )
        metrics = {
            "loss": loss.detach(),
            "transition_loss": transition_loss.detach(),
            "delta_direction_loss": delta_direction_loss.detach(),
            "source_improvement_hinge": source_improvement_hinge.detach(),
            "rna_to_image_loss": rna_to_image_loss.detach(),
            "image_to_rna_loss": image_to_rna_loss.detach(),
            "pca_anchor_loss": pca_anchor.detach(),
            "action_negative_loss": action_negative.detach(),
            "variance_loss": variance.detach(),
            "mean_positive_transition_cosine": positive_cos.mean().detach(),
            "mean_negative_transition_cosine": negative_cos.mean().detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def update_targets(self, decay: float | None = None) -> None:
        decay = self.config.ema_decay if decay is None else float(decay)
        _ema_update(self.rna_online, self.rna_target, decay)
        _ema_update(self.image_online, self.image_target, decay)


class FloorInitializedTransitionHead(nn.Module):
    """Action-conditioned transition predictor initialized to a frozen ridge floor."""

    def __init__(
        self,
        config: FloorInitializedTransitionHeadConfig,
        *,
        floor_weight: torch.Tensor,
        floor_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        expected_weight_shape = (config.z_dim, config.z_dim + config.action_dim)
        if tuple(floor_weight.shape) != expected_weight_shape:
            raise ValueError(f"floor_weight must have shape {expected_weight_shape}, got {tuple(floor_weight.shape)}")
        if tuple(floor_bias.shape) != (config.z_dim,):
            raise ValueError(f"floor_bias must have shape {(config.z_dim,)}, got {tuple(floor_bias.shape)}")
        self.register_buffer("floor_weight", floor_weight.detach().clone().float())
        self.register_buffer("floor_bias", floor_bias.detach().clone().float())
        self.transition_query = nn.Parameter(torch.zeros(config.z_dim))
        self.residual = _mlp(config.z_dim + config.action_dim + config.z_dim, config.hidden_dim, config.z_dim)
        _zero_last_linear(self.residual)
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))

    def floor_delta(self, source_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((source_z, action.to(source_z)), dim=-1)
        return F.linear(features, self.floor_weight.to(source_z), self.floor_bias.to(source_z))

    def residual_delta(self, source_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        query = self.transition_query.expand(source_z.shape[0], -1)
        return self.residual(torch.cat((source_z, action.to(source_z), query), dim=-1))

    def forward(self, source_z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.floor_delta(source_z, action) + self.residual_scale * self.residual_delta(source_z, action)
        return source_z + delta


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            nn.init.zeros_(child.bias)
            return


def _cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, target.detach(), dim=-1).mean()


def _variance_floor(values: torch.Tensor, floor: float = 0.2) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_zeros(())
    std = torch.sqrt(values.var(dim=0, unbiased=False) + 1.0e-4)
    return F.relu(float(floor) - std).mean()


@torch.no_grad()
def _ema_update(online: nn.Module, target: nn.Module, decay: float) -> None:
    for online_parameter, target_parameter in zip(online.parameters(), target.parameters(), strict=True):
        target_parameter.data.mul_(decay).add_(online_parameter.data, alpha=1.0 - decay)
