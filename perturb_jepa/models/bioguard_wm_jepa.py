from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from perturb_jepa.models.action_adaln_predictor import ActionAdaLNRoPEPredictor
from perturb_jepa.training.biospectral_operator import NumpyRidgeFit, fit_ridge_numpy


@dataclass(frozen=True)
class BioGuardWMJEPAConfig:
    bio_dim: int = 24
    action_dim: int = 32
    tech_dim: int = 8
    predictor_dim: int = 64
    predictor_depth: int = 6
    predictor_heads: int = 4
    context_tokens: tuple[str, ...] = ("control_z_bio", "ridge_floor_z_bio", "action_token")
    use_uncertainty_token: bool = False
    adaln_zero: bool = True
    residual_scale_init: float = 0.0
    max_residual_scale: float = 1.0
    preserve_floor_exactly: bool = True
    floor_head_trainable: bool = False
    detach_floor: bool = True
    residual_target_mode: str = "teacher_delta_minus_floor_delta"
    loss_endpoint_weight: float = 1.0
    loss_delta_cosine_weight: float = 1.0
    loss_source_improvement_hinge_weight: float = 0.2
    loss_residual_norm_weight: float = 0.01
    vicreg_weight: float = 0.0


class RidgeFloorHead(nn.Module):
    def __init__(self, fit: NumpyRidgeFit) -> None:
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(fit.x_mean, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(fit.y_mean, dtype=torch.float32))
        self.register_buffer("coef", torch.as_tensor(fit.coef, dtype=torch.float32))

    @classmethod
    def fit_train_only(cls, source_z_bio: np.ndarray, action_features: np.ndarray, teacher_delta: np.ndarray, *, alpha: float = 1.0e-2) -> "RidgeFloorHead":
        source = np.asarray(source_z_bio, dtype=np.float64)
        action = np.asarray(action_features, dtype=np.float64)
        if action.size == 0:
            action = np.zeros((source.shape[0], 0), dtype=np.float64)
        features = np.concatenate((source, action), axis=1)
        return cls(fit_ridge_numpy(features, np.asarray(teacher_delta, dtype=np.float64), alpha=alpha))

    def forward(self, control_z_bio: torch.Tensor, action_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if action_features.shape[-1] == 0:
            action_features = torch.zeros(control_z_bio.shape[0], 0, device=control_z_bio.device, dtype=control_z_bio.dtype)
        features = torch.cat((control_z_bio, action_features.to(dtype=control_z_bio.dtype)), dim=-1)
        floor_delta = (features - self.x_mean.to(features)) @ self.coef.to(features) + self.y_mean.to(features)
        return {"floor_delta": floor_delta, "floor_endpoint": control_z_bio + floor_delta}


class FloorPreservingJEPAWMTransitionHead(nn.Module):
    def __init__(self, config: BioGuardWMJEPAConfig) -> None:
        super().__init__()
        self.config = config
        token_count = len(config.context_tokens) + (1 if config.use_uncertainty_token else 0)
        self.control_proj = nn.Linear(config.bio_dim, config.predictor_dim)
        self.floor_proj = nn.Linear(config.bio_dim, config.predictor_dim)
        self.action_proj = nn.Linear(config.action_dim, config.predictor_dim)
        self.tech_proj = nn.Linear(config.tech_dim, config.predictor_dim) if config.use_uncertainty_token else None
        self.predictor = ActionAdaLNRoPEPredictor(
            config.predictor_dim,
            config.action_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            max_context_tokens=token_count,
            output_dim=config.bio_dim,
            adaln_zero=config.adaln_zero,
        )
        self.register_buffer("residual_scale", torch.as_tensor(float(config.residual_scale_init), dtype=torch.float32))

    def forward(
        self,
        control_z_bio: torch.Tensor,
        action_features: torch.Tensor,
        *,
        floor_delta: torch.Tensor,
        z_tech: torch.Tensor | None = None,
        residual_scale_override: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        floor_delta = floor_delta.detach() if self.config.detach_floor else floor_delta
        floor_endpoint = control_z_bio + floor_delta
        tokens = [
            self.control_proj(control_z_bio),
            self.floor_proj(floor_endpoint),
            self.action_proj(action_features.to(dtype=control_z_bio.dtype)),
        ]
        if self.config.use_uncertainty_token:
            if z_tech is None:
                z_tech = torch.zeros(control_z_bio.shape[0], self.config.tech_dim, device=control_z_bio.device, dtype=control_z_bio.dtype)
            if self.tech_proj is None:
                raise RuntimeError("tech projection missing")
            tokens.append(self.tech_proj(z_tech))
        context = torch.stack(tokens, dim=1)
        raw_residual = self.predictor(context, action_features)
        scale = self._scale(control_z_bio, residual_scale_override)
        residual_delta = scale * raw_residual
        predicted_delta = floor_delta + residual_delta
        predicted_endpoint = control_z_bio + predicted_delta
        return {
            "floor_delta": floor_delta,
            "floor_endpoint": floor_endpoint,
            "raw_residual_delta": raw_residual,
            "residual_delta": residual_delta,
            "residual_scale": scale,
            "predicted_delta": predicted_delta,
            "predicted_endpoint": predicted_endpoint,
            "context_tokens": context,
            "context_contract_hash": torch.as_tensor(_context_hash(self.config.context_tokens), device=control_z_bio.device),
        }

    def _scale(self, control_z_bio: torch.Tensor, override: float | torch.Tensor | None) -> torch.Tensor:
        value = self.residual_scale if override is None else override
        if torch.is_tensor(value):
            tensor = value.to(device=control_z_bio.device, dtype=control_z_bio.dtype)
        else:
            tensor = torch.as_tensor(float(value), device=control_z_bio.device, dtype=control_z_bio.dtype)
        tensor = torch.clamp(tensor, min=0.0, max=float(self.config.max_residual_scale))
        if tensor.ndim == 0:
            tensor = tensor.expand(control_z_bio.shape[0], 1)
        elif tensor.ndim == 1:
            tensor = tensor[:, None]
        return tensor


def _context_hash(tokens: tuple[str, ...]) -> int:
    digest = hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
