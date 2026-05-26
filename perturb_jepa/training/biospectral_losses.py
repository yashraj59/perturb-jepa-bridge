from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from perturb_jepa.training.bioaction_losses import vicreg_components
from perturb_jepa.training.biotech_losses import _effective_rank


@dataclass(frozen=True)
class BioSpectralLossWeights:
    endpoint_latent_jepa: float = 1.0
    whitened_delta_mse: float = 1.0
    delta_direction: float = 1.0
    source_improvement_hinge: float = 0.5
    floor_distillation: float = 1.0
    spectral_entropy_floor: float = 0.05
    covariance_shape: float = 0.05
    residual_norm_cap: float = 0.1
    residual_orthogonality: float = 0.05
    bio_tech_orthogonality: float = 0.05
    anti_collapse: float = 0.1


def biospectral_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    delta_whitening=None,
    weights: BioSpectralLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioSpectralLossWeights()
    source = outputs["z_control_bio"]
    target = outputs["z_target_bio_teacher"].detach()
    pred_target = outputs["predicted_target_bio"]
    pred_delta = outputs["predicted_delta_bio"]
    target_delta = outputs["delta_teacher"].detach()
    floor_delta = outputs.get("delta_floor", outputs.get("ridge_floor_delta_optional"))
    residual_delta = outputs.get("delta_residual", torch.zeros_like(pred_delta))
    total = torch.zeros((), device=pred_delta.device, dtype=pred_delta.dtype)
    diagnostics: dict[str, torch.Tensor] = {}

    endpoint = endpoint_latent_jepa_loss(pred_target, target)
    total = total + weights.endpoint_latent_jepa * endpoint
    _record(diagnostics, "endpoint_latent_jepa", endpoint, weights.endpoint_latent_jepa)

    if delta_whitening is not None:
        pred_delta_loss = delta_whitening.whiten(pred_delta)
        target_delta_loss = delta_whitening.whiten(target_delta)
    else:
        pred_delta_loss = pred_delta
        target_delta_loss = target_delta
    whitened = whitened_delta_mse_loss(pred_delta_loss, target_delta_loss)
    total = total + weights.whitened_delta_mse * whitened
    _record(diagnostics, "whitened_delta_mse", whitened, weights.whitened_delta_mse)

    direction = delta_direction_cosine_loss(pred_delta, target_delta)
    total = total + weights.delta_direction * direction
    _record(diagnostics, "delta_direction", direction, weights.delta_direction)

    hinge = source_improvement_hinge_loss(source, pred_target, target, margin=0.02)
    total = total + weights.source_improvement_hinge * hinge
    _record(diagnostics, "source_improvement_hinge", hinge, weights.source_improvement_hinge)

    floor = floor_distillation_loss(pred_delta, floor_delta) if floor_delta is not None else torch.zeros((), device=pred_delta.device)
    total = total + weights.floor_distillation * floor
    _record(diagnostics, "floor_distillation", floor, weights.floor_distillation)

    entropy = spectral_entropy_floor_loss(pred_delta, target_delta)
    total = total + weights.spectral_entropy_floor * entropy
    _record(diagnostics, "spectral_entropy", entropy, weights.spectral_entropy_floor)

    covariance = covariance_shape_loss(pred_delta, target_delta)
    total = total + weights.covariance_shape * covariance
    _record(diagnostics, "covariance_shape", covariance, weights.covariance_shape)

    cap_loss = residual_norm_cap_loss(residual_delta, floor_delta, cap=0.25)
    total = total + weights.residual_norm_cap * cap_loss
    _record(diagnostics, "residual_norm_cap", cap_loss, weights.residual_norm_cap)

    orth = residual_orthogonality_loss(residual_delta, floor_delta)
    total = total + weights.residual_orthogonality * orth
    _record(diagnostics, "residual_orthogonality", orth, weights.residual_orthogonality)

    if "z_control_tech" in outputs:
        bio_tech = bio_tech_orthogonality_loss(source, outputs["z_control_tech"])
    else:
        bio_tech = torch.zeros((), device=pred_delta.device)
    total = total + weights.bio_tech_orthogonality * bio_tech
    _record(diagnostics, "bio_tech_orthogonality", bio_tech, weights.bio_tech_orthogonality)

    var, cov = vicreg_components(pred_delta, target_std=0.03)
    anti = var + 0.1 * cov
    total = total + weights.anti_collapse * anti
    _record(diagnostics, "anti_collapse", anti, weights.anti_collapse)

    diagnostics["loss/total"] = total.detach()
    diagnostics.update(biospectral_operator_diagnostics(outputs))
    return total, diagnostics


def endpoint_latent_jepa_loss(pred_target: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred_target, target.detach(), dim=-1)).mean()


def whitened_delta_mse_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_delta, target_delta.detach())


def delta_direction_cosine_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred_delta, target_delta.detach(), dim=-1)).mean()


def source_improvement_hinge_loss(source: torch.Tensor, pred_target: torch.Tensor, target: torch.Tensor, *, margin: float) -> torch.Tensor:
    source_cos = F.cosine_similarity(source.detach(), target.detach(), dim=-1)
    pred_cos = F.cosine_similarity(pred_target, target.detach(), dim=-1)
    return F.relu(source_cos + float(margin) - pred_cos).mean()


def floor_distillation_loss(pred_delta: torch.Tensor, floor_delta: torch.Tensor | None) -> torch.Tensor:
    if floor_delta is None:
        return torch.zeros((), device=pred_delta.device, dtype=pred_delta.dtype)
    return F.smooth_l1_loss(pred_delta, floor_delta.detach())


def spectral_entropy_floor_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return F.relu(_effective_rank(target_delta.detach()) - _effective_rank(pred_delta))


def covariance_shape_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    if pred_delta.shape[0] < 2:
        return torch.zeros((), device=pred_delta.device, dtype=pred_delta.dtype)
    pred_cov = _covariance(pred_delta)
    target_cov = _covariance(target_delta.detach())
    return F.smooth_l1_loss(pred_cov, target_cov)


def residual_norm_cap_loss(residual_delta: torch.Tensor, floor_delta: torch.Tensor | None, *, cap: float) -> torch.Tensor:
    if floor_delta is None:
        return torch.zeros((), device=residual_delta.device, dtype=residual_delta.dtype)
    ratio = residual_delta.norm(dim=-1) / floor_delta.detach().norm(dim=-1).clamp_min(1.0e-8)
    return F.relu(ratio - float(cap)).mean()


def residual_orthogonality_loss(residual_delta: torch.Tensor, floor_delta: torch.Tensor | None) -> torch.Tensor:
    if floor_delta is None:
        return torch.zeros((), device=residual_delta.device, dtype=residual_delta.dtype)
    return F.cosine_similarity(residual_delta, floor_delta.detach(), dim=-1).abs().mean()


def bio_tech_orthogonality_loss(z_bio: torch.Tensor, z_tech: torch.Tensor) -> torch.Tensor:
    bio = z_bio - z_bio.mean(dim=0, keepdim=True)
    tech = z_tech - z_tech.mean(dim=0, keepdim=True)
    bio = F.normalize(bio, dim=0)
    tech = F.normalize(tech, dim=0)
    return (bio.T @ tech).pow(2).mean()


def biospectral_operator_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = outputs["z_control_bio"]
    pred = outputs["predicted_target_bio"]
    target = outputs["z_target_bio_teacher"].detach()
    pred_delta = outputs["predicted_delta_bio"]
    target_delta = outputs["delta_teacher"].detach()
    floor_delta = outputs.get("delta_floor")
    residual_delta = outputs.get("delta_residual", torch.zeros_like(pred_delta))
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    source_cos = F.cosine_similarity(source, target, dim=-1)
    delta_cos = F.cosine_similarity(pred_delta, target_delta, dim=-1)
    ratio = pred_delta.norm(dim=-1) / target_delta.norm(dim=-1).clamp_min(1.0e-8)
    floor_norm = floor_delta.norm(dim=-1).mean().clamp_min(1.0e-8) if floor_delta is not None else torch.ones((), device=pred_delta.device)
    return {
        "operator/floor_delta_norm": floor_delta.norm(dim=-1).mean().detach() if floor_delta is not None else torch.zeros((), device=pred_delta.device),
        "operator/residual_delta_norm": residual_delta.norm(dim=-1).mean().detach(),
        "operator/residual_to_floor_norm_ratio": (residual_delta.norm(dim=-1).mean() / floor_norm).detach(),
        "operator/residual_cap_hit_fraction": outputs.get("residual_cap_hit_fraction", torch.zeros((), device=pred_delta.device)).detach(),
        "operator/router_entropy": outputs.get("rank_ladder_router_entropy", torch.zeros((), device=pred_delta.device)).detach(),
        "operator/effective_rank": _effective_rank(pred_delta.detach().reshape(pred_delta.shape[0], -1)).detach(),
        "operator/floor_gap_improvement": torch.zeros((), device=pred_delta.device),
        "operator/floor_gap_recall": torch.zeros((), device=pred_delta.device),
        "operator/floor_gap_rank": torch.zeros((), device=pred_delta.device),
        "transition_source_cosine_improvement": (pred_cos - source_cos).mean().detach(),
        "delta_cosine": delta_cos.mean().detach(),
        "delta_magnitude_ratio": ratio.mean().detach(),
        "delta_teacher_effective_rank": _effective_rank(target_delta.detach().reshape(target_delta.shape[0], -1)).detach(),
    }


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, values.shape[0] - 1)


def _record(diagnostics: dict[str, torch.Tensor], name: str, loss: torch.Tensor, weight: float) -> None:
    diagnostics[f"loss/{name}"] = loss.detach()
    diagnostics[f"loss/weighted_{name}"] = (float(weight) * loss).detach()
