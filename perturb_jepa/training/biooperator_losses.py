from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from perturb_jepa.training.bioaction_losses import vicreg_components
from perturb_jepa.training.biotech_losses import _effective_rank


@dataclass(frozen=True)
class BioOperatorLossWeights:
    whitened_delta_mse: float = 1.0
    delta_direction: float = 0.5
    endpoint_jepa: float = 0.5
    ridge_distillation: float = 0.2
    source_hinge: float = 0.1
    delta_rank_variance_floor: float = 0.05


def biooperator_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    delta_whitening=None,
    weights: BioOperatorLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioOperatorLossWeights()
    pred_delta = outputs["predicted_delta_bio"]
    target_delta = outputs["delta_teacher"].detach()
    pred_target = outputs["predicted_target_bio"]
    target = outputs["z_target_bio_teacher"].detach()
    source = outputs["z_control_bio"]
    if delta_whitening is not None:
        pred_delta_loss = delta_whitening.whiten(pred_delta)
        target_delta_loss = delta_whitening.whiten(target_delta)
    else:
        pred_delta_loss = pred_delta
        target_delta_loss = target_delta
    total = torch.zeros((), device=pred_delta.device)
    diagnostics: dict[str, torch.Tensor] = {}

    whitened = F.smooth_l1_loss(pred_delta_loss, target_delta_loss)
    total = total + float(weights.whitened_delta_mse) * whitened
    diagnostics["unweighted_whitened_delta_mse"] = whitened.detach()
    diagnostics["weighted_whitened_delta_mse"] = (float(weights.whitened_delta_mse) * whitened).detach()

    delta_cos = F.cosine_similarity(pred_delta, target_delta, dim=-1)
    direction = (1.0 - delta_cos).mean()
    total = total + float(weights.delta_direction) * direction
    diagnostics["unweighted_delta_direction"] = direction.detach()
    diagnostics["weighted_delta_direction"] = (float(weights.delta_direction) * direction).detach()

    endpoint = (1.0 - F.cosine_similarity(pred_target, target, dim=-1)).mean()
    total = total + float(weights.endpoint_jepa) * endpoint
    diagnostics["unweighted_endpoint_jepa"] = endpoint.detach()
    diagnostics["weighted_endpoint_jepa"] = (float(weights.endpoint_jepa) * endpoint).detach()

    ridge = outputs.get("ridge_floor_delta_optional")
    if ridge is not None:
        ridge_loss = F.smooth_l1_loss(pred_delta, ridge.detach())
    else:
        ridge_loss = torch.zeros((), device=pred_delta.device)
    total = total + float(weights.ridge_distillation) * ridge_loss
    diagnostics["unweighted_ridge_distillation"] = ridge_loss.detach()
    diagnostics["weighted_ridge_distillation"] = (float(weights.ridge_distillation) * ridge_loss).detach()

    hinge = source_improvement_hinge(source, pred_target, target, margin=0.02)
    total = total + float(weights.source_hinge) * hinge
    diagnostics["unweighted_source_hinge"] = hinge.detach()
    diagnostics["weighted_source_hinge"] = (float(weights.source_hinge) * hinge).detach()

    var, cov = vicreg_components(pred_delta, target_std=0.03)
    rank_floor = var + 0.1 * cov
    total = total + float(weights.delta_rank_variance_floor) * rank_floor
    diagnostics["unweighted_delta_rank_variance_floor"] = rank_floor.detach()
    diagnostics["weighted_delta_rank_variance_floor"] = (float(weights.delta_rank_variance_floor) * rank_floor).detach()

    diagnostics["loss/total"] = total.detach()
    diagnostics["weighted_operator_to_main_ratio"] = torch.ones((), device=pred_delta.device)
    diagnostics.update(operator_diagnostics(outputs))
    return total, diagnostics


def source_improvement_hinge(source: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, *, margin: float = 0.02) -> torch.Tensor:
    source_cos = F.cosine_similarity(source, target, dim=-1).detach()
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    return F.relu(source_cos + float(margin) - pred_cos).mean()


def magnitude_contract_violation(pred_delta: torch.Tensor, target_delta: torch.Tensor, *, limit: float = 2.0) -> torch.Tensor:
    ratio = pred_delta.norm(dim=-1) / target_delta.norm(dim=-1).clamp_min(1.0e-8)
    return (ratio > float(limit)).float().mean()


def operator_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = outputs["z_control_bio"]
    pred = outputs["predicted_target_bio"]
    target = outputs["z_target_bio_teacher"].detach()
    pred_delta = outputs["predicted_delta_bio"]
    target_delta = outputs["delta_teacher"].detach()
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    source_cos = F.cosine_similarity(source, target, dim=-1)
    delta_cos = F.cosine_similarity(pred_delta, target_delta, dim=-1)
    ratio = pred_delta.norm(dim=-1) / target_delta.norm(dim=-1).clamp_min(1.0e-8)
    return {
        "transition_source_cosine_improvement": (pred_cos - source_cos).mean().detach(),
        "delta_cosine": delta_cos.mean().detach(),
        "delta_magnitude_ratio": ratio.mean().detach(),
        "delta_prediction_effective_rank": _effective_rank(pred_delta.detach().reshape(pred_delta.shape[0], -1)).detach(),
        "delta_teacher_effective_rank": _effective_rank(target_delta.detach().reshape(target_delta.shape[0], -1)).detach(),
        "source_improvement_hinge_violation_fraction": (pred_cos < source_cos + 0.02).float().mean().detach(),
        "magnitude_contract_violation_fraction": magnitude_contract_violation(pred_delta, target_delta).detach(),
    }
