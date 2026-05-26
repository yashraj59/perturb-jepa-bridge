from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from perturb_jepa.training.bioaction_losses import vicreg_components
from perturb_jepa.training.biotech_losses import _effective_rank


@dataclass(frozen=True)
class BioFlowJEPALossWeights:
    endpoint_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    delta_direction_loss_weight: float = 2.0
    delta_magnitude_loss_weight: float = 0.2
    source_improvement_loss_weight: float = 2.0
    delta_rank_variance_weight: float = 0.1
    action_negative_loss_weight: float = 0.2
    zero_action_identity_weight: float = 0.5


def bioflow_jepa_loss(
    outputs: Mapping[str, torch.Tensor],
    weights: BioFlowJEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioFlowJEPALossWeights()
    device = _infer_device(outputs)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {}

    pred = outputs["z_pred"]
    target = outputs["target_z_bio_teacher"].detach()
    source = outputs["source_z_bio_online"]
    true_delta = outputs["true_delta"].detach()
    pred_delta = outputs["pred_delta"]

    velocity_pred = outputs.get("velocity_pred_whitened", outputs["velocity_pred"])
    velocity_target = outputs.get("velocity_target_whitened", outputs["velocity_target"]).detach()
    velocity = F.smooth_l1_loss(velocity_pred, velocity_target)
    total = total + float(weights.velocity_loss_weight) * velocity
    diagnostics["loss/velocity"] = velocity.detach()

    endpoint = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    total = total + float(weights.endpoint_loss_weight) * endpoint
    diagnostics["loss/endpoint"] = endpoint.detach()

    delta_cosine = F.cosine_similarity(pred_delta, true_delta, dim=-1)
    direction = (1.0 - delta_cosine).mean() + F.relu(-delta_cosine).mean()
    total = total + float(weights.delta_direction_loss_weight) * direction
    diagnostics["loss/delta_direction"] = direction.detach()

    magnitude = F.smooth_l1_loss(
        torch.log(pred_delta.norm(dim=-1).clamp_min(1.0e-8)),
        torch.log(true_delta.norm(dim=-1).clamp_min(1.0e-8)),
    )
    total = total + float(weights.delta_magnitude_loss_weight) * magnitude
    diagnostics["loss/delta_magnitude"] = magnitude.detach()

    source_cos = F.cosine_similarity(source, target, dim=-1).detach()
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    improve = F.relu(source_cos + 0.02 - pred_cos).mean()
    total = total + float(weights.source_improvement_loss_weight) * improve
    diagnostics["loss/source_improvement_hinge"] = improve.detach()

    var, cov = vicreg_components(pred_delta, target_std=0.03)
    rank_variance = var + 0.1 * cov
    total = total + float(weights.delta_rank_variance_weight) * rank_variance
    diagnostics["loss/delta_rank_variance"] = rank_variance.detach()

    wrong = outputs.get("wrong_action_z_pred")
    if wrong is not None and float(weights.action_negative_loss_weight):
        wrong_cos = F.cosine_similarity(wrong, target, dim=-1)
        action_neg = F.relu(wrong_cos - pred_cos + 0.02).mean()
        total = total + float(weights.action_negative_loss_weight) * action_neg
        diagnostics["loss/action_negative"] = action_neg.detach()
    else:
        diagnostics["loss/action_negative"] = torch.zeros((), device=device)

    diagnostics["loss/total"] = total.detach()
    diagnostics.update(bioflow_diagnostics(outputs))
    return total, diagnostics


def bioflow_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    source = outputs["source_z_bio_online"]
    target = outputs["target_z_bio_teacher"].detach()
    pred = outputs["z_pred"]
    true_delta = outputs["true_delta"].detach()
    pred_delta = outputs["pred_delta"]
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    source_cos = F.cosine_similarity(source, target, dim=-1)
    delta_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1)
    true_norm = true_delta.norm(dim=-1).clamp_min(1.0e-8)
    pred_norm = pred_delta.norm(dim=-1)
    return {
        "transition_bio_cosine_to_teacher": pred_cos.mean().detach(),
        "source_as_target_bio_cosine_to_teacher": source_cos.mean().detach(),
        "transition_source_cosine_improvement": (pred_cos - source_cos).mean().detach(),
        "delta_cosine": delta_cos.mean().detach(),
        "delta_magnitude_ratio": (pred_norm / true_norm).mean().detach(),
        "delta_prediction_effective_rank": _effective_rank(pred_delta.detach().reshape(pred_delta.shape[0], -1)).detach(),
        "delta_teacher_effective_rank": _effective_rank(true_delta.detach().reshape(true_delta.shape[0], -1)).detach(),
        "source_improvement_hinge_violation_fraction": (pred_cos < source_cos + 0.02).float().mean().detach(),
        "std_mean_pred_delta": pred_delta.detach().std(dim=0, unbiased=False).mean(),
        "std_mean_teacher_delta": true_delta.detach().std(dim=0, unbiased=False).mean(),
    }


def source_improvement_hinge_loss(
    source: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    source_cos = F.cosine_similarity(source, target, dim=-1).detach()
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    return F.relu(source_cos + float(margin) - pred_cos).mean()


def _infer_device(outputs: Mapping[str, torch.Tensor]) -> torch.device:
    for value in outputs.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")
