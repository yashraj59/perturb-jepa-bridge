from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_endpoint_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred_z, target_z.detach(), dim=-1)).mean()


def l2_endpoint_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_z, target_z.detach())


def endpoint_latent_loss(predicted_endpoint: torch.Tensor, teacher_target: torch.Tensor) -> torch.Tensor:
    return l2_endpoint_loss(predicted_endpoint, teacher_target)


def delta_cosine_loss(pred_delta: torch.Tensor, teacher_delta: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred_delta, teacher_delta.detach(), dim=-1)).mean()


def source_improvement_hinge(pred_z: torch.Tensor, source_z: torch.Tensor, target_z: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    source_cos = F.cosine_similarity(source_z, target_z.detach(), dim=-1)
    pred_cos = F.cosine_similarity(pred_z, target_z.detach(), dim=-1)
    return F.relu(source_cos + float(margin) - pred_cos).mean()


def source_improvement_hinge_loss(control: torch.Tensor, predicted_endpoint: torch.Tensor, teacher_target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    return source_improvement_hinge(predicted_endpoint, control, teacher_target, margin=margin)


def floor_gap_hinge(pred_z: torch.Tensor, floor_z: torch.Tensor, source_z: torch.Tensor, target_z: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    del source_z
    floor_cos = F.cosine_similarity(floor_z, target_z.detach(), dim=-1)
    pred_cos = F.cosine_similarity(pred_z, target_z.detach(), dim=-1)
    return F.relu(floor_cos + float(margin) - pred_cos).mean()


def action_negative_contrast_loss(pred_z: torch.Tensor, target_z: torch.Tensor, negative_pred_z: torch.Tensor, *, margin: float = 0.05) -> torch.Tensor:
    pos = F.cosine_similarity(pred_z, target_z.detach(), dim=-1)
    neg = F.cosine_similarity(negative_pred_z, target_z.detach(), dim=-1)
    return F.relu(float(margin) + neg - pos).mean()


def residual_norm_penalty(residual_delta: torch.Tensor, floor_delta: torch.Tensor) -> torch.Tensor:
    floor_norm = floor_delta.detach().norm(dim=-1).clamp_min(1.0e-8)
    return (residual_delta.norm(dim=-1) / floor_norm).mean()


def bioguard_wm_transition_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    control = batch["control_z_bio"]
    target = batch["teacher_target_z_bio"]
    teacher_delta = target.detach() - control
    pred_endpoint = outputs["predicted_endpoint"]
    pred_delta = outputs["predicted_delta"]
    loss = torch.zeros((), device=pred_endpoint.device, dtype=pred_endpoint.dtype)
    loss = loss + float(weights.get("endpoint", 1.0)) * endpoint_latent_loss(pred_endpoint, target)
    loss = loss + float(weights.get("delta_cosine", 1.0)) * delta_cosine_loss(pred_delta, teacher_delta)
    loss = loss + float(weights.get("source_improvement_hinge", 0.2)) * source_improvement_hinge_loss(control, pred_endpoint, target)
    loss = loss + float(weights.get("residual_norm", 0.01)) * residual_norm_penalty(outputs["residual_delta"], outputs["floor_delta"])
    return loss


def multistep_rollout_loss(
    step_predictions: list[torch.Tensor],
    step_targets: list[torch.Tensor],
    *,
    mode: str = "last_gradient_only",
) -> torch.Tensor:
    if len(step_predictions) != len(step_targets):
        raise ValueError("rollout prediction/target length mismatch")
    if not step_predictions:
        raise ValueError("at least one rollout step is required")
    losses = []
    for index, (pred, target) in enumerate(zip(step_predictions, step_targets)):
        if mode == "last_gradient_only" and index < len(step_predictions) - 1:
            pred = pred.detach()
        losses.append(l2_endpoint_loss(pred, target))
    return torch.stack(losses).mean()
