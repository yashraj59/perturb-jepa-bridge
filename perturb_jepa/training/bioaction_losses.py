from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BioActionJEPALossWeights:
    rna_program_jepa: float = 1.0
    image_region_jepa: float = 1.0
    rna_to_image_jepa: float = 2.0
    image_to_rna_jepa: float = 2.0
    joint_to_rna_jepa: float = 1.0
    joint_to_image_jepa: float = 1.0
    transition_rna_jepa: float = 2.0
    transition_image_jepa: float = 2.0
    transition_joint_jepa: float = 2.0
    distributional_jepa: float = 1.0
    vicreg_variance: float = 0.1
    vicreg_covariance: float = 0.05
    barlow_cross_correlation: float = 0.05
    count_nb_nll_aux: float = 0.1
    program_delta_aux: float = 0.2
    cycle_latent_aux: float = 0.1
    inverse_action_aux: float = 0.05
    batch_invariance: float = 0.0
    raw_rna_reconstruction: float = 0.0
    raw_image_reconstruction: float = 0.0


def cosine_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(target.detach(), dim=-1), dim=-1).mean()


def smooth_l1_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(F.normalize(pred, dim=-1), F.normalize(target.detach(), dim=-1))


def masked_cosine_jepa_loss(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, target_mask: torch.Tensor | None = None) -> torch.Tensor:
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError("pred_tokens and target_tokens must have matching shapes")
    cosine = F.cosine_similarity(F.normalize(pred_tokens, dim=-1), F.normalize(target_tokens.detach(), dim=-1), dim=-1)
    if target_mask is None:
        return 1.0 - cosine.mean()
    if target_mask.shape != cosine.shape:
        raise ValueError(f"target_mask must have shape {cosine.shape}")
    selected = target_mask.to(dtype=torch.bool, device=cosine.device)
    if not bool(selected.any()):
        return torch.zeros((), device=pred_tokens.device, dtype=pred_tokens.dtype)
    return 1.0 - cosine[selected].mean()


def vicreg_loss(latents: torch.Tensor, *, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    variance, covariance = vicreg_components(latents, target_std=target_std, eps=eps)
    return variance + covariance


def vicreg_components(latents: torch.Tensor, *, target_std: float = 1.0, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    values = latents.reshape(-1, latents.shape[-1])
    if values.shape[0] < 2:
        zero = torch.zeros((), device=latents.device, dtype=latents.dtype)
        return zero, zero
    std = torch.sqrt(values.var(dim=0, unbiased=False) + eps)
    variance = F.relu(float(target_std) - std).mean()
    centered = values - values.mean(dim=0, keepdim=True)
    normalized = centered / torch.sqrt(centered.var(dim=0, unbiased=False, keepdim=True) + eps)
    covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
    offdiag = covariance - torch.diag_embed(torch.diagonal(covariance))
    return variance, offdiag.pow(2).sum() / values.shape[1]


def barlow_cross_correlation_loss(rna_latents: torch.Tensor, image_latents: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    if rna_latents.shape != image_latents.shape:
        raise ValueError("barlow inputs must have matching shapes")
    x = rna_latents.reshape(-1, rna_latents.shape[-1])
    y = image_latents.reshape(-1, image_latents.shape[-1])
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x = (x - x.mean(dim=0, keepdim=True)) / torch.sqrt(x.var(dim=0, unbiased=False, keepdim=True) + eps)
    y = (y - y.mean(dim=0, keepdim=True)) / torch.sqrt(y.var(dim=0, unbiased=False, keepdim=True) + eps)
    corr = x.T @ y / x.shape[0]
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    return (corr - eye).pow(2).sum() / corr.shape[0]


def prototype_ot_jepa_loss(pred_prototypes: torch.Tensor, target_prototypes: torch.Tensor) -> torch.Tensor:
    if pred_prototypes.shape != target_prototypes.shape:
        raise ValueError("prototype tensors must have matching shapes")
    pred = F.normalize(pred_prototypes, dim=-1)
    target = F.normalize(target_prototypes.detach(), dim=-1)
    dist = torch.cdist(pred, target, p=2)
    return 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=2).values.mean())


def batch_mean_invariance_loss(latents: torch.Tensor, batch_id: torch.Tensor) -> torch.Tensor:
    values = F.normalize(latents.reshape(latents.shape[0], -1), dim=-1)
    labels = batch_id.reshape(-1).to(device=values.device)
    if values.shape[0] != labels.shape[0]:
        raise ValueError("batch_id must have one label per latent row")
    classes = torch.unique(labels)
    if classes.numel() < 2:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    global_mean = values.mean(dim=0, keepdim=True)
    loss = torch.zeros((), device=values.device, dtype=values.dtype)
    total = torch.zeros((), device=values.device, dtype=values.dtype)
    for label in classes:
        mask = labels == label
        weight = mask.float().mean()
        class_mean = values[mask].mean(dim=0, keepdim=True)
        loss = loss + weight * (class_mean - global_mean).pow(2).sum(dim=-1).mean()
        total = total + weight
    return loss / total.clamp_min(1e-8)


def bioaction_jepa_loss(
    outputs: Mapping[str, torch.Tensor],
    weights: BioActionJEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioActionJEPALossWeights()
    device = _infer_device(outputs)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {}
    jepa_total = torch.zeros((), device=device)
    aux_total = torch.zeros((), device=device)
    for field in fields(BioActionJEPALossWeights):
        if not field.name.endswith("_jepa"):
            continue
        if field.name == "distributional_jepa":
            continue
        available = outputs.get(f"{field.name}_available")
        if available is not None and not bool(available.detach().cpu()):
            component = torch.zeros((), device=device)
        else:
            component = cosine_jepa_loss(outputs[f"{field.name}_pred"], outputs[f"{field.name}_target"])
        weighted = float(getattr(weights, field.name)) * component
        total = total + weighted
        jepa_total = jepa_total + weighted
        diagnostics[f"loss/{field.name}"] = component.detach()
        diagnostics[f"weighted/{field.name}"] = weighted.detach()

    if float(weights.distributional_jepa) and "transition_joint_jepa_pred" in outputs:
        component = prototype_ot_jepa_loss(outputs["transition_joint_jepa_pred"], outputs["transition_joint_jepa_target"])
        weighted = float(weights.distributional_jepa) * component
        total = total + weighted
        jepa_total = jepa_total + weighted
        diagnostics["loss/distributional_jepa"] = component.detach()

    latents = [outputs[key] for key in ("rna_condition_state", "image_condition_state", "joint_condition_state", "shared_state") if key in outputs]
    if latents:
        stacked = torch.cat([value.reshape(value.shape[0], -1) for value in latents], dim=0)
        variance, covariance = vicreg_components(stacked, target_std=0.05)
        total = total + float(weights.vicreg_variance) * variance + float(weights.vicreg_covariance) * covariance
        diagnostics["loss/vicreg_variance"] = variance.detach()
        diagnostics["loss/vicreg_covariance"] = covariance.detach()
    if "rna_condition_state" in outputs and "image_condition_state" in outputs and float(weights.barlow_cross_correlation):
        barlow = barlow_cross_correlation_loss(outputs["rna_condition_state"], outputs["image_condition_state"])
        total = total + float(weights.barlow_cross_correlation) * barlow
        diagnostics["loss/barlow"] = barlow.detach()
    batch_id = outputs.get("batch_id_for_loss")
    if batch_id is not None and float(weights.batch_invariance):
        batch_losses = [
            batch_mean_invariance_loss(outputs[key], batch_id)
            for key in ("rna_condition_state", "image_condition_state", "joint_condition_state")
            if key in outputs
        ]
        if batch_losses:
            component = torch.stack(batch_losses).mean()
            weighted = float(weights.batch_invariance) * component
            total = total + weighted
            aux_total = aux_total + weighted
            diagnostics["loss/batch_invariance"] = component.detach()
            diagnostics["weighted/batch_invariance"] = weighted.detach()

    diagnostics["loss/total"] = total.detach()
    diagnostics["jepa_weighted_to_aux_ratio"] = (jepa_total.detach() / aux_total.detach().clamp_min(1e-8)) if aux_total.detach().abs() > 0 else torch.full((), float("inf"), device=device)
    diagnostics["raw_reconstruction_weighted_to_jepa_ratio"] = torch.zeros((), device=device)
    diagnostics["count_aux_weighted_to_jepa_ratio"] = torch.zeros((), device=device)
    return total, diagnostics


def collapse_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("rna_condition_state", "image_condition_state", "joint_condition_state"):
        value = outputs.get(name)
        if value is None:
            continue
        flat = value.detach().reshape(value.shape[0], -1)
        std = flat.std(dim=0, unbiased=False)
        result[f"latent/{name.replace('_condition_state', '')}_std_mean"] = float(std.mean().cpu())
        result[f"latent/{name.replace('_condition_state', '')}_effective_rank"] = float(_effective_rank(flat).cpu())
        result[f"latent/{name.replace('_condition_state', '')}_collapse_fraction_dims_below_0.05"] = float((std < 0.05).float().mean().cpu())
    pred = outputs.get("transition_joint_jepa_pred")
    if pred is not None:
        flat = pred.detach().reshape(-1, pred.shape[-1])
        result["latent/transition_pred_std_mean"] = float(flat.std(dim=0, unbiased=False).mean().cpu())
    shared = outputs.get("shared_state")
    if shared is not None and shared.shape[0] > 1:
        normalized = F.normalize(shared.detach(), dim=-1)
        cosine = normalized @ normalized.T
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
        result["latent/cosine_mean_pairwise"] = float(cosine[mask].mean().cpu())
    target = outputs.get("transition_joint_jepa_target")
    if pred is not None and target is not None:
        result["teacher/pred_target_cosine"] = float(F.cosine_similarity(pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1).mean().detach().cpu())
        result["teacher/target_norm_mean"] = float(target.detach().norm(dim=-1).mean().cpu())
        result["teacher/pred_norm_mean"] = float(pred.detach().norm(dim=-1).mean().cpu())
    return result


def _effective_rank(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros((), device=values.device)
    centered = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered)
    probs = spectrum / spectrum.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return torch.exp(entropy)


def _infer_device(outputs: Mapping[str, torch.Tensor]) -> torch.device:
    for value in outputs.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")
