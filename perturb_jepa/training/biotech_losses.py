from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
import torch.nn.functional as F

from perturb_jepa.training.bioaction_losses import cosine_jepa_loss, vicreg_components


@dataclass(frozen=True)
class BioTechJEPALossWeights:
    rna_program_jepa: float = 1.0
    image_region_jepa: float = 1.0
    rna_to_image_jepa: float = 2.0
    image_to_rna_jepa: float = 2.0
    transition_bio_jepa: float = 3.0
    z_tech_batch_ce: float = 0.5
    bio_tech_orthogonality: float = 0.1
    vicreg_bio_variance: float = 0.1
    vicreg_bio_covariance: float = 0.05
    vicreg_tech_variance: float = 0.05
    vicreg_tech_covariance: float = 0.02
    count_aux: float = 0.0


def biotech_jepa_loss(
    outputs: Mapping[str, torch.Tensor],
    weights: BioTechJEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioTechJEPALossWeights()
    device = _infer_device(outputs)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {}
    jepa_total = torch.zeros((), device=device)
    aux_total = torch.zeros((), device=device)

    for field in fields(BioTechJEPALossWeights):
        if not field.name.endswith("_jepa"):
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

    batch_logits = outputs.get("batch_logits_from_z_tech")
    batch_id = outputs.get("batch_id_for_loss")
    if (
        batch_logits is not None
        and batch_id is not None
        and batch_logits.shape[-1] > 1
        and float(weights.z_tech_batch_ce)
    ):
        component = F.cross_entropy(batch_logits, batch_id.reshape(-1).to(dtype=torch.long, device=batch_logits.device))
        weighted = float(weights.z_tech_batch_ce) * component
        total = total + weighted
        aux_total = aux_total + weighted
        diagnostics["loss/z_tech_batch_ce"] = component.detach()
        diagnostics["weighted/z_tech_batch_ce"] = weighted.detach()
    else:
        diagnostics["loss/z_tech_batch_ce"] = torch.zeros((), device=device)

    if "joint_z_bio" in outputs and "joint_z_tech" in outputs:
        orthogonality = cross_covariance_penalty(outputs["joint_z_bio"], outputs["joint_z_tech"])
        weighted = float(weights.bio_tech_orthogonality) * orthogonality
        total = total + weighted
        aux_total = aux_total + weighted
        diagnostics["loss/bio_tech_orthogonality"] = orthogonality.detach()
        diagnostics["weighted/bio_tech_orthogonality"] = weighted.detach()

    bio_latents = [outputs[key] for key in ("rna_z_bio", "image_z_bio", "joint_z_bio") if key in outputs]
    if bio_latents:
        variance, covariance = vicreg_components(torch.cat(bio_latents, dim=0), target_std=0.05)
        total = total + float(weights.vicreg_bio_variance) * variance + float(weights.vicreg_bio_covariance) * covariance
        diagnostics["loss/vicreg_bio_variance"] = variance.detach()
        diagnostics["loss/vicreg_bio_covariance"] = covariance.detach()
    tech_latents = [outputs[key] for key in ("rna_z_tech", "image_z_tech", "joint_z_tech") if key in outputs]
    if tech_latents:
        variance, covariance = vicreg_components(torch.cat(tech_latents, dim=0), target_std=0.05)
        total = total + float(weights.vicreg_tech_variance) * variance + float(weights.vicreg_tech_covariance) * covariance
        diagnostics["loss/vicreg_tech_variance"] = variance.detach()
        diagnostics["loss/vicreg_tech_covariance"] = covariance.detach()

    if float(weights.count_aux) and "count_aux_logits" in outputs and "count_aux_target" in outputs:
        component = F.smooth_l1_loss(outputs["count_aux_logits"], outputs["count_aux_target"].detach())
        weighted = float(weights.count_aux) * component
        total = total + weighted
        aux_total = aux_total + weighted
        diagnostics["loss/count_aux"] = component.detach()
        diagnostics["weighted/count_aux"] = weighted.detach()

    diagnostics["loss/total"] = total.detach()
    diagnostics["jepa_weighted_to_aux_ratio"] = (
        jepa_total.detach() / aux_total.detach().clamp_min(1e-8)
        if aux_total.detach().abs() > 0
        else torch.full((), float("inf"), device=device)
    )
    diagnostics["raw_reconstruction_weighted_to_jepa_ratio"] = torch.zeros((), device=device)
    return total, diagnostics


def biotech_collapse_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("rna_z_bio", "image_z_bio", "joint_z_bio", "rna_z_tech", "image_z_tech", "joint_z_tech"):
        value = outputs.get(key)
        if value is None:
            continue
        flat = value.detach().reshape(value.shape[0], -1)
        std = flat.std(dim=0, unbiased=False)
        result[f"latent/{key}_std_mean"] = float(std.mean().cpu())
        result[f"latent/{key}_effective_rank"] = float(_effective_rank(flat).cpu())
        result[f"latent/{key}_collapse_fraction_dims_below_0.05"] = float((std < 0.05).float().mean().cpu())
    pred = outputs.get("transition_bio_jepa_pred")
    target = outputs.get("transition_bio_jepa_target")
    if pred is not None:
        flat = pred.detach().reshape(-1, pred.shape[-1])
        result["latent/transition_bio_pred_std_mean"] = float(flat.std(dim=0, unbiased=False).mean().cpu())
    if pred is not None and target is not None:
        result["teacher/transition_bio_pred_target_cosine"] = float(
            F.cosine_similarity(pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1)
            .mean()
            .detach()
            .cpu()
        )
    if "joint_z_bio" in outputs and "joint_z_tech" in outputs:
        result["latent/joint_bio_tech_cross_covariance_abs_mean"] = float(
            cross_covariance_penalty(outputs["joint_z_bio"], outputs["joint_z_tech"]).detach().cpu()
        )
    return result


def cross_covariance_penalty(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if left.shape[0] < 2:
        return torch.zeros((), device=left.device, dtype=left.dtype)
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    left = (left - left.mean(dim=0, keepdim=True)) / torch.sqrt(left.var(dim=0, unbiased=False, keepdim=True) + eps)
    right = (right - right.mean(dim=0, keepdim=True)) / torch.sqrt(right.var(dim=0, unbiased=False, keepdim=True) + eps)
    return (left.T @ right / left.shape[0]).pow(2).mean()


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
