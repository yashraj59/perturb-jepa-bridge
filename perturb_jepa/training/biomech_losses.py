from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from perturb_jepa.training.biotech_losses import cross_covariance_penalty
from perturb_jepa.training.bioaction_losses import cosine_jepa_loss, prototype_ot_jepa_loss, vicreg_components


@dataclass(frozen=True)
class BioMechanisticJEPALossWeights:
    delta_jepa: float = 3.0
    target_transition_jepa: float = 1.5
    delta_mse: float = 0.1
    control_zero: float = 0.2
    rna_to_image_jepa: float = 1.5
    image_to_rna_jepa: float = 1.5
    rna_program_jepa: float = 0.5
    image_region_jepa: float = 0.5
    prototype_transition_jepa: float = 1.0
    prototype_set: float = 0.5
    prototype_diversity: float = 0.05
    z_tech_batch_ce: float = 0.3
    bio_tech_orthogonality: float = 0.1
    vicreg_bio_variance: float = 0.1
    vicreg_bio_covariance: float = 0.05
    vicreg_delta_variance: float = 0.1
    vicreg_delta_covariance: float = 0.05
    vicreg_tech_variance: float = 0.05
    vicreg_tech_covariance: float = 0.02


def biomech_jepa_loss(
    outputs: Mapping[str, torch.Tensor],
    weights: BioMechanisticJEPALossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BioMechanisticJEPALossWeights()
    device = _infer_device(outputs)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {}
    main = _available_cosine(outputs, "delta_jepa")
    weighted_main = float(weights.delta_jepa) * main
    total = total + weighted_main
    diagnostics["loss/delta_jepa"] = main.detach()
    diagnostics["weighted/delta_jepa"] = weighted_main.detach()
    main_abs = weighted_main.detach().abs().clamp_min(1e-8)

    for name, weight in (
        ("target_transition_jepa", weights.target_transition_jepa),
        ("rna_to_image_jepa", weights.rna_to_image_jepa),
        ("image_to_rna_jepa", weights.image_to_rna_jepa),
        ("rna_program_jepa", weights.rna_program_jepa),
        ("image_region_jepa", weights.image_region_jepa),
        ("prototype_transition_jepa", weights.prototype_transition_jepa),
    ):
        component = _available_cosine(outputs, name)
        weighted = float(weight) * component
        total = total + weighted
        diagnostics[f"loss/{name}"] = component.detach()
        diagnostics[f"weighted/{name}"] = weighted.detach()
        diagnostics[f"weighted_to_main/{name}"] = weighted.detach().abs() / main_abs

    if "delta_pred" in outputs and "delta_teacher" in outputs:
        component = F.smooth_l1_loss(outputs["delta_pred"], outputs["delta_teacher"].detach())
        weighted = float(weights.delta_mse) * component
        total = total + weighted
        diagnostics["loss/delta_mse"] = component.detach()
        diagnostics["weighted/delta_mse"] = weighted.detach()
        diagnostics["weighted_to_main/delta_mse"] = weighted.detach().abs() / main_abs

    batch_id = outputs.get("batch_id_for_loss")
    if batch_id is not None and "delta_pred" in outputs and float(weights.control_zero):
        control_mask = batch_id.new_zeros(batch_id.shape, dtype=torch.bool)
        # Control rows are rare in condition-pair training; this term is active
        # when tests or future loaders include control->control pairs.
        if "perturbation_id_for_loss" in outputs:
            control_mask = outputs["perturbation_id_for_loss"].eq(0)
        if bool(control_mask.any()):
            component = outputs["delta_pred"][control_mask].pow(2).mean()
            weighted = float(weights.control_zero) * component
            total = total + weighted
            diagnostics["loss/control_zero"] = component.detach()
            diagnostics["weighted/control_zero"] = weighted.detach()

    if outputs.get("prototype_transition_jepa_available") is not None and bool(outputs["prototype_transition_jepa_available"].detach().cpu()):
        component = prototype_ot_jepa_loss(outputs["prototype_transition_jepa_pred"], outputs["prototype_transition_jepa_target"])
        weighted = float(weights.prototype_set) * component
        total = total + weighted
        diagnostics["loss/prototype_set"] = component.detach()
        diagnostics["weighted/prototype_set"] = weighted.detach()
        diagnostics["weighted_to_main/prototype_set"] = weighted.detach().abs() / main_abs
        diversity = prototype_diversity_loss(outputs["prototype_transition_jepa_pred"])
        weighted_div = float(weights.prototype_diversity) * diversity
        total = total + weighted_div
        diagnostics["loss/prototype_diversity"] = diversity.detach()
        diagnostics["weighted/prototype_diversity"] = weighted_div.detach()

    if "joint_z_bio" in outputs and "joint_z_tech" in outputs:
        orthogonality = cross_covariance_penalty(outputs["joint_z_bio"], outputs["joint_z_tech"])
        weighted = float(weights.bio_tech_orthogonality) * orthogonality
        total = total + weighted
        diagnostics["loss/bio_tech_orthogonality"] = orthogonality.detach()
        diagnostics["weighted/bio_tech_orthogonality"] = weighted.detach()
        diagnostics["weighted_to_main/bio_tech_orthogonality"] = weighted.detach().abs() / main_abs

    batch_logits = outputs.get("batch_logits_from_z_tech")
    if batch_logits is not None and batch_id is not None and batch_logits.shape[-1] > 1 and float(weights.z_tech_batch_ce):
        component = F.cross_entropy(batch_logits, batch_id.to(dtype=torch.long, device=batch_logits.device))
        weighted = float(weights.z_tech_batch_ce) * component
        total = total + weighted
        diagnostics["loss/z_tech_batch_ce"] = component.detach()
        diagnostics["weighted/z_tech_batch_ce"] = weighted.detach()
        diagnostics["weighted_to_main/z_tech_batch_ce"] = weighted.detach().abs() / main_abs

    _add_vicreg(diagnostics, outputs, total_ref=[total], keys=("rna_z_bio", "image_z_bio", "joint_z_bio"), prefix="bio", variance_weight=weights.vicreg_bio_variance, covariance_weight=weights.vicreg_bio_covariance)
    total = total + diagnostics.pop("_vicreg_bio_weighted", torch.zeros((), device=device))
    _add_vicreg(diagnostics, outputs, total_ref=[total], keys=("delta_pred",), prefix="delta", variance_weight=weights.vicreg_delta_variance, covariance_weight=weights.vicreg_delta_covariance)
    total = total + diagnostics.pop("_vicreg_delta_weighted", torch.zeros((), device=device))
    _add_vicreg(diagnostics, outputs, total_ref=[total], keys=("rna_z_tech", "image_z_tech", "joint_z_tech"), prefix="tech", variance_weight=weights.vicreg_tech_variance, covariance_weight=weights.vicreg_tech_covariance)
    total = total + diagnostics.pop("_vicreg_tech_weighted", torch.zeros((), device=device))

    diagnostics["loss/total"] = total.detach()
    diagnostics["main_delta_weighted"] = weighted_main.detach()
    diagnostics["auxiliary_weighted_to_main_ratio"] = (total.detach().abs() - weighted_main.detach().abs()).clamp_min(0.0) / main_abs
    diagnostics["raw_reconstruction_weighted_to_jepa_ratio"] = torch.zeros((), device=device)
    return total, diagnostics


def biomech_collapse_diagnostics(outputs: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("joint_z_bio", "joint_z_tech", "delta_pred", "delta_teacher", "predicted_target_bio_prototypes"):
        value = outputs.get(key)
        if value is None:
            continue
        flat = value.detach().reshape(-1, value.shape[-1])
        std = flat.std(dim=0, unbiased=False)
        result[f"latent/{key}_std_mean"] = float(std.mean().cpu())
        result[f"latent/{key}_effective_rank"] = float(_effective_rank(flat).cpu())
        result[f"latent/{key}_collapse_fraction_dims_below_0.05"] = float((std < 0.05).float().mean().cpu())
    if "delta_pred" in outputs and "delta_teacher" in outputs:
        result["delta_cosine"] = float(F.cosine_similarity(outputs["delta_pred"], outputs["delta_teacher"].detach(), dim=-1).mean().detach().cpu())
    if "z_target_pred" in outputs and "z_target_teacher_bio" in outputs:
        result["absolute_target_cosine"] = float(F.cosine_similarity(outputs["z_target_pred"], outputs["z_target_teacher_bio"].detach(), dim=-1).mean().detach().cpu())
    return result


def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    if prototypes.shape[1] < 2:
        return torch.zeros((), device=prototypes.device, dtype=prototypes.dtype)
    values = F.normalize(prototypes, dim=-1)
    cosine = values @ values.transpose(1, 2)
    eye = torch.eye(cosine.shape[1], dtype=torch.bool, device=cosine.device)[None, :, :]
    return F.relu(cosine.masked_fill(eye, 0.0)).mean()


def _available_cosine(outputs: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    device = _infer_device(outputs)
    available = outputs.get(f"{name}_available")
    if available is not None and not bool(available.detach().cpu()):
        return torch.zeros((), device=device)
    return cosine_jepa_loss(outputs[f"{name}_pred"], outputs[f"{name}_target"])


def _add_vicreg(
    diagnostics: dict[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    *,
    total_ref: list[torch.Tensor],
    keys: tuple[str, ...],
    prefix: str,
    variance_weight: float,
    covariance_weight: float,
) -> None:
    latents = [outputs[key].reshape(-1, outputs[key].shape[-1]) for key in keys if key in outputs]
    if not latents:
        return
    values = torch.cat(latents, dim=0)
    variance, covariance = vicreg_components(values, target_std=0.05)
    weighted = float(variance_weight) * variance + float(covariance_weight) * covariance
    diagnostics[f"loss/vicreg_{prefix}_variance"] = variance.detach()
    diagnostics[f"loss/vicreg_{prefix}_covariance"] = covariance.detach()
    diagnostics[f"weighted/vicreg_{prefix}"] = weighted.detach()
    diagnostics[f"_vicreg_{prefix}_weighted"] = weighted


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
