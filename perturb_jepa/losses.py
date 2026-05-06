from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
    error = (prediction - target).pow(2)
    if mask is None:
        return error.mean()
    weights = mask.to(dtype=error.dtype)
    while weights.ndim < error.ndim:
        weights = weights.unsqueeze(-1)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def cosine_jepa_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(student, teacher.detach(), dim=-1).mean()


def info_nce_loss(x: torch.Tensor, y: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("InfoNCE inputs must have matching shapes")
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def supervised_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("supervised InfoNCE inputs must have matching shapes")
    if labels.shape != (x.shape[0],):
        raise ValueError("labels must have shape [batch]")
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    positive = labels[:, None].eq(labels[None, :])
    log_prob_xy = logits.log_softmax(dim=1)
    log_prob_yx = logits.T.log_softmax(dim=1)
    denom = positive.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
    loss_xy = -(log_prob_xy * positive.to(dtype=x.dtype)).sum(dim=1) / denom
    loss_yx = -(log_prob_yx * positive.to(dtype=x.dtype)).sum(dim=1) / denom
    return 0.5 * (loss_xy.mean() + loss_yx.mean())


def multi_resolution_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    label_levels: Sequence[torch.Tensor],
    *,
    weights: Sequence[float] | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if not label_levels:
        return info_nce_loss(x, y, temperature=temperature)
    if weights is None:
        weights = [1.0 / len(label_levels)] * len(label_levels)
    if len(weights) != len(label_levels):
        raise ValueError("weights and label_levels must have the same length")
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    norm = float(sum(weights))
    if norm <= 0:
        raise ValueError("multi-resolution InfoNCE weights must sum to a positive value")
    for weight, labels in zip(weights, label_levels, strict=True):
        total = total + float(weight) * supervised_info_nce_loss(x, y, labels, temperature=temperature)
    return total / norm


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD expects two matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD feature dimensions must match")

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(a, b).pow(2)
        value = torch.zeros_like(dist)
        for sigma in sigmas:
            value = value + torch.exp(-dist / (2.0 * sigma * sigma))
        return value / len(sigmas)

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()


def sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sliced Wasserstein expects two matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein feature dimensions must match")
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")
    projections = torch.randn(x.shape[1], num_projections, device=x.device, dtype=x.dtype, generator=generator)
    projections = F.normalize(projections, dim=0)
    x_proj = (x @ projections).sort(dim=0).values
    y_proj = (y @ projections).sort(dim=0).values
    if x_proj.shape[0] != y_proj.shape[0]:
        sample_count = min(x_proj.shape[0], y_proj.shape[0])
        x_idx = torch.linspace(0, x_proj.shape[0] - 1, sample_count, device=x.device).long()
        y_idx = torch.linspace(0, y_proj.shape[0] - 1, sample_count, device=y.device).long()
        x_proj = x_proj.index_select(0, x_idx)
        y_proj = y_proj.index_select(0, y_idx)
    return (x_proj - y_proj).pow(2).mean()


def bag_sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    num_projections: int = 32,
) -> torch.Tensor:
    if labels is None:
        return sliced_wasserstein_loss(x, y, num_projections=num_projections)
    if labels.shape != (x.shape[0],) or labels.shape != (y.shape[0],):
        raise ValueError("bag labels must have shape [batch] for both modalities")
    values = []
    for label in labels.unique(sorted=True):
        mask = labels.eq(label)
        if int(mask.sum()) == 0:
            continue
        values.append(sliced_wasserstein_loss(x[mask], y[mask], num_projections=num_projections))
    if not values:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.stack(values).mean()


@dataclass(frozen=True)
class BridgeLossWeights:
    rna_mask: float = 1.0
    image_mask: float = 1.0
    jepa: float = 1.0
    align: float = 1.0
    sliced_wasserstein: float = 0.25
    perturbation_cls: float = 0.2
    state_perturbation_adv: float = 0.1
    batch_adv: float = 0.05
    counterfactual: float = 0.2
    cycle: float = 0.1
    response_bottleneck: float = 0.01


def bridge_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    rna_values: torch.Tensor | None = None,
    rna_mask: torch.Tensor | None = None,
    image_patches: torch.Tensor | None = None,
    image_patch_mask: torch.Tensor | None = None,
    perturbation_id: torch.Tensor | None = None,
    batch_id: torch.Tensor | None = None,
    bag_labels: torch.Tensor | None = None,
    hierarchy_labels: Sequence[torch.Tensor] | None = None,
    weights: BridgeLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BridgeLossWeights()
    terms: dict[str, torch.Tensor] = {}

    if rna_values is not None and "rna_reconstruction" in outputs:
        terms["rna_mask"] = masked_mse(outputs["rna_reconstruction"], rna_values, rna_mask)
    if image_patches is not None and "image_patch_reconstruction" in outputs:
        terms["image_mask"] = masked_mse(outputs["image_patch_reconstruction"], image_patches, image_patch_mask)
    if "rna_shared" in outputs and "rna_teacher_shared" in outputs:
        terms["rna_jepa"] = cosine_jepa_loss(outputs["rna_shared"], outputs["rna_teacher_shared"])
    if "image_shared" in outputs and "image_teacher_shared" in outputs:
        terms["image_jepa"] = cosine_jepa_loss(outputs["image_shared"], outputs["image_teacher_shared"])
    if "rna_shared" in outputs and "image_shared" in outputs:
        if hierarchy_labels:
            terms["align"] = multi_resolution_info_nce_loss(
                outputs["rna_shared"],
                outputs["image_shared"],
                hierarchy_labels,
            )
        else:
            terms["align"] = info_nce_loss(outputs["rna_shared"], outputs["image_shared"])
        terms["sliced_wasserstein"] = bag_sliced_wasserstein_loss(
            outputs["rna_shared"],
            outputs["image_shared"],
            bag_labels,
        )
    if "counterfactual_image" in outputs and "image_shared" in outputs:
        terms["counterfactual_image"] = mmd_rbf(outputs["counterfactual_image"], outputs["image_shared"].detach())
    if "cycle_reconstruction" in outputs and "z_state" in outputs:
        terms["cycle"] = F.mse_loss(outputs["cycle_reconstruction"], outputs["z_state"].detach())
    response_terms = [
        outputs[name].pow(2).mean()
        for name in ("rna_response", "image_response", "z_response")
        if name in outputs
    ]
    if response_terms:
        terms["response_bottleneck"] = torch.stack(response_terms).mean()
    if perturbation_id is not None:
        if "rna_perturbation_logits" in outputs:
            terms["rna_perturbation_cls"] = F.cross_entropy(outputs["rna_perturbation_logits"], perturbation_id)
        if "image_perturbation_logits" in outputs:
            terms["image_perturbation_cls"] = F.cross_entropy(outputs["image_perturbation_logits"], perturbation_id)
        if "rna_state_perturbation_logits" in outputs:
            terms["rna_state_perturbation_adv"] = F.cross_entropy(
                outputs["rna_state_perturbation_logits"],
                perturbation_id,
            )
        if "image_state_perturbation_logits" in outputs:
            terms["image_state_perturbation_adv"] = F.cross_entropy(
                outputs["image_state_perturbation_logits"],
                perturbation_id,
            )
    if batch_id is not None:
        if "rna_batch_logits" in outputs:
            terms["rna_batch_adv"] = F.cross_entropy(outputs["rna_batch_logits"], batch_id)
        if "image_batch_logits" in outputs:
            terms["image_batch_adv"] = F.cross_entropy(outputs["image_batch_logits"], batch_id)

    device = next(iter(outputs.values())).device
    total = torch.zeros((), device=device)
    total = total + weights.rna_mask * terms.get("rna_mask", torch.zeros((), device=device))
    total = total + weights.image_mask * terms.get("image_mask", torch.zeros((), device=device))
    total = total + weights.jepa * (
        terms.get("rna_jepa", torch.zeros((), device=device))
        + terms.get("image_jepa", torch.zeros((), device=device))
    )
    total = total + weights.align * terms.get("align", torch.zeros((), device=device))
    total = total + weights.sliced_wasserstein * terms.get("sliced_wasserstein", torch.zeros((), device=device))
    total = total + weights.perturbation_cls * (
        terms.get("rna_perturbation_cls", torch.zeros((), device=device))
        + terms.get("image_perturbation_cls", torch.zeros((), device=device))
    )
    total = total + weights.state_perturbation_adv * (
        terms.get("rna_state_perturbation_adv", torch.zeros((), device=device))
        + terms.get("image_state_perturbation_adv", torch.zeros((), device=device))
    )
    total = total + weights.batch_adv * (
        terms.get("rna_batch_adv", torch.zeros((), device=device))
        + terms.get("image_batch_adv", torch.zeros((), device=device))
    )
    total = total + weights.counterfactual * terms.get("counterfactual_image", torch.zeros((), device=device))
    total = total + weights.cycle * terms.get("cycle", torch.zeros((), device=device))
    total = total + weights.response_bottleneck * terms.get("response_bottleneck", torch.zeros((), device=device))
    terms["total"] = total
    return total, terms
