from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


@dataclass(frozen=True)
class BridgeLossWeights:
    rna_mask: float = 1.0
    image_mask: float = 1.0
    jepa: float = 1.0
    align: float = 1.0
    perturbation_cls: float = 0.2
    batch_adv: float = 0.05
    counterfactual: float = 0.2


def bridge_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    rna_values: torch.Tensor | None = None,
    rna_mask: torch.Tensor | None = None,
    image_patches: torch.Tensor | None = None,
    image_patch_mask: torch.Tensor | None = None,
    perturbation_id: torch.Tensor | None = None,
    batch_id: torch.Tensor | None = None,
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
        terms["align"] = info_nce_loss(outputs["rna_shared"], outputs["image_shared"])
    if "counterfactual_image" in outputs and "image_shared" in outputs:
        terms["counterfactual_image"] = mmd_rbf(outputs["counterfactual_image"], outputs["image_shared"].detach())
    if perturbation_id is not None:
        if "rna_perturbation_logits" in outputs:
            terms["rna_perturbation_cls"] = F.cross_entropy(outputs["rna_perturbation_logits"], perturbation_id)
        if "image_perturbation_logits" in outputs:
            terms["image_perturbation_cls"] = F.cross_entropy(outputs["image_perturbation_logits"], perturbation_id)
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
    total = total + weights.perturbation_cls * (
        terms.get("rna_perturbation_cls", torch.zeros((), device=device))
        + terms.get("image_perturbation_cls", torch.zeros((), device=device))
    )
    total = total + weights.batch_adv * (
        terms.get("rna_batch_adv", torch.zeros((), device=device))
        + terms.get("image_batch_adv", torch.zeros((), device=device))
    )
    total = total + weights.counterfactual * terms.get("counterfactual_image", torch.zeros((), device=device))
    terms["total"] = total
    return total, terms
