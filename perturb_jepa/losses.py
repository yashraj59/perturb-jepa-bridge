from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.distribution_losses import (
    mmd_rbf_loss as prototype_mmd_rbf_loss,
    sliced_wasserstein_loss as prototype_sliced_wasserstein_loss,
)


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


def masked_cosine_jepa_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"JEPA shape mismatch: student={student.shape}, teacher={teacher.shape}")
    cosine = F.cosine_similarity(student, teacher.detach(), dim=-1)
    if mask is None:
        return 1.0 - cosine.mean()
    if mask.shape != cosine.shape:
        raise ValueError(f"JEPA mask must have shape {cosine.shape}, got {mask.shape}")
    selected = mask.to(device=cosine.device, dtype=torch.bool)
    if not bool(selected.any()):
        return torch.zeros((), device=student.device, dtype=student.dtype)
    return 1.0 - cosine[selected].mean()


def info_nce_loss(x: torch.Tensor, y: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("InfoNCE inputs must have matching shapes")
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _as_label_tensor(values: Any, *, device: torch.device) -> torch.Tensor | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.to(device=device)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        values = list(values)
        try:
            return torch.as_tensor(values, device=device)
        except (TypeError, ValueError):
            encoded: dict[Any, int] = {}
            labels: list[int] = []
            for value in values:
                if value not in encoded:
                    encoded[value] = len(encoded)
                labels.append(encoded[value])
            return torch.tensor(labels, device=device)
    return None


def _metadata_value_matrix(values: Sequence[Any], key: str, *, device: torch.device) -> torch.Tensor | None:
    extracted = [item.get(key) if isinstance(item, Mapping) else None for item in values]
    if all(value is None for value in extracted):
        return None
    encoded: dict[Any, int] = {}
    labels: list[int] = []
    for value in extracted:
        if value not in encoded:
            encoded[value] = len(encoded)
        labels.append(encoded[value])
    return torch.tensor(labels, device=device)


def _apply_label_positives(weights: torch.Tensor, labels: torch.Tensor, value: float) -> torch.Tensor:
    labels = labels.reshape(-1)
    if labels.shape != (weights.shape[0],):
        raise ValueError("positive labels must have shape [batch]")
    positive = labels[:, None].eq(labels[None, :])
    return torch.maximum(weights, positive.to(dtype=weights.dtype) * float(value))


def build_multi_positive_weights(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    positive_mask: torch.Tensor | None = None,
    bio_keys: Any = None,
    labels: Any = None,
    positive_weights: torch.Tensor | None = None,
    weak_positive_weight: float = 0.2,
) -> torch.Tensor:
    if positive_weights is not None:
        if positive_weights.shape != (batch_size, batch_size):
            raise ValueError(f"positive_weights must have shape {(batch_size, batch_size)}")
        weights = positive_weights.to(device=device, dtype=dtype)
    else:
        weights = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)

    if positive_mask is not None:
        if positive_mask.shape != (batch_size, batch_size):
            raise ValueError(f"positive_mask must have shape {(batch_size, batch_size)}")
        weights = torch.maximum(weights, positive_mask.to(device=device, dtype=dtype))

    label_tensor = _as_label_tensor(labels, device=device)
    if label_tensor is not None:
        weights = _apply_label_positives(weights, label_tensor, 1.0)

    if bio_keys is not None:
        if isinstance(bio_keys, Mapping):
            exact_keys = ("condition", "condition_id", "condition_key", "bio_key", "label")
            weak_keys = ("perturbation", "perturbation_id", "moa", "pathway")
            for key in exact_keys:
                label_tensor = _as_label_tensor(bio_keys.get(key), device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, 1.0)
            for key in weak_keys:
                label_tensor = _as_label_tensor(bio_keys.get(key), device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, weak_positive_weight)
        elif isinstance(bio_keys, Sequence) and bio_keys and isinstance(bio_keys[0], Mapping):
            for key in ("condition", "condition_id", "condition_key", "bio_key", "label"):
                label_tensor = _metadata_value_matrix(bio_keys, key, device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, 1.0)
            for key in ("perturbation", "perturbation_id", "moa", "pathway"):
                label_tensor = _metadata_value_matrix(bio_keys, key, device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, weak_positive_weight)
        else:
            label_tensor = _as_label_tensor(bio_keys, device=device)
            if label_tensor is not None:
                weights = _apply_label_positives(weights, label_tensor, 1.0)

    diagonal = torch.eye(batch_size, device=device, dtype=dtype)
    weights = torch.maximum(weights, diagonal)
    return weights


class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(
        self,
        *,
        temperature: float = 0.1,
        weak_positive_weight: float = 0.2,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if weak_positive_weight < 0:
            raise ValueError("weak_positive_weight must be non-negative")
        self.temperature = temperature
        self.weak_positive_weight = weak_positive_weight
        self.symmetric = symmetric

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        positive_mask: torch.Tensor | None = None,
        bio_keys: Any = None,
        labels: Any = None,
        positive_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("MultiPositiveInfoNCE inputs must have matching shapes")
        if x.ndim != 2:
            raise ValueError("MultiPositiveInfoNCE inputs must have shape [batch, features]")
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logits = x @ y.T / self.temperature
        weights = build_multi_positive_weights(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
            positive_mask=positive_mask,
            bio_keys=bio_keys,
            labels=labels,
            positive_weights=positive_weights,
            weak_positive_weight=self.weak_positive_weight,
        )

        def directional_loss(direction_logits: torch.Tensor, direction_weights: torch.Tensor) -> torch.Tensor:
            log_prob = direction_logits.log_softmax(dim=1)
            norm = direction_weights.sum(dim=1).clamp_min(torch.finfo(direction_weights.dtype).eps)
            return -((direction_weights * log_prob).sum(dim=1) / norm).mean()

        loss_xy = directional_loss(logits, weights)
        if not self.symmetric:
            return loss_xy
        return 0.5 * (loss_xy + directional_loss(logits.T, weights.T))


def multi_positive_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
    bio_keys: Any = None,
    labels: Any = None,
    positive_weights: torch.Tensor | None = None,
    temperature: float = 0.1,
    weak_positive_weight: float = 0.2,
    symmetric: bool = True,
) -> torch.Tensor:
    return MultiPositiveInfoNCELoss(
        temperature=temperature,
        weak_positive_weight=weak_positive_weight,
        symmetric=symmetric,
    )(
        x,
        y,
        positive_mask=positive_mask,
        bio_keys=bio_keys,
        labels=labels,
        positive_weights=positive_weights,
    )


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
    temperature: float = 0.1
    rna_mask: float = 1.0
    image_mask: float = 1.0
    jepa: float = 1.0
    align: float = 1.0
    mmd: float = 0.1
    sliced_wasserstein: float = 0.05
    perturbation_cls: float = 0.1
    state_perturbation_adv: float = 0.0
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
    rna_batch_id: torch.Tensor | None = None,
    image_batch_id: torch.Tensor | None = None,
    bag_labels: torch.Tensor | None = None,
    hierarchy_labels: Sequence[torch.Tensor] | None = None,
    bio_keys: Any = None,
    positive_mask: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    counterfactual_rna_target: torch.Tensor | None = None,
    counterfactual_control_rna: torch.Tensor | None = None,
    counterfactual_image_target: torch.Tensor | None = None,
    counterfactual_control_image: torch.Tensor | None = None,
    temperature: float = 0.1,
    weights: BridgeLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BridgeLossWeights()
    terms: dict[str, torch.Tensor] = {}

    if rna_values is not None and "rna_reconstruction" in outputs:
        terms["rna_mask"] = masked_mse(outputs["rna_reconstruction"], rna_values, rna_mask)
    if image_patches is not None and "image_patch_reconstruction" in outputs:
        terms["image_mask"] = masked_mse(outputs["image_patch_reconstruction"], image_patches, image_patch_mask)
    if "rna_token_prediction" in outputs and "rna_teacher_tokens" in outputs:
        terms["rna_jepa"] = masked_cosine_jepa_loss(
            outputs["rna_token_prediction"],
            outputs["rna_teacher_tokens"],
            rna_mask,
        )
    elif "rna_shared" in outputs and "rna_teacher_shared" in outputs:
        terms["rna_jepa"] = cosine_jepa_loss(outputs["rna_shared"], outputs["rna_teacher_shared"])
    if "image_patch_prediction" in outputs and "image_teacher_patches" in outputs:
        terms["image_jepa"] = masked_cosine_jepa_loss(
            outputs["image_patch_prediction"],
            outputs["image_teacher_patches"],
            image_patch_mask,
        )
    elif "image_shared" in outputs and "image_teacher_shared" in outputs:
        terms["image_jepa"] = cosine_jepa_loss(outputs["image_shared"], outputs["image_teacher_shared"])
    if "rna_shared" in outputs and "image_shared" in outputs:
        if bio_keys is not None or positive_mask is not None or positive_weights is not None:
            terms["align"] = multi_positive_info_nce_loss(
                outputs["rna_shared"],
                outputs["image_shared"],
                bio_keys=bio_keys,
                positive_mask=positive_mask,
                positive_weights=positive_weights,
                temperature=temperature,
            )
        elif hierarchy_labels:
            terms["align"] = multi_resolution_info_nce_loss(
                outputs["rna_shared"],
                outputs["image_shared"],
                hierarchy_labels,
                temperature=temperature,
            )
        else:
            terms["align"] = info_nce_loss(outputs["rna_shared"], outputs["image_shared"], temperature=temperature)
        if "rna_prototypes" in outputs and "image_prototypes" in outputs:
            terms["mmd"] = prototype_mmd_rbf_loss(outputs["rna_prototypes"], outputs["image_prototypes"])
            terms["sliced_wasserstein"] = prototype_sliced_wasserstein_loss(
                outputs["rna_prototypes"],
                outputs["image_prototypes"],
            )
        else:
            terms["sliced_wasserstein"] = bag_sliced_wasserstein_loss(
                outputs["rna_shared"],
                outputs["image_shared"],
                bag_labels,
            )
    if (
        counterfactual_rna_target is not None
        and counterfactual_control_rna is not None
        and "counterfactual_rna" in outputs
    ):
        terms["counterfactual_rna"] = mmd_rbf(outputs["counterfactual_rna"], counterfactual_rna_target.detach())
    if (
        counterfactual_image_target is not None
        and counterfactual_control_image is not None
        and "counterfactual_image" in outputs
    ):
        terms["counterfactual_image"] = mmd_rbf(outputs["counterfactual_image"], counterfactual_image_target.detach())
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
    rna_batch_target = rna_batch_id if rna_batch_id is not None else batch_id
    image_batch_target = image_batch_id if image_batch_id is not None else batch_id
    if rna_batch_target is not None and "rna_batch_logits" in outputs:
        terms["rna_batch_adv"] = F.cross_entropy(outputs["rna_batch_logits"], rna_batch_target)
    if image_batch_target is not None and "image_batch_logits" in outputs:
        terms["image_batch_adv"] = F.cross_entropy(outputs["image_batch_logits"], image_batch_target)

    device = next(iter(outputs.values())).device
    total = torch.zeros((), device=device)
    total = total + weights.rna_mask * terms.get("rna_mask", torch.zeros((), device=device))
    total = total + weights.image_mask * terms.get("image_mask", torch.zeros((), device=device))
    total = total + weights.jepa * (
        terms.get("rna_jepa", torch.zeros((), device=device))
        + terms.get("image_jepa", torch.zeros((), device=device))
    )
    total = total + weights.align * terms.get("align", torch.zeros((), device=device))
    total = total + weights.mmd * terms.get("mmd", torch.zeros((), device=device))
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
    total = total + weights.counterfactual * (
        terms.get("counterfactual_rna", torch.zeros((), device=device))
        + terms.get("counterfactual_image", torch.zeros((), device=device))
    )
    total = total + weights.cycle * terms.get("cycle", torch.zeros((), device=device))
    total = total + weights.response_bottleneck * terms.get("response_bottleneck", torch.zeros((), device=device))
    terms["total"] = total
    return total, terms
