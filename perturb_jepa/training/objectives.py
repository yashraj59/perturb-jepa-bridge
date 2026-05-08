from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import torch
from torch import nn

from perturb_jepa.config import KendallUncertaintyConfig, ObjectiveScheduleConfig
from perturb_jepa.losses import BridgeLossWeights


RECONSTRUCTION_TERMS = frozenset({"rna_mask", "image_mask"})
BRIDGE_LOSS_TERM_WEIGHTS = {
    "rna_mask": "rna_mask",
    "image_mask": "image_mask",
    "rna_jepa": "jepa",
    "image_jepa": "jepa",
    "align": "align",
    "mmd": "mmd",
    "sliced_wasserstein": "sliced_wasserstein",
    "rna_perturbation_cls": "perturbation_cls",
    "image_perturbation_cls": "perturbation_cls",
    "rna_state_perturbation_adv": "state_perturbation_adv",
    "image_state_perturbation_adv": "state_perturbation_adv",
    "rna_batch_adv": "batch_adv",
    "image_batch_adv": "batch_adv",
    "counterfactual_rna": "counterfactual",
    "counterfactual_image": "counterfactual",
    "cycle": "cycle",
    "response_bottleneck": "response_bottleneck",
}


def schedule_scales(
    schedule: ObjectiveScheduleConfig | None,
    *,
    step: int,
) -> tuple[float, float]:
    if schedule is None or not schedule.enabled:
        return 1.0, 1.0
    if step < 0:
        raise ValueError("step must be non-negative")

    if step < schedule.reconstruction_warmup_steps:
        return 1.0, schedule.warmup_non_reconstruction_scale

    if schedule.reconstruction_anneal_steps == 0:
        progress = 1.0
    else:
        anneal_step = step - schedule.reconstruction_warmup_steps + 1
        progress = min(1.0, anneal_step / schedule.reconstruction_anneal_steps)

    reconstruction_scale = _lerp(1.0, schedule.reconstruction_final_scale, progress)
    non_reconstruction_scale = _lerp(schedule.warmup_non_reconstruction_scale, 1.0, progress)
    return reconstruction_scale, non_reconstruction_scale


def scheduled_loss_weights(
    weights: BridgeLossWeights | None,
    schedule: ObjectiveScheduleConfig | None,
    *,
    step: int,
) -> BridgeLossWeights:
    base = weights or BridgeLossWeights()
    reconstruction_scale, non_reconstruction_scale = schedule_scales(schedule, step=step)
    kwargs = {}
    for field in fields(BridgeLossWeights):
        scale = reconstruction_scale if field.name in RECONSTRUCTION_TERMS else non_reconstruction_scale
        kwargs[field.name] = float(getattr(base, field.name)) * scale
    return BridgeLossWeights(**kwargs)


class KendallUncertaintyWeighting(nn.Module):
    def __init__(
        self,
        term_names: tuple[str, ...],
        *,
        initial_log_variance: float = 0.0,
        loss_scale: float = 0.5,
        regularizer_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if not term_names:
            raise ValueError("term_names must contain at least one loss term")
        if len(set(term_names)) != len(term_names):
            raise ValueError("term_names must not contain duplicates")
        if loss_scale < 0.0:
            raise ValueError("loss_scale must be non-negative")
        if regularizer_scale < 0.0:
            raise ValueError("regularizer_scale must be non-negative")
        self.term_names = tuple(term_names)
        self.loss_scale = float(loss_scale)
        self.regularizer_scale = float(regularizer_scale)
        self.log_variances = nn.Parameter(
            torch.full((len(self.term_names),), float(initial_log_variance), dtype=torch.float32)
        )

    @classmethod
    def from_config(cls, config: KendallUncertaintyConfig) -> KendallUncertaintyWeighting | None:
        if not config.enabled:
            return None
        return cls(
            config.term_names,
            initial_log_variance=config.initial_log_variance,
            loss_scale=config.loss_scale,
            regularizer_scale=config.regularizer_scale,
        )

    def weight_term(
        self,
        name: str,
        component: torch.Tensor,
        *,
        include_regularizer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        try:
            index = self.term_names.index(name)
        except ValueError as exc:
            raise KeyError(f"uncertainty weighting is not configured for term: {name}") from exc

        log_variance = self.log_variances[index].to(device=component.device, dtype=component.dtype)
        precision = torch.exp(-log_variance)
        total = self.loss_scale * precision * component
        if include_regularizer:
            total = total + self.regularizer_scale * log_variance
        diagnostics = {
            f"uncertainty/{name}_log_variance": log_variance,
            f"uncertainty/{name}_precision": precision,
        }
        return total, diagnostics

    def forward(self, components: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = set(self.term_names).difference(components)
        if missing:
            missing_terms = ", ".join(sorted(missing))
            raise KeyError(f"uncertainty-weighted loss terms were not produced: {missing_terms}")

        total = torch.zeros((), device=_infer_device(components))
        diagnostics: dict[str, torch.Tensor] = {}
        for name in self.term_names:
            weighted, term_diagnostics = self.weight_term(name, components[name])
            total = total + weighted
            diagnostics.update(term_diagnostics)
        return total, diagnostics


def build_uncertainty_weighting(
    config: KendallUncertaintyConfig | KendallUncertaintyWeighting | None,
) -> KendallUncertaintyWeighting | None:
    if config is None:
        return None
    if isinstance(config, KendallUncertaintyWeighting):
        return config
    return KendallUncertaintyWeighting.from_config(config)


def weighted_bridge_total(
    raw_terms: Mapping[str, torch.Tensor],
    *,
    weights: BridgeLossWeights | None = None,
    schedule: ObjectiveScheduleConfig | None = None,
    step: int = 0,
    uncertainty_weighting: KendallUncertaintyWeighting | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scheduled_weights = scheduled_loss_weights(weights, schedule, step=step)
    reconstruction_scale, non_reconstruction_scale = schedule_scales(schedule, step=step)

    device = _infer_device(raw_terms)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {
        "schedule/reconstruction_scale": torch.tensor(reconstruction_scale, device=device),
        "schedule/non_reconstruction_scale": torch.tensor(non_reconstruction_scale, device=device),
    }
    uncertainty_terms = set(uncertainty_weighting.term_names) if uncertainty_weighting is not None else set()
    grouped_uncertainty_terms: dict[str, torch.Tensor] = {}

    for name, term in raw_terms.items():
        weight_name = _weight_name_for_term(name, scheduled_weights)
        if weight_name is None:
            static_weight = non_reconstruction_scale
        else:
            static_weight = float(getattr(scheduled_weights, weight_name))
        if name == "total":
            continue
        component = term * static_weight
        if uncertainty_weighting is not None and name in uncertainty_terms:
            weighted, term_diagnostics = uncertainty_weighting.weight_term(
                name,
                component,
                include_regularizer=static_weight > 0.0,
            )
            total = total + weighted
            diagnostics.update(term_diagnostics)
        elif uncertainty_weighting is not None and weight_name in uncertainty_terms:
            grouped_uncertainty_terms[weight_name] = grouped_uncertainty_terms.get(
                weight_name,
                torch.zeros((), device=component.device, dtype=component.dtype),
            ) + component
        else:
            total = total + component
        diagnostics[f"weighted/{name}"] = component

    for name, component in grouped_uncertainty_terms.items():
        weighted, term_diagnostics = uncertainty_weighting.weight_term(
            name,
            component,
            include_regularizer=bool(component.detach().abs().gt(0.0)),
        )
        total = total + weighted
        diagnostics[f"weighted/{name}"] = component
        diagnostics.update(term_diagnostics)

    if uncertainty_terms:
        missing = uncertainty_terms.difference(raw_terms).difference(grouped_uncertainty_terms)
        if missing:
            missing_terms = ", ".join(sorted(missing))
            raise KeyError(f"uncertainty-weighted loss terms were not produced: {missing_terms}")

    diagnostics["total"] = total
    return total, diagnostics


def _infer_device(raw_terms: Mapping[str, torch.Tensor]) -> torch.device:
    for value in raw_terms.values():
        return value.device
    return torch.device("cpu")


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _weight_name_for_term(name: str, weights: BridgeLossWeights) -> str | None:
    mapped = BRIDGE_LOSS_TERM_WEIGHTS.get(name)
    if mapped is not None and hasattr(weights, mapped):
        return mapped
    if hasattr(weights, name):
        return name
    return None
