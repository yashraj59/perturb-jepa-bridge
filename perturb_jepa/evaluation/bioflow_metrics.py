from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics, directional_retrieval_metrics
from perturb_jepa.models.bioflow_jepa import BioFlowJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.biotech_losses import biotech_collapse_diagnostics


def evaluate_bioflow_batches(
    model: BioFlowJEPA,
    batches: Iterable[BioActionConditionBatch],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    pred_z: list[torch.Tensor] = []
    target_z: list[torch.Tensor] = []
    source_z: list[torch.Tensor] = []
    pred_delta: list[torch.Tensor] = []
    teacher_delta: list[torch.Tensor] = []
    target_joint_tech: list[torch.Tensor] = []
    target_rna: list[np.ndarray] = []
    target_image: list[np.ndarray] = []
    labels: list[str] = []
    batch_labels: list[str] = []
    perturbation_labels: list[str] = []
    identity: dict[str, float] = {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": 0.0,
        "condition_key_feature_present": 0.0,
        "biological_key_onehot_present": 0.0,
        "teacher_stop_gradient_verified": 1.0,
        "EMA_target_present": 1.0,
        "transition_target_stop_gradient": 1.0,
        "separate_bio_and_tech_latents_present": 1.0,
        "z_bio_used_for_transition": 1.0,
        "z_tech_used_as_transition_shortcut": 0.0,
    }
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            outputs = model.forward_bioflow(batch)
            pred_z.append(outputs["z_pred"].detach().cpu())
            target_z.append(outputs["target_z_bio_teacher"].detach().cpu())
            source_z.append(outputs["source_z_bio_online"].detach().cpu())
            pred_delta.append(outputs["pred_delta"].detach().cpu())
            teacher_delta.append(outputs["true_delta"].detach().cpu())
            target_joint_tech.append(outputs["target_z_tech_teacher"].detach().cpu())
            labels.extend(batch.condition_key or [str(index) for index in range(outputs["z_pred"].shape[0])])
            batch_labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
            perturbation_labels.extend([f"pert_{int(value)}" for value in batch.perturbation_id.detach().cpu().tolist()])
            if "target_rna_z_bio" in outputs:
                target_rna.append(outputs["target_rna_z_bio"].detach().cpu().numpy())
            if "target_image_z_bio" in outputs:
                target_image.append(outputs["target_image_z_bio"].detach().cpu().numpy())
            for key, metric_name in (
                ("condition_key_exact_feature_present", "condition_key_feature_present"),
                ("biological_key_onehot_present", "biological_key_onehot_present"),
                ("pls_raw_linear_used_as_main_path", "pls_raw_linear_main_path_used"),
                ("teacher_stop_gradient_verified", "teacher_stop_gradient_verified"),
                ("ema_target_present", "EMA_target_present"),
                ("transition_target_stop_gradient", "transition_target_stop_gradient"),
                ("separate_bio_and_tech_latents_present", "separate_bio_and_tech_latents_present"),
                ("z_bio_used_for_transition", "z_bio_used_for_transition"),
                ("z_tech_used_as_transition_shortcut", "z_tech_used_as_transition_shortcut"),
            ):
                identity[metric_name] = float(outputs[key].detach().cpu())

    metrics: dict[str, float] = dict(identity)
    if not pred_z:
        return metrics
    pred = torch.cat(pred_z, dim=0)
    target = torch.cat(target_z, dim=0)
    source = torch.cat(source_z, dim=0)
    delta_pred = torch.cat(pred_delta, dim=0)
    delta_teacher = torch.cat(teacher_delta, dim=0)
    tech = torch.cat(target_joint_tech, dim=0)
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    source_cos = F.cosine_similarity(source, target, dim=-1)
    delta_cos = F.cosine_similarity(delta_pred, delta_teacher, dim=-1)
    metrics.update(
        {
            "transition_bio_cosine_to_teacher": float(pred_cos.mean().cpu()),
            "source_as_target_bio_cosine_to_teacher": float(source_cos.mean().cpu()),
            "transition_source_cosine_improvement": float((pred_cos - source_cos).mean().cpu()),
            "absolute_target_cosine": float(pred_cos.mean().cpu()),
            "delta_cosine": float(delta_cos.mean().cpu()),
            "delta_magnitude_ratio": float((delta_pred.norm(dim=-1) / delta_teacher.norm(dim=-1).clamp_min(1.0e-8)).mean().cpu()),
            "delta_prediction_effective_rank": _effective_rank_np(delta_pred.numpy()),
            "delta_teacher_effective_rank": _effective_rank_np(delta_teacher.numpy()),
            "delta_rank_ratio": _effective_rank_np(delta_pred.numpy()) / max(1.0e-8, _effective_rank_np(delta_teacher.numpy())),
            "source_improvement_hinge_violation_fraction": float((pred_cos < source_cos + model.config.source_improvement_margin).float().mean().cpu()),
            "std_mean_pred_delta": float(delta_pred.std(dim=0, unbiased=False).mean().cpu()),
            "std_mean_teacher_delta": float(delta_teacher.std(dim=0, unbiased=False).mean().cpu()),
        }
    )
    frame = pd.DataFrame(
        {
            "condition_key": labels,
            "perturbation": perturbation_labels,
            "batch": batch_labels,
            "dose": ["ignored" for _ in labels],
            "time": ["0" for _ in labels],
            "cell_line": ["context" for _ in labels],
        }
    )
    metrics.update(
        directional_retrieval_metrics(
            pred.numpy(),
            target.numpy(),
            frame,
            frame,
            label_col="condition_key",
            ks=(1, 5, 10),
            prefix="transition_to_target",
            stratify_by=(),
        )
    )
    metrics.update(
        biotech_collapse_diagnostics(
            {
                "joint_z_bio": target,
                "joint_z_tech": tech,
            }
        )
    )
    metrics.update(batch_probe_metrics(target.numpy(), frame, label_col="batch", prefix="joint_z_bio_batch_probe"))
    metrics.update(batch_probe_metrics(tech.numpy(), frame, label_col="batch", prefix="joint_z_tech_batch_probe"))
    metrics["bio_tech_batch_probe_accuracy_gap"] = _probe_gap(metrics, "joint_z_tech_batch_probe", "joint_z_bio_batch_probe")
    if target_rna and target_image:
        rna = np.concatenate(target_rna, axis=0)
        image = np.concatenate(target_image, axis=0)
        if rna.shape[0] == image.shape[0] == len(frame):
            metrics.update(cross_modal_retrieval_metrics(rna, image, frame, frame, ks=(1, 5, 10), stratify_by=()))
    else:
        metrics["rna_only_diagnostic"] = 1.0
    return metrics


def _effective_rank_np(values: np.ndarray, eps: float = 1.0e-12) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        return 0.0
    centered = array - array.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    return float(np.exp(-np.sum(probs * np.log(np.maximum(probs, eps)))))


def _probe_gap(metrics: dict[str, float], left_prefix: str, right_prefix: str) -> float:
    left = metrics.get(f"{left_prefix}_accuracy")
    right = metrics.get(f"{right_prefix}_accuracy")
    if left is None or right is None or np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left - right)
