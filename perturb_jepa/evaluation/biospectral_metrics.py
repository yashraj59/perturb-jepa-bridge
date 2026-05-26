from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from perturb_jepa.training.biospectral_operator import effective_rank, metadata_frame, row_cosine, spectral_entropy, transition_metrics


def biospectral_transition_metrics(
    source: np.ndarray,
    target: np.ndarray,
    pred_delta: np.ndarray,
    *,
    floor_delta: np.ndarray | None = None,
    residual_delta: np.ndarray | None = None,
    metadata: pd.DataFrame | dict[str, Any] | None = None,
    floor_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    frame = metadata_frame(metadata) if metadata is not None else None
    metrics = transition_metrics(source, target, pred_delta, frame)
    metrics["delta_prediction_spectral_entropy"] = spectral_entropy(pred_delta)
    metrics["delta_prediction_effective_rank"] = effective_rank(pred_delta)
    if floor_delta is not None:
        floor = transition_metrics(source, target, floor_delta, frame)
        metrics["floor_gap_transition_improvement"] = (
            metrics["transition_source_cosine_improvement"] - floor["transition_source_cosine_improvement"]
        )
        metrics["floor_gap_delta_cosine"] = metrics["delta_cosine"] - floor["delta_cosine"]
        metrics["floor_gap_recall@1"] = metrics.get("transition_to_target_recall@1", 0.0) - floor.get("transition_to_target_recall@1", 0.0)
        metrics["floor_gap_delta_rank"] = metrics["delta_prediction_effective_rank"] - floor["delta_prediction_effective_rank"]
    if floor_metrics is not None:
        metrics["floor_gap_transition_improvement_registered"] = (
            metrics["transition_source_cosine_improvement"] - floor_metrics["transition_source_cosine_improvement"]
        )
        metrics["floor_gap_delta_cosine_registered"] = metrics["delta_cosine"] - floor_metrics["delta_cosine"]
        metrics["floor_gap_recall@1_registered"] = metrics.get("transition_to_target_recall@1", 0.0) - floor_metrics["transition_to_target_recall@1"]
        metrics["floor_gap_delta_rank_registered"] = metrics["delta_prediction_effective_rank"] - floor_metrics["delta_prediction_effective_rank"]
    if residual_delta is not None and floor_delta is not None:
        floor_norm = np.linalg.norm(floor_delta, axis=1).mean()
        residual_norm = np.linalg.norm(residual_delta, axis=1).mean()
        metrics["residual_to_floor_norm_ratio"] = float(residual_norm / max(floor_norm, 1.0e-8))
        metrics["residual_cosine_with_floor"] = float(row_cosine(residual_delta, floor_delta).mean())
        metrics["residual_cosine_with_teacher_residual"] = float(row_cosine(residual_delta, (target - source) - floor_delta).mean())
    return metrics


def principal_angle_to_teacher_delta_subspace(teacher_delta: np.ndarray, pred_delta: np.ndarray, *, rank: int = 8) -> float:
    teacher_delta = np.asarray(teacher_delta, dtype=float)
    pred_delta = np.asarray(pred_delta, dtype=float)
    if teacher_delta.shape[0] < 2 or pred_delta.shape[0] < 2:
        return 0.0
    _, _, left_vt = np.linalg.svd(teacher_delta - teacher_delta.mean(axis=0, keepdims=True), full_matrices=False)
    _, _, right_vt = np.linalg.svd(pred_delta - pred_delta.mean(axis=0, keepdims=True), full_matrices=False)
    use_rank = min(rank, left_vt.shape[0], right_vt.shape[0])
    singular = np.linalg.svd(left_vt[:use_rank] @ right_vt[:use_rank].T, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
    return float(angles.mean()) if angles.size else 0.0
