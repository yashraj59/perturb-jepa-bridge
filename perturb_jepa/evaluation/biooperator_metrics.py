from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics


def biooperator_transition_metrics(
    source: np.ndarray,
    target: np.ndarray,
    pred_delta: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, float]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    pred_delta = np.asarray(pred_delta, dtype=float)
    pred = source + pred_delta
    true_delta = target - source
    pred_cos = row_cosine(pred, target)
    source_cos = row_cosine(source, target)
    delta_cos = row_cosine(pred_delta, true_delta)
    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    metrics = {
        "transition_source_cosine_improvement": float((pred_cos - source_cos).mean()),
        "absolute_target_cosine": float(pred_cos.mean()),
        "delta_cosine": float(delta_cos.mean()),
        "delta_magnitude_ratio": float(np.mean(pred_norm / np.maximum(true_norm, 1.0e-8))),
        "delta_prediction_effective_rank": effective_rank(pred_delta),
        "delta_teacher_effective_rank": effective_rank(true_delta),
        "source_improvement_hinge_violation_fraction": float((pred_cos < source_cos + 0.02).mean()),
    }
    if metadata is not None:
        frame = metadata_frame(metadata)
        metrics.update(
            directional_retrieval_metrics(
                l2_normalize(pred),
                l2_normalize(target),
                frame,
                frame,
                label_col="condition_key",
                ks=(1, 5, 10),
                prefix="transition_to_target",
                stratify_by=(),
            )
        )
    return metrics


def metadata_frame(metadata: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if "perturbation" not in frame.columns and "perturbation_id" in frame.columns:
        frame["perturbation"] = "pert_" + frame["perturbation_id"].astype(str)
    if "batch" not in frame.columns and "batch_id" in frame.columns:
        frame["batch"] = "batch_" + frame["batch_id"].astype(str)
    if "cell_line" not in frame.columns and "cell_line_id" in frame.columns:
        frame["cell_line"] = "cell_" + frame["cell_line_id"].astype(str)
    if "dose" not in frame.columns:
        frame["dose"] = "ignored"
    if "time" not in frame.columns:
        frame["time"] = "0"
    return frame


def row_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), eps)
    return np.sum(left * right, axis=1) / denom


def l2_normalize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def effective_rank(values: np.ndarray, eps: float = 1.0e-12) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    return float(np.exp(-np.sum(probs * np.log(np.maximum(probs, eps)))))
