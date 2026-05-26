from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from perturb_jepa.training.biospectral_operator import effective_rank, transition_metrics


def bioguard_wm_transition_metrics(
    source_z: np.ndarray,
    target_z: np.ndarray,
    floor_delta: np.ndarray,
    predicted_delta: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, float]:
    floor = transition_metrics(source_z, target_z, floor_delta, metadata)
    pred = transition_metrics(source_z, target_z, predicted_delta, metadata)
    return {
        "transition_source_cosine_improvement": pred["transition_source_cosine_improvement"],
        "absolute_target_cosine": pred["absolute_target_cosine"],
        "delta_cosine": pred["delta_cosine"],
        "transition_to_target_recall@1": pred.get("transition_to_target_recall@1", 0.0),
        "transition_to_target_recall@5": pred.get("transition_to_target_recall@5", 0.0),
        "transition_to_target_median_rank": pred.get("transition_to_target_median_rank", 0.0),
        "delta_prediction_effective_rank": pred["delta_prediction_effective_rank"],
        "delta_magnitude_ratio": pred["delta_magnitude_ratio"],
        "floor_gap_transition": pred["transition_source_cosine_improvement"] - floor["transition_source_cosine_improvement"],
        "floor_gap_recall@1": pred.get("transition_to_target_recall@1", 0.0) - floor.get("transition_to_target_recall@1", 0.0),
        "floor_gap_delta_cosine": pred["delta_cosine"] - floor["delta_cosine"],
        "floor_gap_delta_rank": pred["delta_prediction_effective_rank"] - floor["delta_prediction_effective_rank"],
        "z_bio_effective_rank": effective_rank(predicted_delta),
    }


def collapse_diagnostics(z_bio: np.ndarray, *, min_rank: float = 2.0) -> dict[str, Any]:
    rank = effective_rank(np.asarray(z_bio, dtype=np.float64))
    return {"effective_rank_z_bio": rank, "collapse_flag": bool(rank < float(min_rank))}


def write_metric_artifacts(root: Path, metrics: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["# BioGuard-WM Metrics", ""]
    rows.extend(f"- {key}: `{value}`" for key, value in sorted(metrics.items()))
    (root / "metrics.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
