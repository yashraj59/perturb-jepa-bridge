from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from perturb_jepa.training.biospectral_operator import effective_rank, row_cosine


SCALE_GRID = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass(frozen=True)
class CalibrationSelection:
    residual_scale: float
    residual_gate: float
    status: str
    cv_lcb_transition_gap: float
    cv_lcb_recall_gap: float
    cv_lcb_delta_cosine_gap: float
    mean_transition_gap: float
    mean_recall_gap: float
    action_negative_gap: float
    selected_row: dict[str, Any]


@dataclass(frozen=True)
class CalibrationResult:
    selected: bool
    selected_scale: float
    cv_lcb_transition_gap: float
    cv_lcb_recall_gap: float
    cv_lcb_delta_cosine_gap: float
    mean_transition_gap: float
    mean_recall_gap: float
    mean_delta_cosine_gap: float
    action_negative_gap: float
    fold_rows: list[dict[str, Any]]
    decision_label: str


def make_action_group_folds(action_ids: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    records = pd.DataFrame({"action_id": np.asarray(action_ids).astype(str)})
    folds = make_action_grouped_folds(records, n_folds=n_folds, action_key="action_id", seed=seed)
    return [(fold["fit_indices"], fold["calibration_indices"]) for fold in folds if fold["status"] == "OK"]


def make_action_grouped_folds(records: pd.DataFrame, n_folds: int = 3, action_key: str = "action_id", seed: int = 0) -> list[dict[str, Any]]:
    if action_key not in records.columns:
        raise ValueError(f"missing action key: {action_key}")
    actions = np.unique(records[action_key].astype(str).to_numpy())
    if actions.size < 2:
        return [{"status": "INSUFFICIENT_ACTION_GROUPS", "fit_indices": np.array([], dtype=int), "calibration_indices": np.array([], dtype=int)}]
    use_folds = min(max(2, int(n_folds)), int(actions.size))
    rng = np.random.default_rng(seed)
    shuffled = np.array(actions, dtype=object)
    rng.shuffle(shuffled)
    row_actions = records[action_key].astype(str).to_numpy()
    folds = []
    indices = np.arange(len(records))
    for fold_id in range(use_folds):
        cal_actions = np.asarray(shuffled[fold_id::use_folds], dtype=str)
        cal_mask = np.isin(row_actions, cal_actions)
        fit_actions = tuple(sorted(np.unique(row_actions[~cal_mask]).tolist()))
        calibration_actions = tuple(sorted(np.unique(row_actions[cal_mask]).tolist()))
        if set(fit_actions).intersection(calibration_actions):
            raise ValueError("action group leakage")
        folds.append(
            {
                "status": "OK",
                "fold_id": fold_id,
                "fit_indices": indices[~cal_mask],
                "calibration_indices": indices[cal_mask],
                "fit_actions": fit_actions,
                "calibration_actions": calibration_actions,
            }
        )
    return folds


def calibrate_residual_scale(
    floor_predictions: np.ndarray,
    residual_predictions: np.ndarray,
    teacher_targets: np.ndarray,
    action_ids: np.ndarray,
    candidate_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    min_lcb_transition_gap: float = 0.0001,
    min_lcb_recall_gap: float = 0.0,
    min_lcb_delta_cosine_gap: float = 0.0,
    max_fold_recall_drop: float = 0.05,
) -> CalibrationResult:
    floor_predictions = np.asarray(floor_predictions, dtype=np.float64)
    residual_predictions = np.asarray(residual_predictions, dtype=np.float64)
    teacher_targets = np.asarray(teacher_targets, dtype=np.float64)
    folds = make_action_group_folds(np.asarray(action_ids), n_folds=4, seed=0)
    if not folds:
        return CalibrationResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], "BGWM_RESIDUAL_REJECTED_USE_FLOOR")
    rows: list[dict[str, Any]] = []
    for fold_id, (_, calibration_indices) in enumerate(folds):
        floor = floor_predictions[calibration_indices]
        residual = residual_predictions[calibration_indices]
        target = teacher_targets[calibration_indices]
        floor_recall = _recall_at_1(floor, target)
        floor_cos = row_cosine(floor, target).mean()
        floor_delta_target = target - floor
        for scale in candidate_scales:
            pred = floor + float(scale) * residual
            pred_cos = row_cosine(pred, target).mean()
            pred_recall = _recall_at_1(pred, target)
            delta_gap = row_cosine(float(scale) * residual, floor_delta_target).mean() if float(scale) != 0.0 else 0.0
            rows.append(
                {
                    "fold_id": fold_id,
                    "scale": float(scale),
                    "floor_gap_transition": float(pred_cos - floor_cos),
                    "floor_gap_recall": float(pred_recall - floor_recall),
                    "floor_gap_delta_cosine": float(delta_gap),
                    "action_negative_gap": float(max(0.0, pred_cos - floor_cos)),
                }
            )
    selected = select_residual_scale_crossfit(
        rows,
        list(candidate_scales),
        min_lcb_transition_gap=min_lcb_transition_gap,
        min_lcb_recall_gap=min_lcb_recall_gap,
        min_lcb_delta_cosine_gap=min_lcb_delta_cosine_gap,
    )
    if selected.selected_row.get("min_fold_recall_gap", 0.0) < -float(max_fold_recall_drop):
        selected = CalibrationSelection(
            residual_scale=0.0,
            residual_gate=0.0,
            status="CALIBRATION_DEFAULT_TO_FLOOR",
            cv_lcb_transition_gap=0.0,
            cv_lcb_recall_gap=0.0,
            cv_lcb_delta_cosine_gap=0.0,
            mean_transition_gap=0.0,
            mean_recall_gap=0.0,
            action_negative_gap=0.0,
            selected_row={},
        )
    return CalibrationResult(
        selected=selected.residual_scale > 0.0,
        selected_scale=selected.residual_scale,
        cv_lcb_transition_gap=selected.cv_lcb_transition_gap,
        cv_lcb_recall_gap=selected.cv_lcb_recall_gap,
        cv_lcb_delta_cosine_gap=selected.cv_lcb_delta_cosine_gap,
        mean_transition_gap=selected.mean_transition_gap,
        mean_recall_gap=selected.mean_recall_gap,
        mean_delta_cosine_gap=selected.cv_lcb_delta_cosine_gap,
        action_negative_gap=selected.action_negative_gap,
        fold_rows=rows,
        decision_label="BGWM_RESIDUAL_SELECTED" if selected.residual_scale > 0.0 else "BGWM_RESIDUAL_REJECTED_USE_FLOOR",
    )


def compute_transition_metrics(source_z: np.ndarray, target_z: np.ndarray, floor_z: np.ndarray, pred_z: np.ndarray) -> dict[str, float]:
    source_z = np.asarray(source_z, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    floor_z = np.asarray(floor_z, dtype=np.float64)
    pred_z = np.asarray(pred_z, dtype=np.float64)
    pred_cos = row_cosine(pred_z, target_z)
    floor_cos = row_cosine(floor_z, target_z)
    source_cos = row_cosine(source_z, target_z)
    pred_delta = pred_z - source_z
    floor_delta = floor_z - source_z
    teacher_delta = target_z - source_z
    pred_delta_cos = row_cosine(pred_delta, teacher_delta)
    floor_delta_cos = row_cosine(floor_delta, teacher_delta)
    return {
        "transition_source_cosine_improvement": float((pred_cos - source_cos).mean()),
        "transition_floor_cosine_improvement": float((floor_cos - source_cos).mean()),
        "floor_gap_transition": float((pred_cos - floor_cos).mean()),
        "delta_cosine": float(pred_delta_cos.mean()),
        "floor_delta_cosine": float(floor_delta_cos.mean()),
        "floor_gap_delta_cosine": float(pred_delta_cos.mean() - floor_delta_cos.mean()),
        "recall_at_1": _recall_at_1(pred_z, target_z),
        "floor_recall_at_1": _recall_at_1(floor_z, target_z),
        "floor_gap_recall": _recall_at_1(pred_z, target_z) - _recall_at_1(floor_z, target_z),
        "delta_rank": effective_rank(pred_delta),
        "magnitude_ratio": float(np.linalg.norm(pred_delta, axis=1).mean() / max(np.linalg.norm(teacher_delta, axis=1).mean(), 1.0e-8)),
    }


def select_residual_scale_crossfit(
    fold_metrics: list[dict[str, float]],
    scale_grid: list[float] | None = None,
    *,
    min_lcb_transition_gap: float = 0.0001,
    min_lcb_recall_gap: float = 0.0,
    min_lcb_delta_cosine_gap: float = 0.0,
    require_action_negative_gap: bool = True,
) -> CalibrationSelection:
    scale_grid = SCALE_GRID if scale_grid is None else scale_grid
    rows = []
    for scale in scale_grid:
        subset = [row for row in fold_metrics if abs(float(row["scale"]) - float(scale)) < 1.0e-12]
        if not subset:
            continue
        transition = np.asarray([row["floor_gap_transition"] for row in subset], dtype=float)
        recall = np.asarray([row["floor_gap_recall"] for row in subset], dtype=float)
        delta_cosine = np.asarray([row["floor_gap_delta_cosine"] for row in subset], dtype=float)
        action_neg = np.asarray([row.get("action_negative_gap", 0.0) for row in subset], dtype=float)
        rows.append(
            {
                "scale": float(scale),
                "cv_lcb_transition_gap": _lcb(transition),
                "cv_lcb_recall_gap": _lcb(recall),
                "cv_lcb_delta_cosine_gap": _lcb(delta_cosine),
                "mean_transition_gap": float(transition.mean()),
                "mean_recall_gap": float(recall.mean()),
                "action_negative_gap": float(action_neg.mean()),
                "min_fold_recall_gap": float(recall.min()),
            }
        )
    passing = [
        row
        for row in rows
        if row["scale"] > 0.0
        and row["cv_lcb_transition_gap"] >= min_lcb_transition_gap
        and row["cv_lcb_recall_gap"] >= min_lcb_recall_gap
        and row["cv_lcb_delta_cosine_gap"] >= min_lcb_delta_cosine_gap
        and (row["action_negative_gap"] > 0.0 or not require_action_negative_gap)
        and row["min_fold_recall_gap"] >= -0.05
    ]
    if not passing:
        zero = next((row for row in rows if row["scale"] == 0.0), rows[0] if rows else {})
        return CalibrationSelection(
            residual_scale=0.0,
            residual_gate=0.0,
            status="CALIBRATION_DEFAULT_TO_FLOOR",
            cv_lcb_transition_gap=float(zero.get("cv_lcb_transition_gap", 0.0)),
            cv_lcb_recall_gap=float(zero.get("cv_lcb_recall_gap", 0.0)),
            cv_lcb_delta_cosine_gap=float(zero.get("cv_lcb_delta_cosine_gap", 0.0)),
            mean_transition_gap=float(zero.get("mean_transition_gap", 0.0)),
            mean_recall_gap=float(zero.get("mean_recall_gap", 0.0)),
            action_negative_gap=float(zero.get("action_negative_gap", 0.0)),
            selected_row=zero,
        )
    best = max(passing, key=lambda row: (row["cv_lcb_transition_gap"], row["cv_lcb_recall_gap"], -row["scale"]))
    return CalibrationSelection(
        residual_scale=float(best["scale"]),
        residual_gate=1.0,
        status="CALIBRATION_SELECTED_NONZERO_RESIDUAL",
        cv_lcb_transition_gap=float(best["cv_lcb_transition_gap"]),
        cv_lcb_recall_gap=float(best["cv_lcb_recall_gap"]),
        cv_lcb_delta_cosine_gap=float(best["cv_lcb_delta_cosine_gap"]),
        mean_transition_gap=float(best["mean_transition_gap"]),
        mean_recall_gap=float(best["mean_recall_gap"]),
        action_negative_gap=float(best["action_negative_gap"]),
        selected_row=best,
    )


def select_residual_scale_small_cap_continuous(
    fold_metrics: list[dict[str, float]],
    *,
    cap: float = 0.05,
    scale_grid: list[float] | None = None,
    require_nonnegative_min_recall: bool = True,
) -> CalibrationSelection:
    """Select a tiny residual scale from train-only continuous metrics.

    This is deliberately narrower than the default crossfit selector: it only
    considers scales up to a small pre-registered cap and ignores retrieval
    margin diagnostics, which F041 showed were over-conservative for tiny
    residuals. It still defaults exactly to the floor when train transition,
    delta, or recall evidence is negative.
    """

    base_grid = SCALE_GRID if scale_grid is None else scale_grid
    candidate_grid = [float(scale) for scale in base_grid if float(scale) <= float(cap) + 1.0e-12]
    if 0.0 not in candidate_grid:
        candidate_grid.insert(0, 0.0)
    rows = []
    for scale in candidate_grid:
        subset = [row for row in fold_metrics if abs(float(row["scale"]) - float(scale)) < 1.0e-12]
        if not subset:
            continue
        transition = np.asarray([row["floor_gap_transition"] for row in subset], dtype=float)
        recall = np.asarray([row["floor_gap_recall"] for row in subset], dtype=float)
        delta_cosine = np.asarray([row["floor_gap_delta_cosine"] for row in subset], dtype=float)
        action_neg = np.asarray([row.get("action_negative_gap", 0.0) for row in subset], dtype=float)
        rows.append(
            {
                "scale": float(scale),
                "cv_lcb_transition_gap": _lcb(transition),
                "cv_lcb_recall_gap": _lcb(recall),
                "cv_lcb_delta_cosine_gap": _lcb(delta_cosine),
                "mean_transition_gap": float(transition.mean()),
                "mean_recall_gap": float(recall.mean()),
                "mean_delta_cosine_gap": float(delta_cosine.mean()),
                "action_negative_gap": float(action_neg.mean()),
                "min_fold_recall_gap": float(recall.min()),
                "min_fold_transition_gap": float(transition.min()),
                "min_fold_delta_cosine_gap": float(delta_cosine.min()),
            }
        )
    passing = [
        row
        for row in rows
        if row["scale"] > 0.0
        and row["mean_transition_gap"] >= 0.0
        and row["mean_delta_cosine_gap"] >= 0.0
        and row["mean_recall_gap"] >= 0.0
        and (row["min_fold_recall_gap"] >= 0.0 or not require_nonnegative_min_recall)
    ]
    if not passing:
        zero = next((row for row in rows if row["scale"] == 0.0), rows[0] if rows else {})
        return CalibrationSelection(
            residual_scale=0.0,
            residual_gate=0.0,
            status="CALIBRATION_DEFAULT_TO_FLOOR",
            cv_lcb_transition_gap=float(zero.get("cv_lcb_transition_gap", 0.0)),
            cv_lcb_recall_gap=float(zero.get("cv_lcb_recall_gap", 0.0)),
            cv_lcb_delta_cosine_gap=float(zero.get("cv_lcb_delta_cosine_gap", 0.0)),
            mean_transition_gap=float(zero.get("mean_transition_gap", 0.0)),
            mean_recall_gap=float(zero.get("mean_recall_gap", 0.0)),
            action_negative_gap=float(zero.get("action_negative_gap", 0.0)),
            selected_row=zero,
        )
    best = max(passing, key=lambda row: (row["mean_transition_gap"], row["mean_delta_cosine_gap"], row["scale"]))
    return CalibrationSelection(
        residual_scale=float(best["scale"]),
        residual_gate=1.0,
        status="CALIBRATION_SELECTED_SMALL_CAP_CONTINUOUS_RESIDUAL",
        cv_lcb_transition_gap=float(best["cv_lcb_transition_gap"]),
        cv_lcb_recall_gap=float(best["cv_lcb_recall_gap"]),
        cv_lcb_delta_cosine_gap=float(best["cv_lcb_delta_cosine_gap"]),
        mean_transition_gap=float(best["mean_transition_gap"]),
        mean_recall_gap=float(best["mean_recall_gap"]),
        action_negative_gap=float(best["action_negative_gap"]),
        selected_row=best,
    )


def fit_calibrated_residual_gate(selection: CalibrationSelection) -> tuple[float, float]:
    return selection.residual_gate, selection.residual_scale


def _lcb(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    return float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(max(values.size, 1)))


def _recall_at_1(pred_z: np.ndarray, target_z: np.ndarray) -> float:
    pred = pred_z / np.maximum(np.linalg.norm(pred_z, axis=1, keepdims=True), 1.0e-8)
    target = target_z / np.maximum(np.linalg.norm(target_z, axis=1, keepdims=True), 1.0e-8)
    scores = pred @ target.T
    return float((scores.argmax(axis=1) == np.arange(scores.shape[0])).mean())
