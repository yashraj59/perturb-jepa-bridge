from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from perturb_jepa.operators.bioguard_residuals import (
    KernelResidualHead,
    ProgramActionResidualHead,
    ResidualTargetCache,
    SpectralResidualHead,
)
from perturb_jepa.training.bioguard_splits import ResidualFold
from perturb_jepa.training.biospectral_operator import LatentOperatorBundle, bundle_transition_metrics


FLOOR_TRANSITION = 0.0057
FLOOR_DELTA_COSINE = 0.3980
FLOOR_RECALL = 0.4815
FLOOR_RANK = 10.2835
FLOOR_MAGNITUDE = 0.7744


@dataclass(frozen=True)
class ResidualCandidateResult:
    candidate_id: str
    selected: bool
    decision_label: str
    cv_lcb_transition_gap: float
    cv_lcb_recall_gap: float
    mean_transition_gap: float
    mean_recall_gap: float
    mean_delta_cosine_gap: float
    mean_delta_rank_gap: float
    magnitude_ratio_mean: float
    residual_scale: float
    residual_gate_mean: float
    residual_gate_nonzero_fraction: float
    train_residual_fit_metric: float
    calibration_residual_fit_metric: float
    residual_train_to_calibration_gap: float
    action_negative_gap: float
    permutation_action_gap: float
    fold_rows: list[dict[str, Any]]


class FloorPreservationContract:
    def __init__(self, tolerance: float = 1.0e-8) -> None:
        self.tolerance = float(tolerance)

    def assert_preserved(self, floor_delta: np.ndarray, residual_delta: np.ndarray, *, gate: float, scale: float) -> float:
        pred = floor_delta + float(gate) * float(scale) * residual_delta
        max_diff = float(np.max(np.abs(pred - floor_delta)))
        if max_diff > self.tolerance:
            raise AssertionError(f"floor preservation failed with max diff {max_diff}")
        return max_diff


class ResidualRiskControlGate:
    def __init__(
        self,
        *,
        transition_lcb_min: float = 0.0,
        recall_lcb_min: float = 0.0,
        delta_cosine_gap_min: float = 0.0,
        magnitude_min: float = 0.4,
        magnitude_max: float = 1.4,
        max_train_to_calibration_gap: float = 0.05,
    ) -> None:
        self.transition_lcb_min = float(transition_lcb_min)
        self.recall_lcb_min = float(recall_lcb_min)
        self.delta_cosine_gap_min = float(delta_cosine_gap_min)
        self.magnitude_min = float(magnitude_min)
        self.magnitude_max = float(magnitude_max)
        self.max_train_to_calibration_gap = float(max_train_to_calibration_gap)

    def passes(self, result: ResidualCandidateResult) -> bool:
        action_negative_ok = result.action_negative_gap <= 0.0 or result.action_negative_gap < 0.25 * max(result.mean_transition_gap, 1.0e-8)
        return (
            result.cv_lcb_transition_gap > self.transition_lcb_min
            and result.cv_lcb_recall_gap >= self.recall_lcb_min
            and result.mean_delta_cosine_gap >= self.delta_cosine_gap_min
            and self.magnitude_min <= result.magnitude_ratio_mean <= self.magnitude_max
            and result.residual_train_to_calibration_gap <= self.max_train_to_calibration_gap
            and action_negative_ok
        )


class CrossFittedResidualSelector:
    def __init__(
        self,
        *,
        folds: list[ResidualFold],
        candidate_factory: Callable[[], Any],
        candidate_id: str,
        scales: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
        gate: ResidualRiskControlGate | None = None,
    ) -> None:
        self.folds = folds
        self.candidate_factory = candidate_factory
        self.candidate_id = candidate_id
        self.scales = scales
        self.gate = gate or ResidualRiskControlGate()

    def evaluate(self, train: LatentOperatorBundle) -> ResidualCandidateResult:
        cache = ResidualTargetCache.from_bundle(train)
        fold_payloads = []
        for fold in self.folds:
            fit = fold.fit_indices
            cal = fold.calibration_indices
            fold_cache = ResidualTargetCache.from_bundle(train, fit_rows=fit)
            candidate = self.candidate_factory().fit(
                fold_cache.source[fit],
                fold_cache.action[fit],
                fold_cache.residual_star[fit],
            )
            fit_residual_pred = candidate.predict(fold_cache.source[fit], fold_cache.action[fit])
            cal_residual_pred = candidate.predict(fold_cache.source[cal], fold_cache.action[cal])
            neg_action = _permuted_actions(fold_cache.action[cal], seed=fold.fold_id + 917)
            neg_residual_pred = candidate.predict(fold_cache.source[cal], neg_action)
            train_mse = float(np.mean((fit_residual_pred - fold_cache.residual_star[fit]) ** 2))
            cal_mse = float(np.mean((cal_residual_pred - fold_cache.residual_star[cal]) ** 2))
            fold_payloads.append(
                {
                    "fold": fold,
                    "floor_delta": fold_cache.floor_delta[cal],
                    "residual_pred": cal_residual_pred,
                    "negative_residual_pred": neg_residual_pred,
                    "fit_indices": fit,
                    "calibration_indices": cal,
                    "train_mse": train_mse,
                    "calibration_mse": cal_mse,
                    "bundle": _subset_bundle(train, cal),
                }
            )
        scale_rows = [self._score_scale(payloads=fold_payloads, scale=scale) for scale in self.scales]
        deployable_rows = [row for row in scale_rows if row["scale"] > 0.0]
        best = max(deployable_rows or scale_rows, key=lambda row: (row["cv_lcb_transition_gap"], row["mean_transition_gap"]))
        result = self._result_from_row(best)
        selected = self.gate.passes(result)
        decision = f"{self.candidate_id}_PASS_CV_GATE" if selected else f"{self.candidate_id}_FAILS_CV_GATE"
        return ResidualCandidateResult(
            candidate_id=result.candidate_id,
            selected=selected,
            decision_label=decision,
            cv_lcb_transition_gap=result.cv_lcb_transition_gap,
            cv_lcb_recall_gap=result.cv_lcb_recall_gap,
            mean_transition_gap=result.mean_transition_gap,
            mean_recall_gap=result.mean_recall_gap,
            mean_delta_cosine_gap=result.mean_delta_cosine_gap,
            mean_delta_rank_gap=result.mean_delta_rank_gap,
            magnitude_ratio_mean=result.magnitude_ratio_mean,
            residual_scale=result.residual_scale if selected else 0.0,
            residual_gate_mean=1.0 if selected else 0.0,
            residual_gate_nonzero_fraction=1.0 if selected else 0.0,
            train_residual_fit_metric=result.train_residual_fit_metric,
            calibration_residual_fit_metric=result.calibration_residual_fit_metric,
            residual_train_to_calibration_gap=result.residual_train_to_calibration_gap,
            action_negative_gap=result.action_negative_gap,
            permutation_action_gap=result.permutation_action_gap,
            fold_rows=result.fold_rows,
        )

    def _score_scale(self, *, payloads: list[dict[str, Any]], scale: float) -> dict[str, Any]:
        fold_rows = []
        for payload in payloads:
            bundle = payload["bundle"]
            floor_delta = payload["floor_delta"]
            pred_delta = floor_delta + float(scale) * payload["residual_pred"]
            neg_delta = floor_delta + float(scale) * payload["negative_residual_pred"]
            floor_metrics = bundle_transition_metrics(bundle, floor_delta)
            pred_metrics = bundle_transition_metrics(bundle, pred_delta)
            neg_metrics = bundle_transition_metrics(bundle, neg_delta)
            fold_rows.append(
                {
                    "fold_id": payload["fold"].fold_id,
                    "scale": float(scale),
                    "transition_improvement_gap": pred_metrics["transition_source_cosine_improvement"] - floor_metrics["transition_source_cosine_improvement"],
                    "recall_at_1_gap": pred_metrics["transition_to_target_recall@1"] - floor_metrics["transition_to_target_recall@1"],
                    "delta_cosine_gap": pred_metrics["delta_cosine"] - floor_metrics["delta_cosine"],
                    "delta_rank_gap": pred_metrics["delta_prediction_effective_rank"] - floor_metrics["delta_prediction_effective_rank"],
                    "magnitude_ratio": pred_metrics["delta_magnitude_ratio"],
                    "action_negative_gap": neg_metrics["transition_source_cosine_improvement"] - floor_metrics["transition_source_cosine_improvement"],
                    "permutation_action_gap": neg_metrics["transition_source_cosine_improvement"] - pred_metrics["transition_source_cosine_improvement"],
                    "train_residual_mse": payload["train_mse"],
                    "calibration_residual_mse": payload["calibration_mse"],
                    "fit_rows": int(payload["fit_indices"].size),
                    "calibration_rows": int(payload["calibration_indices"].size),
                }
            )
        transition = np.asarray([row["transition_improvement_gap"] for row in fold_rows], dtype=float)
        recall = np.asarray([row["recall_at_1_gap"] for row in fold_rows], dtype=float)
        train_mse = np.asarray([row["train_residual_mse"] for row in fold_rows], dtype=float)
        cal_mse = np.asarray([row["calibration_residual_mse"] for row in fold_rows], dtype=float)
        return {
            "candidate_id": self.candidate_id,
            "scale": float(scale),
            "cv_lcb_transition_gap": _lcb(transition),
            "cv_lcb_recall_gap": _lcb(recall),
            "mean_transition_gap": float(transition.mean()),
            "mean_recall_gap": float(recall.mean()),
            "mean_delta_cosine_gap": float(np.mean([row["delta_cosine_gap"] for row in fold_rows])),
            "mean_delta_rank_gap": float(np.mean([row["delta_rank_gap"] for row in fold_rows])),
            "magnitude_ratio_mean": float(np.mean([row["magnitude_ratio"] for row in fold_rows])),
            "train_residual_fit_metric": float(train_mse.mean()),
            "calibration_residual_fit_metric": float(cal_mse.mean()),
            "residual_train_to_calibration_gap": float(cal_mse.mean() - train_mse.mean()),
            "action_negative_gap": float(np.mean([row["action_negative_gap"] for row in fold_rows])),
            "permutation_action_gap": float(np.mean([row["permutation_action_gap"] for row in fold_rows])),
            "fold_rows": fold_rows,
        }

    def _result_from_row(self, row: dict[str, Any]) -> ResidualCandidateResult:
        return ResidualCandidateResult(
            candidate_id=row["candidate_id"],
            selected=False,
            decision_label="UNDECIDED",
            cv_lcb_transition_gap=row["cv_lcb_transition_gap"],
            cv_lcb_recall_gap=row["cv_lcb_recall_gap"],
            mean_transition_gap=row["mean_transition_gap"],
            mean_recall_gap=row["mean_recall_gap"],
            mean_delta_cosine_gap=row["mean_delta_cosine_gap"],
            mean_delta_rank_gap=row["mean_delta_rank_gap"],
            magnitude_ratio_mean=row["magnitude_ratio_mean"],
            residual_scale=row["scale"],
            residual_gate_mean=0.0,
            residual_gate_nonzero_fraction=0.0,
            train_residual_fit_metric=row["train_residual_fit_metric"],
            calibration_residual_fit_metric=row["calibration_residual_fit_metric"],
            residual_train_to_calibration_gap=row["residual_train_to_calibration_gap"],
            action_negative_gap=row["action_negative_gap"],
            permutation_action_gap=row["permutation_action_gap"],
            fold_rows=row["fold_rows"],
        )


def candidate_factory(candidate: str) -> Callable[[], Any]:
    if candidate == "spectral":
        return lambda: SpectralResidualHead(rank=8, alpha=1.0e-2)
    if candidate == "kernel":
        return lambda: KernelResidualHead(k=5, bandwidth=0.5)
    if candidate == "program":
        return lambda: ProgramActionResidualHead(alpha=1.0e-2)
    raise ValueError(f"unknown residual candidate: {candidate}")


def _subset_bundle(bundle: LatentOperatorBundle, indices: np.ndarray) -> LatentOperatorBundle:
    arrays = {key: value[indices] if value.shape[0] == bundle.source.shape[0] else value for key, value in bundle.arrays.items()}
    metadata = bundle.metadata.iloc[indices].reset_index(drop=True)
    return LatentOperatorBundle(name=f"{bundle.name}_subset", arrays=arrays, metadata=metadata)


def _permuted_actions(action: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(action.shape[0])
    rng.shuffle(order)
    return action[order]


def _lcb(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("-inf")
    if values.size == 1:
        return float(values[0])
    return float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(values.size))
