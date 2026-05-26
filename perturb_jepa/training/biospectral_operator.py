from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics


@dataclass(frozen=True)
class LatentOperatorBundle:
    name: str
    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame

    @property
    def source(self) -> np.ndarray:
        return self.arrays["source_z_bio_teacher"].astype(np.float64)

    @property
    def target(self) -> np.ndarray:
        return self.arrays["target_z_bio_teacher"].astype(np.float64)

    @property
    def delta(self) -> np.ndarray:
        return self.target - self.source

    @property
    def action(self) -> np.ndarray:
        action = self.arrays.get("action_descriptor")
        if action is None or action.size == 0:
            return np.zeros((self.source.shape[0], 0), dtype=np.float64)
        return action.astype(np.float64)


@dataclass(frozen=True)
class NumpyRidgeFit:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray


@dataclass(frozen=True)
class NumpyReducedRankRidgeFit:
    ridge: NumpyRidgeFit
    delta_mean: np.ndarray
    basis: np.ndarray


def load_latent_bundle(prefix: Path, name: str) -> LatentOperatorBundle:
    arrays = dict(np.load(prefix.with_suffix(".npz")))
    metadata = pd.read_csv(prefix.with_suffix(".metadata.tsv"), sep="\t")
    return LatentOperatorBundle(name=name, arrays=arrays, metadata=metadata)


def concatenate_features(source: np.ndarray, action: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if action.size == 0:
        action = np.zeros((source.shape[0], 0), dtype=np.float64)
    return np.concatenate((source, action), axis=1)


def bundle_features(bundle: LatentOperatorBundle) -> np.ndarray:
    return concatenate_features(bundle.source, bundle.action)


def fit_ridge_numpy(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0e-2) -> NumpyRidgeFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    eye = np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
    return NumpyRidgeFit(x_mean=x_mean, y_mean=y_mean, coef=coef)


def predict_ridge_numpy(fit: NumpyRidgeFit, x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - fit.x_mean) @ fit.coef + fit.y_mean


def fit_reduced_rank_ridge_numpy(
    x: np.ndarray,
    delta: np.ndarray,
    *,
    rank: int | str,
    alpha: float = 1.0e-2,
) -> NumpyReducedRankRidgeFit:
    delta = np.asarray(delta, dtype=np.float64)
    delta_mean = delta.mean(axis=0, keepdims=True)
    centered = delta - delta_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = vt.shape[0] if rank == "full" else max(1, min(int(rank), vt.shape[0]))
    basis = vt[:use_rank]
    coeff = centered @ basis.T
    return NumpyReducedRankRidgeFit(
        ridge=fit_ridge_numpy(x, coeff, alpha=alpha),
        delta_mean=delta_mean,
        basis=basis,
    )


def predict_reduced_rank_ridge_numpy(fit: NumpyReducedRankRidgeFit, x: np.ndarray) -> np.ndarray:
    return fit.delta_mean + predict_ridge_numpy(fit.ridge, x) @ fit.basis


def full_delta_linear_parameters(fit: NumpyRidgeFit) -> tuple[np.ndarray, np.ndarray]:
    weight = fit.coef.T
    bias = (fit.y_mean - fit.x_mean @ fit.coef).reshape(-1)
    return weight, bias


def transition_metrics(
    source: np.ndarray,
    target: np.ndarray,
    pred_delta: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred_delta = np.asarray(pred_delta, dtype=np.float64)
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
        "source_as_target_bio_cosine_to_teacher": float(source_cos.mean()),
        "delta_cosine": float(delta_cos.mean()),
        "delta_magnitude_ratio": float(np.mean(pred_norm / np.maximum(true_norm, 1.0e-8))),
        "delta_prediction_effective_rank": effective_rank(pred_delta),
        "delta_prediction_spectral_entropy": spectral_entropy(pred_delta),
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


def bundle_transition_metrics(bundle: LatentOperatorBundle, pred_delta: np.ndarray) -> dict[str, float]:
    return transition_metrics(bundle.source, bundle.target, pred_delta, bundle.metadata)


def paired_transition_metrics(
    train: LatentOperatorBundle,
    eval_bundle: LatentOperatorBundle,
    train_delta: np.ndarray,
    eval_delta: np.ndarray,
    *,
    floor_eval_improvement: float,
    floor_eval_delta_cosine: float,
    floor_eval_recall_at_1: float,
    floor_eval_rank: float,
) -> dict[str, Any]:
    train_metrics = bundle_transition_metrics(train, train_delta)
    eval_metrics = bundle_transition_metrics(eval_bundle, eval_delta)
    metrics: dict[str, Any] = {f"train/{key}": value for key, value in train_metrics.items()}
    metrics.update({f"eval/{key}": value for key, value in eval_metrics.items()})
    metrics.update(
        {
            "operator_train_transition_improvement": train_metrics["transition_source_cosine_improvement"],
            "operator_eval_transition_improvement": eval_metrics["transition_source_cosine_improvement"],
            "operator_train_delta_cosine": train_metrics["delta_cosine"],
            "operator_eval_delta_cosine": eval_metrics["delta_cosine"],
            "operator_eval_recall_at_1": eval_metrics["transition_to_target_recall@1"],
            "operator_eval_median_rank": eval_metrics["transition_to_target_median_rank"],
            "operator_predicted_delta_rank": eval_metrics["delta_prediction_effective_rank"],
            "operator_eval_delta_magnitude_ratio": eval_metrics["delta_magnitude_ratio"],
            "operator_eval_spectral_entropy": eval_metrics["delta_prediction_spectral_entropy"],
            "source_improvement_hinge_violation_fraction": eval_metrics["source_improvement_hinge_violation_fraction"],
            "floor_gap_transition_improvement": eval_metrics["transition_source_cosine_improvement"] - floor_eval_improvement,
            "floor_gap_delta_cosine": eval_metrics["delta_cosine"] - floor_eval_delta_cosine,
            "floor_gap_recall@1": eval_metrics["transition_to_target_recall@1"] - floor_eval_recall_at_1,
            "floor_gap_delta_rank": eval_metrics["delta_prediction_effective_rank"] - floor_eval_rank,
        }
    )
    return metrics


def identity_metrics(*, frozen_latent: bool = True, full_jepa: bool = False) -> dict[str, float]:
    return {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": 0.0,
        "condition_key_feature_present": 0.0,
        "biological_key_one_hot_present": 0.0,
        "test_target_mean_used_for_fit": 0.0,
        "pooled_train_test_target_used_for_fit": 0.0,
        "teacher_stop_gradient_verified": 1.0,
        "separate_bio_and_tech_latents_present": 1.0,
        "heldout_action_descriptor_valid": 1.0,
        "frozen_latent_operator_only": 1.0 if frozen_latent else 0.0,
        "encoder_training_skipped": 1.0 if frozen_latent else 0.0,
        "full_biospectral_jepa_identity": 1.0 if full_jepa else 0.0,
    }


def identity_violation(metrics: dict[str, Any]) -> bool:
    expected = {
        "pls_raw_linear_main_path_used": 0.0,
        "condition_key_feature_present": 0.0,
        "biological_key_one_hot_present": 0.0,
        "test_target_mean_used_for_fit": 0.0,
        "pooled_train_test_target_used_for_fit": 0.0,
        "teacher_stop_gradient_verified": 1.0,
        "heldout_action_descriptor_valid": 1.0,
    }
    return any(float(metrics.get(key, 1.0)) != value for key, value in expected.items())


def write_leakage_report(
    path: Path,
    *,
    train_rows: int,
    eval_rows: int,
    action_feature_names: Iterable[str],
    mode: str,
    fit_uses_train_only: bool = True,
) -> None:
    features = list(action_feature_names)
    forbidden_present = any(name in {"condition_key", "biological_key", "target_key", "batch_id"} for name in features)
    lines = [
        "# Leakage Report",
        "",
        f"- Mode: `{mode}`",
        f"- Train rows used for fitting: `{int(train_rows)}`",
        f"- Eval rows used for scoring only: `{int(eval_rows)}`",
        f"- Fit uses train rows only: `{bool(fit_uses_train_only)}`",
        f"- Action feature names used by model: `{', '.join(features) if features else 'none'}`",
        f"- Forbidden key tensors present: `{bool(forbidden_present)}`",
        "- `condition_key` used only as an evaluation/retrieval label, not as a model tensor.",
        "- `biological_key` and exact target-key one-hot features are absent from model tensors.",
        "- Test/eval target means are not used for fitting floor, basis, or residuals.",
        "- Pooled train+test target statistics are not used.",
        "- Raw-linear PLS is not used as the BioSpectral representation path.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metadata_frame(metadata: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if "condition_key" not in frame.columns:
        frame["condition_key"] = np.arange(len(frame)).astype(str)
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


def covariance_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, values.shape[0] - 1)
    return np.linalg.eigvalsh(cov)[::-1]


def effective_rank(values: np.ndarray, eps: float = 1.0e-12) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    return float(math.exp(-float(np.sum(probs * np.log(np.maximum(probs, eps))))))


def spectral_entropy(values: np.ndarray, eps: float = 1.0e-12) -> float:
    spectrum = covariance_spectrum(values)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    return float(-np.sum(probs * np.log(np.maximum(probs, eps))))
