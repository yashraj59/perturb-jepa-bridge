from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from perturb_jepa.training.biospectral_operator import (
    LatentOperatorBundle,
    bundle_features,
    effective_rank,
    fit_ridge_numpy,
    predict_ridge_numpy,
    row_cosine,
)


class RidgeFloorHead:
    """Train-only full-ridge floor in NumPy form."""

    def __init__(self, *, alpha: float = 1.0e-2) -> None:
        self.alpha = float(alpha)
        self.fit_state = None

    def fit(self, source: np.ndarray, action: np.ndarray, delta: np.ndarray) -> "RidgeFloorHead":
        self.fit_state = fit_ridge_numpy(_features(source, action), delta, alpha=self.alpha)
        return self

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.fit_state is None:
            raise RuntimeError("RidgeFloorHead must be fit before predict")
        return predict_ridge_numpy(self.fit_state, _features(source, action))


@dataclass(frozen=True)
class ResidualTargetCache:
    source: np.ndarray
    action: np.ndarray
    teacher_delta: np.ndarray
    floor_delta: np.ndarray
    residual_star: np.ndarray
    metadata: pd.DataFrame
    floor: RidgeFloorHead
    fit_rows: np.ndarray

    @classmethod
    def from_bundle(
        cls,
        bundle: LatentOperatorBundle,
        *,
        fit_rows: np.ndarray | None = None,
        alpha: float = 1.0e-2,
    ) -> "ResidualTargetCache":
        rows = np.arange(bundle.source.shape[0]) if fit_rows is None else np.asarray(fit_rows, dtype=int)
        floor = RidgeFloorHead(alpha=alpha).fit(bundle.source[rows], bundle.action[rows], bundle.delta[rows])
        floor_delta = floor.predict(bundle.source, bundle.action)
        residual_star = bundle.delta - floor_delta
        return cls(
            source=bundle.source,
            action=bundle.action,
            teacher_delta=bundle.delta,
            floor_delta=floor_delta,
            residual_star=residual_star,
            metadata=bundle.metadata.reset_index(drop=True).copy(),
            floor=floor,
            fit_rows=rows,
        )


class ResidualWhitening:
    def __init__(self, eps: float = 1.0e-3) -> None:
        self.eps = float(eps)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, residual: np.ndarray) -> "ResidualWhitening":
        values = np.asarray(residual, dtype=np.float64)
        self.mean = values.mean(axis=0, keepdims=True)
        self.scale = values.std(axis=0, keepdims=True) + self.eps
        return self

    def transform(self, residual: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("ResidualWhitening must be fit before transform")
        return (np.asarray(residual, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, whitened: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("ResidualWhitening must be fit before inverse_transform")
        return np.asarray(whitened, dtype=np.float64) * self.scale + self.mean


class ZeroResidual:
    candidate_id = "zero"

    def fit(self, source: np.ndarray, action: np.ndarray, residual: np.ndarray) -> "ZeroResidual":
        self.dim = residual.shape[1]
        return self

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.zeros((source.shape[0], self.dim), dtype=np.float64)


class SpectralResidualHead:
    candidate_id = "spectral"

    def __init__(self, *, rank: int = 8, alpha: float = 1.0e-2) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.ridge = None

    def fit(self, source: np.ndarray, action: np.ndarray, residual: np.ndarray) -> "SpectralResidualHead":
        residual = np.asarray(residual, dtype=np.float64)
        self.mean = residual.mean(axis=0, keepdims=True)
        centered = residual - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        use_rank = max(1, min(self.rank, vt.shape[0]))
        self.basis = vt[:use_rank]
        coeff = centered @ self.basis.T
        self.ridge = fit_ridge_numpy(_features(source, action), coeff, alpha=self.alpha)
        return self

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.mean is None or self.basis is None or self.ridge is None:
            raise RuntimeError("SpectralResidualHead must be fit before predict")
        return self.mean + predict_ridge_numpy(self.ridge, _features(source, action)) @ self.basis


class KernelResidualHead:
    candidate_id = "kernel"

    def __init__(self, *, k: int = 5, bandwidth: float = 1.0) -> None:
        self.k = int(k)
        self.bandwidth = float(bandwidth)
        self.support_features: np.ndarray | None = None
        self.support_residual: np.ndarray | None = None

    def fit(self, source: np.ndarray, action: np.ndarray, residual: np.ndarray) -> "KernelResidualHead":
        features = _normalize(_features(source, action))
        self.support_features = features
        self.support_residual = np.asarray(residual, dtype=np.float64)
        return self

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.support_features is None or self.support_residual is None:
            raise RuntimeError("KernelResidualHead must be fit before predict")
        query = _normalize(_features(source, action))
        sims = query @ self.support_features.T
        k = max(1, min(self.k, self.support_features.shape[0]))
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        gathered = np.take_along_axis(sims, idx, axis=1)
        weights = np.exp((gathered - gathered.max(axis=1, keepdims=True)) / max(self.bandwidth, 1.0e-6))
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
        return np.einsum("nk,nkd->nd", weights, self.support_residual[idx])


class ProgramActionResidualHead:
    candidate_id = "program"

    def __init__(self, *, alpha: float = 1.0e-2) -> None:
        self.alpha = float(alpha)
        self.ridge = None

    def fit(self, source: np.ndarray, action: np.ndarray, residual: np.ndarray) -> "ProgramActionResidualHead":
        self.ridge = fit_ridge_numpy(np.asarray(action, dtype=np.float64), residual, alpha=self.alpha)
        return self

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.ridge is None:
            raise RuntimeError("ProgramActionResidualHead must be fit before predict")
        return predict_ridge_numpy(self.ridge, np.asarray(action, dtype=np.float64))


class PrototypeTransportResidualHead(KernelResidualHead):
    candidate_id = "prototype_transport"


class ResidualEnsembleHead:
    candidate_id = "ensemble"

    def __init__(self, heads: list[Any], weights: np.ndarray) -> None:
        self.heads = heads
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.sum() > 0:
            self.weights = self.weights / self.weights.sum()

    def predict(self, source: np.ndarray, action: np.ndarray) -> np.ndarray:
        if not self.heads:
            raise RuntimeError("ResidualEnsembleHead requires at least one head")
        preds = np.stack([head.predict(source, action) for head in self.heads], axis=0)
        return np.einsum("h,hnd->nd", self.weights, preds)


def residual_target_stats(cache: ResidualTargetCache) -> dict[str, float]:
    residual = cache.residual_star
    floor_error = cache.teacher_delta - cache.floor_delta
    return {
        "teacher_delta_rank": effective_rank(cache.teacher_delta),
        "floor_delta_rank": effective_rank(cache.floor_delta),
        "residual_target_rank": effective_rank(residual),
        "residual_target_magnitude": float(np.linalg.norm(residual, axis=1).mean()),
        "floor_error_magnitude": float(np.linalg.norm(floor_error, axis=1).mean()),
        "residual_near_zero_fraction": float((np.linalg.norm(residual, axis=1) < 1.0e-6).mean()),
        "residual_floor_cosine": float(row_cosine(residual, cache.floor_delta).mean()),
    }


def _features(source: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(source, dtype=np.float64), np.asarray(action, dtype=np.float64)), axis=1)


def _normalize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)
