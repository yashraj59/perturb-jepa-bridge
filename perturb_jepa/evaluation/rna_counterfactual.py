from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.metrics import pearson_delta, topk_de_recovery

DEFAULT_GROUPING_COLUMNS = {
    "condition": ("perturbation", "dose", "time", "cell_line"),
    "perturbation": ("perturbation",),
    "dose_time": ("perturbation", "dose", "time"),
    "heldout_perturbation": ("heldout_perturbation",),
}


def rna_counterfactual_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
    *,
    groupby: str | Sequence[str] | None = "condition",
    topk: Sequence[int] = (20, 50, 100),
    gene_sets: Mapping[str, Sequence[int]] | None = None,
    latent_predicted: np.ndarray | None = None,
    latent_observed: np.ndarray | None = None,
    prefix: str = "rna_counterfactual",
) -> dict[str, float]:
    """Evaluate RNA counterfactual predictions against observed profiles."""

    predicted = _as_sample_feature_array(predicted, name="predicted")
    observed = _as_sample_feature_array(observed, name="observed")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")
    control = _control_like(control, n_samples=predicted.shape[0], n_features=predicted.shape[1])
    frame = _as_metadata_frame(metadata, n_samples=predicted.shape[0])
    groups = _group_keys(frame, groupby, n_samples=predicted.shape[0])

    metrics = _rna_metrics_for_arrays(
        predicted,
        observed,
        control,
        topk=topk,
        gene_sets=gene_sets,
        prefix=prefix,
    )
    for group, indices in groups.items():
        if group == "overall":
            continue
        group_prefix = f"{prefix}_by_group={_metric_key(group)}"
        metrics.update(
            _rna_metrics_for_arrays(
                predicted[indices],
                observed[indices],
                control[indices],
                topk=topk,
                gene_sets=gene_sets,
                prefix=group_prefix,
            )
        )

    if latent_predicted is not None and latent_observed is not None:
        latent_pred = _as_sample_feature_array(latent_predicted, name="latent_predicted")
        latent_obs = _as_sample_feature_array(latent_observed, name="latent_observed")
        if latent_pred.shape != latent_obs.shape:
            raise ValueError("latent_predicted and latent_observed must have the same shape")
        metrics[f"{prefix}_latent_mmd"] = maximum_mean_discrepancy(latent_pred, latent_obs)
    return metrics


def maximum_mean_discrepancy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> float:
    x = _as_sample_feature_array(x, name="x")
    y = _as_sample_feature_array(y, name="y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have matching feature dimensions")
    if bandwidth is None:
        joined = np.vstack([x, y])
        distances = _squared_distances(joined, joined)
        nonzero = distances[distances > 0.0]
        bandwidth = float(np.sqrt(np.median(nonzero))) if nonzero.size else 1.0
    gamma = 1.0 / max(2.0 * bandwidth * bandwidth, 1e-12)
    k_xx = np.exp(-gamma * _squared_distances(x, x)).mean()
    k_yy = np.exp(-gamma * _squared_distances(y, y)).mean()
    k_xy = np.exp(-gamma * _squared_distances(x, y)).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def direction_accuracy(predicted_delta: np.ndarray, observed_delta: np.ndarray) -> float:
    pred = np.asarray(predicted_delta, dtype=float).ravel()
    obs = np.asarray(observed_delta, dtype=float).ravel()
    if pred.shape != obs.shape:
        raise ValueError("predicted_delta and observed_delta must have the same size")
    mask = obs != 0.0
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(pred[mask]) == np.sign(obs[mask])))


def pathway_score_correlations(
    predicted_delta: np.ndarray,
    observed_delta: np.ndarray,
    gene_sets: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    pred = _as_sample_feature_array(predicted_delta, name="predicted_delta")
    obs = _as_sample_feature_array(observed_delta, name="observed_delta")
    if pred.shape != obs.shape:
        raise ValueError("predicted_delta and observed_delta must have the same shape")
    result: dict[str, float] = {}
    for name, indices in gene_sets.items():
        index = np.asarray(list(indices), dtype=int)
        if index.size == 0:
            continue
        if np.any(index < 0) or np.any(index >= pred.shape[1]):
            raise ValueError(f"gene set {name!r} contains out-of-range indices")
        result[str(name)] = _safe_corr(pred[:, index].mean(axis=1), obs[:, index].mean(axis=1))
    return result


def _rna_metrics_for_arrays(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    *,
    topk: Sequence[int],
    gene_sets: Mapping[str, Sequence[int]] | None,
    prefix: str,
) -> dict[str, float]:
    pred_bulk = predicted.mean(axis=0)
    obs_bulk = observed.mean(axis=0)
    ctrl_bulk = control.mean(axis=0)
    pred_delta = predicted - control
    obs_delta = observed - control
    metrics = {
        f"{prefix}_pseudobulk_correlation": _safe_corr(pred_bulk, obs_bulk),
        f"{prefix}_logfc_correlation": pearson_delta(pred_bulk, obs_bulk, ctrl_bulk),
        f"{prefix}_direction_accuracy": direction_accuracy(pred_delta, obs_delta),
    }
    for k in topk:
        if k <= 0:
            raise ValueError("topk values must be positive")
        metrics[f"{prefix}_top{k}_de_overlap"] = topk_de_recovery(
            predicted,
            observed,
            control,
            k=min(k, predicted.shape[1]),
            direction="absolute",
        )
    if gene_sets:
        pathway_corrs = pathway_score_correlations(pred_delta, obs_delta, gene_sets)
        if pathway_corrs:
            metrics[f"{prefix}_pathway_score_correlation"] = float(np.mean(list(pathway_corrs.values())))
            for name, value in pathway_corrs.items():
                metrics[f"{prefix}_pathway_score_correlation:{_metric_key(name)}"] = value
    return metrics


def _group_keys(
    frame: pd.DataFrame,
    groupby: str | Sequence[str] | None,
    *,
    n_samples: int,
) -> dict[str, np.ndarray]:
    groups = {"overall": np.arange(n_samples)}
    if groupby is None:
        return groups
    columns = DEFAULT_GROUPING_COLUMNS[groupby] if isinstance(groupby, str) and groupby in DEFAULT_GROUPING_COLUMNS else groupby
    if isinstance(columns, str):
        columns = (columns,)
    columns = tuple(columns)
    available = [column for column in columns if column in frame.columns]
    if not available:
        return groups
    keys = [
        "|".join(_string_value(value) for value in row)
        for row in frame.loc[:, available].itertuples(index=False, name=None)
    ]
    for key in sorted(set(keys)):
        groups[key] = np.flatnonzero(np.asarray(keys, dtype=object) == key)
    return groups


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("correlation inputs must have the same size")
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _as_sample_feature_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _control_like(control: np.ndarray, *, n_samples: int, n_features: int) -> np.ndarray:
    array = np.asarray(control, dtype=float)
    if array.shape == (n_samples, n_features):
        return array
    if array.ndim > 2 and array.shape[0] == n_samples:
        return array.reshape(n_samples, -1)
    try:
        return np.broadcast_to(array, (n_samples, n_features)).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError("control must be broadcastable to predicted and observed") from exc


def _as_metadata_frame(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None,
    *,
    n_samples: int,
) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame(index=np.arange(n_samples))
    frame = metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if len(frame) != n_samples:
        raise ValueError("metadata row count must match array rows")
    return frame.reset_index(drop=True)


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)


def _metric_key(value: object) -> str:
    return _string_value(value).replace(" ", "_").replace("/", "_")
