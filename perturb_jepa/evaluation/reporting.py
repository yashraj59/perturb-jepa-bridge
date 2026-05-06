from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.metrics import pearson_delta, spearman_delta, topk_jaccard

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def delta_mean_absolute_error(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
    pred_delta = np.asarray(predicted, dtype=float) - np.asarray(control, dtype=float)
    obs_delta = np.asarray(observed, dtype=float) - np.asarray(control, dtype=float)
    return float(np.mean(np.abs(pred_delta - obs_delta)))


def delta_mean_squared_error(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
    pred_delta = np.asarray(predicted, dtype=float) - np.asarray(control, dtype=float)
    obs_delta = np.asarray(observed, dtype=float) - np.asarray(control, dtype=float)
    return float(np.mean((pred_delta - obs_delta) ** 2))


def default_delta_metrics(*, topk: int = 50) -> dict[str, MetricFn]:
    """Default expression delta metrics used by baseline reports."""

    if topk <= 0:
        raise ValueError("topk must be positive")

    def topk_delta(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
        pred_delta = np.asarray(predicted, dtype=float) - np.asarray(control, dtype=float)
        obs_delta = np.asarray(observed, dtype=float) - np.asarray(control, dtype=float)
        return topk_jaccard(pred_delta.ravel(), obs_delta.ravel(), k=topk)

    return {
        "pearson_delta": pearson_delta,
        "spearman_delta": spearman_delta,
        "delta_mae": delta_mean_absolute_error,
        "delta_mse": delta_mean_squared_error,
        f"top{topk}_jaccard_delta": topk_delta,
    }


def grouped_metric_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
    *,
    groupby: str | Sequence[str] | None = None,
    metrics: Mapping[str, MetricFn] | None = None,
    include_overall: bool = True,
    topk: int = 50,
) -> pd.DataFrame:
    """Compute expression metrics overall and within metadata groups."""

    predicted = _as_sample_feature_array(predicted, name="predicted")
    observed = _as_sample_feature_array(observed, name="observed")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")
    if predicted.shape[0] == 0:
        raise ValueError("predicted and observed must contain at least one row")

    metric_fns = dict(default_delta_metrics(topk=topk) if metrics is None else metrics)
    records: list[dict[str, object]] = []
    n_samples, n_features = predicted.shape

    if include_overall:
        records.append(
            _evaluate_subset(
                predicted,
                observed,
                control,
                np.arange(n_samples),
                n_samples=n_samples,
                n_features=n_features,
                metric_fns=metric_fns,
                group_values={},
                group_label="overall",
            )
        )

    groupby_columns = _as_groupby_tuple(groupby)
    if groupby_columns is not None:
        frame = _as_metadata_frame(metadata, n_samples=n_samples)
        missing = [column for column in groupby_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"metadata is missing groupby columns: {missing}")
        grouped = frame.groupby(list(groupby_columns), sort=True, dropna=False).indices
        for key, indices in grouped.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_values = {
                column: _string_key_value(value)
                for column, value in zip(groupby_columns, key_tuple, strict=True)
            }
            records.append(
                _evaluate_subset(
                    predicted,
                    observed,
                    control,
                    np.asarray(indices, dtype=int),
                    n_samples=n_samples,
                    n_features=n_features,
                    metric_fns=metric_fns,
                    group_values=group_values,
                    group_label="|".join(group_values.values()),
                )
            )

    return pd.DataFrame.from_records(records)


def _evaluate_subset(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    indices: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
    metric_fns: Mapping[str, MetricFn],
    group_values: Mapping[str, str],
    group_label: str,
) -> dict[str, object]:
    subset_predicted = predicted[indices]
    subset_observed = observed[indices]
    subset_control = _slice_control(control, indices, n_samples=n_samples, n_features=n_features)
    record: dict[str, object] = {"group": group_label, **group_values, "n_samples": int(indices.size)}
    for name, metric_fn in metric_fns.items():
        record[name] = float(metric_fn(subset_predicted, subset_observed, subset_control))
    return record


def _slice_control(
    control: np.ndarray,
    indices: np.ndarray,
    *,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    control = np.asarray(control, dtype=float)
    if control.shape == (n_samples, n_features):
        return control[indices]
    if control.ndim > 2 and control.shape[0] == n_samples:
        return control[indices]
    if control.ndim == 1 and control.shape[0] == n_samples and n_features == 1:
        return control[indices].reshape(-1, 1)
    return control


def _as_sample_feature_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional")
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _as_metadata_frame(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None,
    *,
    n_samples: int,
) -> pd.DataFrame:
    if metadata is None:
        raise ValueError("metadata is required when groupby is provided")
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.reset_index(drop=True).copy()
    else:
        frame = pd.DataFrame(metadata).reset_index(drop=True)
    if len(frame) != n_samples:
        raise ValueError("metadata row count must match predicted rows")
    return frame


def _as_groupby_tuple(groupby: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if groupby is None:
        return None
    if isinstance(groupby, str):
        columns = (groupby,)
    else:
        columns = tuple(groupby)
    if not columns:
        return None
    return columns


def _string_key_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
