from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.metrics import (
    dose_response_monotonicity,
    pearson_delta,
    spearman_delta,
    topk_de_recovery,
    topk_differential_expression_recovery,
    topk_jaccard,
)

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
        f"top{topk}_de_recovery": topk_de_recovery_metric(k=topk),
    }


def topk_de_recovery_metric(*, k: int = 50, direction: str = "absolute") -> MetricFn:
    """Build a grouped-report metric for top-k differential-expression recovery."""

    if k <= 0:
        raise ValueError("k must be positive")

    def metric(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
        return topk_de_recovery(predicted, observed, control, k=k, direction=direction)

    return metric


def topk_differential_expression_recovery_metric(
    *,
    k: int = 50,
    direction: str = "absolute",
) -> MetricFn:
    """Alias for :func:`topk_de_recovery_metric` with the expanded DE name."""

    return topk_de_recovery_metric(k=k, direction=direction)


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


def heldout_perturbation_prediction_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    perturbation_col: str = "perturbation",
    status_col: str = "perturbation_split",
    groupby: str | Sequence[str] | None = "perturbation",
    metrics: Mapping[str, MetricFn] | None = None,
    include_overall: bool = True,
    include_seen: bool = False,
    topk: int = 50,
) -> pd.DataFrame:
    """Report prediction quality for perturbations absent from training metadata."""

    predicted = _as_sample_feature_array(predicted, name="predicted")
    observed = _as_sample_feature_array(observed, name="observed")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")

    annotated = add_heldout_perturbation_status(
        metadata,
        train_metadata,
        perturbation_col=perturbation_col,
        status_col=status_col,
    )
    if include_seen:
        indices = np.arange(predicted.shape[0])
        report_groupby = _prepend_groupby(status_col, groupby)
    else:
        mask = annotated[status_col].eq("held_out").to_numpy()
        indices = np.flatnonzero(mask)
        report_groupby = groupby
    report = _grouped_prediction_subset_report(
        predicted,
        observed,
        control,
        annotated,
        indices,
        groupby=report_groupby,
        metrics=metrics,
        include_overall=include_overall,
        topk=topk,
    )
    if not include_seen and status_col not in report.columns:
        report.insert(1, status_col, "held_out")
    return report


def heldout_perturbation_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`heldout_perturbation_prediction_report`."""

    return heldout_perturbation_prediction_report(
        predicted,
        observed,
        control,
        metadata,
        train_metadata,
        **kwargs,
    )


def held_out_perturbation_prediction_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`heldout_perturbation_prediction_report`."""

    return heldout_perturbation_prediction_report(
        predicted,
        observed,
        control,
        metadata,
        train_metadata,
        **kwargs,
    )


def held_out_perturbation_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`heldout_perturbation_prediction_report`."""

    return heldout_perturbation_prediction_report(
        predicted,
        observed,
        control,
        metadata,
        train_metadata,
        **kwargs,
    )


def cell_line_transfer_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    cell_line_col: str = "cell_line",
    status_col: str = "cell_line_transfer",
    groupby: str | Sequence[str] | None = "cell_line",
    metrics: Mapping[str, MetricFn] | None = None,
    include_overall: bool = True,
    include_seen: bool = True,
    topk: int = 50,
) -> pd.DataFrame:
    """Report prediction quality by seen vs held-out cell-line transfer status."""

    predicted = _as_sample_feature_array(predicted, name="predicted")
    observed = _as_sample_feature_array(observed, name="observed")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")

    annotated = add_cell_line_transfer_status(
        metadata,
        train_metadata,
        cell_line_col=cell_line_col,
        status_col=status_col,
    )
    if include_seen:
        indices = np.arange(predicted.shape[0])
    else:
        indices = np.flatnonzero(annotated[status_col].eq("held_out").to_numpy())
    return _grouped_prediction_subset_report(
        predicted,
        observed,
        control,
        annotated,
        indices,
        groupby=_prepend_groupby(status_col, groupby),
        metrics=metrics,
        include_overall=include_overall,
        topk=topk,
    )


def cell_line_transfer_grouped_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`cell_line_transfer_report`."""

    return cell_line_transfer_report(
        predicted,
        observed,
        control,
        metadata,
        train_metadata,
        **kwargs,
    )


def dose_response_monotonicity_report(
    predicted: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    dose_col: str = "dose",
    groupby: str | Sequence[str] | None = ("perturbation", "cell_line"),
    response: str = "l2",
    direction: str = "increasing",
    min_points: int = 2,
    metric_name: str = "dose_response_monotonicity",
) -> pd.DataFrame:
    """Compute dose-response monotonicity overall and within metadata groups."""

    predicted = _as_sample_feature_array(predicted, name="predicted")
    frame = _as_metadata_frame(metadata, n_samples=predicted.shape[0])
    if dose_col not in frame.columns:
        raise ValueError(f"metadata is missing dose column: {dose_col}")

    records = [
        {
            "group": "overall",
            "n_samples": int(predicted.shape[0]),
            metric_name: dose_response_monotonicity(
                predicted,
                frame[dose_col].tolist(),
                control=control,
                response=response,
                direction=direction,
                min_points=min_points,
            ),
        }
    ]
    groupby_columns = _as_groupby_tuple(groupby)
    if groupby_columns is not None:
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
            index_array = np.asarray(indices, dtype=int)
            records.append(
                {
                    "group": "|".join(group_values.values()),
                    **group_values,
                    "n_samples": int(index_array.size),
                    metric_name: dose_response_monotonicity(
                        predicted[index_array],
                        frame.iloc[index_array][dose_col].tolist(),
                        control=_slice_control(
                            control,
                            index_array,
                            n_samples=predicted.shape[0],
                            n_features=predicted.shape[1],
                        ),
                        response=response,
                        direction=direction,
                        min_points=min_points,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def add_heldout_perturbation_status(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    perturbation_col: str = "perturbation",
    status_col: str = "perturbation_split",
) -> pd.DataFrame:
    """Annotate eval metadata with seen vs held-out perturbation status."""

    frame = _as_metadata_frame(metadata)
    train_frame = _as_metadata_frame(train_metadata)
    if perturbation_col not in frame.columns:
        raise ValueError(f"metadata is missing perturbation column: {perturbation_col}")
    if perturbation_col not in train_frame.columns:
        raise ValueError(f"train_metadata is missing perturbation column: {perturbation_col}")
    seen = {_string_key_value(value) for value in train_frame[perturbation_col].tolist()}
    frame[status_col] = [
        "seen" if _string_key_value(value) in seen else "held_out"
        for value in frame[perturbation_col].tolist()
    ]
    return frame


def add_held_out_perturbation_status(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    **kwargs: object,
) -> pd.DataFrame:
    """Alias for :func:`add_heldout_perturbation_status`."""

    return add_heldout_perturbation_status(metadata, train_metadata, **kwargs)


def add_cell_line_transfer_status(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    train_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    cell_line_col: str = "cell_line",
    status_col: str = "cell_line_transfer",
) -> pd.DataFrame:
    """Annotate eval metadata with seen vs held-out cell-line status."""

    frame = _as_metadata_frame(metadata)
    train_frame = _as_metadata_frame(train_metadata)
    if cell_line_col not in frame.columns:
        raise ValueError(f"metadata is missing cell-line column: {cell_line_col}")
    if cell_line_col not in train_frame.columns:
        raise ValueError(f"train_metadata is missing cell-line column: {cell_line_col}")
    seen = {_string_key_value(value) for value in train_frame[cell_line_col].tolist()}
    frame[status_col] = [
        "seen" if _string_key_value(value) in seen else "held_out"
        for value in frame[cell_line_col].tolist()
    ]
    return frame


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


def _grouped_prediction_subset_report(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    *,
    groupby: str | Sequence[str] | None,
    metrics: Mapping[str, MetricFn] | None,
    include_overall: bool,
    topk: int,
) -> pd.DataFrame:
    if indices.size == 0:
        columns = ["group"]
        columns.extend(_as_groupby_tuple(groupby) or ())
        columns.append("n_samples")
        columns.extend((default_delta_metrics(topk=topk) if metrics is None else metrics).keys())
        return pd.DataFrame(columns=columns)
    subset_metadata = metadata.iloc[indices].reset_index(drop=True)
    return grouped_metric_report(
        predicted[indices],
        observed[indices],
        _slice_control(control, indices, n_samples=predicted.shape[0], n_features=predicted.shape[1]),
        subset_metadata,
        groupby=groupby,
        metrics=metrics,
        include_overall=include_overall,
        topk=topk,
    )


def _prepend_groupby(
    column: str,
    groupby: str | Sequence[str] | None,
) -> tuple[str, ...]:
    columns = _as_groupby_tuple(groupby) or ()
    return (column, *(value for value in columns if value != column))


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
    n_samples: int | None = None,
) -> pd.DataFrame:
    if metadata is None:
        if n_samples is None:
            raise ValueError("metadata is required")
        return pd.DataFrame(index=np.arange(n_samples))
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.reset_index(drop=True).copy()
    else:
        frame = pd.DataFrame(metadata).reset_index(drop=True)
    if n_samples is not None and len(frame) != n_samples:
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
