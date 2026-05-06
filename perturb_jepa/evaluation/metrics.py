from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pandas as pd


_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def pearson_delta(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
    pred_delta = np.asarray(predicted) - np.asarray(control)
    obs_delta = np.asarray(observed) - np.asarray(control)
    return _safe_corr(pred_delta.ravel(), obs_delta.ravel())


def spearman_delta(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray) -> float:
    pred_delta = pd.Series((np.asarray(predicted) - np.asarray(control)).ravel()).rank().to_numpy()
    obs_delta = pd.Series((np.asarray(observed) - np.asarray(control)).ravel()).rank().to_numpy()
    return _safe_corr(pred_delta, obs_delta)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("correlation inputs must have the same size")
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def topk_jaccard(predicted_scores: np.ndarray, observed_scores: np.ndarray, *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    pred_top = set(np.argsort(np.asarray(predicted_scores))[-k:])
    obs_top = set(np.argsort(np.asarray(observed_scores))[-k:])
    if not pred_top and not obs_top:
        return 1.0
    return len(pred_top & obs_top) / len(pred_top | obs_top)


def topk_de_recovery(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    *,
    k: int,
    direction: str = "absolute",
) -> float:
    """Recover observed top-k differential-expression features from predictions.

    The score is computed per sample and averaged. ``direction="absolute"``
    ranks genes by absolute delta from control, while ``"up"`` and ``"down"``
    rank positive and negative deltas respectively.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    predicted = _as_sample_feature_array(predicted, name="predicted")
    observed = _as_sample_feature_array(observed, name="observed")
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed must have the same shape")
    if predicted.shape[1] == 0:
        raise ValueError("predicted and observed must contain at least one feature")
    control = _control_like(control, n_samples=predicted.shape[0], n_features=predicted.shape[1])

    pred_delta = predicted - control
    obs_delta = observed - control
    actual_k = min(k, predicted.shape[1])
    recoveries = []
    for pred_row, obs_row in zip(pred_delta, obs_delta, strict=True):
        pred_scores = _de_scores(pred_row, direction=direction)
        obs_scores = _de_scores(obs_row, direction=direction)
        pred_top = set(_topk_indices(pred_scores, actual_k))
        obs_top = set(_topk_indices(obs_scores, actual_k))
        recoveries.append(len(pred_top & obs_top) / len(obs_top))
    return float(np.mean(recoveries)) if recoveries else 0.0


def topk_differential_expression_recovery(
    predicted: np.ndarray,
    observed: np.ndarray,
    control: np.ndarray,
    *,
    k: int,
    direction: str = "absolute",
) -> float:
    """Alias for :func:`topk_de_recovery` with the expanded DE name."""

    return topk_de_recovery(predicted, observed, control, k=k, direction=direction)


def dose_response_monotonicity(
    responses: np.ndarray,
    doses: Sequence[object] | np.ndarray,
    *,
    groups: Sequence[object] | np.ndarray | None = None,
    control: np.ndarray | None = None,
    response: str = "l2",
    direction: str = "increasing",
    min_points: int = 2,
    tolerance: float = 1e-12,
) -> float:
    """Measure whether response magnitude changes monotonically with dose.

    ``responses`` may be a vector of scalar responses or a sample-by-feature
    matrix. Matrix rows are reduced to scalar response magnitudes before scoring.
    Duplicate doses within a group are averaged. The returned value is the mean
    pairwise monotonicity score over valid groups, where 1.0 is fully monotonic.
    """

    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    if direction not in {"increasing", "decreasing", "auto"}:
        raise ValueError("direction must be 'increasing', 'decreasing', or 'auto'")
    values = _response_magnitude(responses, control=control, response=response)
    dose_values = _numeric_vector(doses, name="doses")
    if values.shape[0] != dose_values.shape[0]:
        raise ValueError("responses and doses must contain the same number of rows")
    if groups is None:
        group_values = np.zeros(values.shape[0], dtype=object)
    else:
        group_values = np.asarray(list(groups), dtype=object)
        if group_values.shape != (values.shape[0],):
            raise ValueError("groups length must match responses")

    frame = pd.DataFrame({"group": group_values, "dose": dose_values, "response": values})
    scores: list[float] = []
    for _, group_frame in frame.groupby("group", sort=True, dropna=False):
        by_dose = group_frame.groupby("dose", sort=True, dropna=False)["response"].mean()
        if len(by_dose) < min_points:
            continue
        scores.append(
            _pairwise_monotonicity(
                by_dose.index.to_numpy(dtype=float),
                by_dose.to_numpy(dtype=float),
                direction=direction,
                tolerance=tolerance,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def centroid_by_condition(embeddings: np.ndarray, condition_keys: list[str] | np.ndarray) -> tuple[np.ndarray, list[str]]:
    frame = pd.DataFrame(np.asarray(embeddings))
    frame["condition_key"] = list(condition_keys)
    grouped = frame.groupby("condition_key", sort=True).mean()
    return grouped.drop(columns=[], errors="ignore").to_numpy(dtype=float), grouped.index.astype(str).tolist()


def mean_average_precision(query: np.ndarray, gallery: np.ndarray, query_labels: list[str], gallery_labels: list[str]) -> float:
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery feature dimensions must match")
    scores = query @ gallery.T
    average_precisions: list[float] = []
    for row, label in zip(scores, query_labels, strict=True):
        order = np.argsort(-row)
        hits = np.asarray([gallery_labels[idx] == label for idx in order], dtype=bool)
        positives = hits.sum()
        if positives == 0:
            continue
        precision_at_hits = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)
        average_precisions.append(float(precision_at_hits.mean()))
    return float(np.mean(average_precisions)) if average_precisions else 0.0


def recall_at_k(query: np.ndarray, gallery: np.ndarray, query_labels: list[str], gallery_labels: list[str], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    scores = np.asarray(query, dtype=float) @ np.asarray(gallery, dtype=float).T
    hits = []
    for row, label in zip(scores, query_labels, strict=True):
        order = np.argsort(-row)[:k]
        hits.append(any(gallery_labels[idx] == label for idx in order))
    return float(np.mean(hits)) if hits else 0.0


def distance_matrix_spearman(a: np.ndarray, b: np.ndarray) -> float:
    da = _pairwise_distances(np.asarray(a, dtype=float))
    db = _pairwise_distances(np.asarray(b, dtype=float))
    iu = np.triu_indices_from(da, k=1)
    return _safe_corr(pd.Series(da[iu]).rank().to_numpy(), pd.Series(db[iu]).rank().to_numpy())


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


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
    if array.ndim == 1 and array.shape[0] == n_samples and n_features == 1:
        return array.reshape(-1, 1)
    try:
        return np.broadcast_to(array, (n_samples, n_features)).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError("control must be broadcastable to predicted and observed") from exc


def _de_scores(delta: np.ndarray, *, direction: str) -> np.ndarray:
    if direction == "absolute":
        return np.abs(delta)
    if direction == "up":
        return delta
    if direction == "down":
        return -delta
    raise ValueError("direction must be 'absolute', 'up', or 'down'")


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    return order[-k:]


def _response_magnitude(
    responses: np.ndarray,
    *,
    control: np.ndarray | None,
    response: str,
) -> np.ndarray:
    if response not in {"l2", "mean", "absolute_mean"}:
        raise ValueError("response must be 'l2', 'mean', or 'absolute_mean'")
    values = np.asarray(responses, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim == 1:
        if control is not None:
            values = values - np.broadcast_to(np.asarray(control, dtype=float), values.shape)
        return values.astype(float, copy=False)

    matrix = values.reshape(values.shape[0], -1)
    if control is not None:
        matrix = matrix - _control_like(control, n_samples=matrix.shape[0], n_features=matrix.shape[1])
    if response == "l2":
        return np.linalg.norm(matrix, axis=1)
    if response == "mean":
        return matrix.mean(axis=1)
    if response == "absolute_mean":
        return np.abs(matrix).mean(axis=1)
    raise AssertionError("unreachable response mode")


def _numeric_vector(values: Sequence[object] | np.ndarray, *, name: str) -> np.ndarray:
    numeric = np.asarray([_parse_numeric(value) for value in list(values)], dtype=float)
    if numeric.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return numeric


def _parse_numeric(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    match = _NUMERIC_RE.search(text)
    return float(match.group(0)) if match is not None else np.nan


def _pairwise_monotonicity(
    doses: np.ndarray,
    responses: np.ndarray,
    *,
    direction: str,
    tolerance: float,
) -> float:
    if direction not in {"increasing", "decreasing", "auto"}:
        raise ValueError("direction must be 'increasing', 'decreasing', or 'auto'")
    valid = np.isfinite(doses) & np.isfinite(responses)
    doses = doses[valid]
    responses = responses[valid]
    pair_count = 0
    concordant = 0
    for i in range(doses.size):
        for j in range(i + 1, doses.size):
            if doses[i] == doses[j]:
                continue
            pair_count += 1
            lower, higher = (i, j) if doses[i] < doses[j] else (j, i)
            delta = responses[higher] - responses[lower]
            if direction == "increasing":
                concordant += delta >= -tolerance
            elif direction == "decreasing":
                concordant += delta <= tolerance
            elif direction == "auto":
                concordant += abs(delta) <= tolerance
    if pair_count == 0:
        return 0.0
    if direction == "auto":
        increasing = _pairwise_monotonicity(
            doses,
            responses,
            direction="increasing",
            tolerance=tolerance,
        )
        decreasing = _pairwise_monotonicity(
            doses,
            responses,
            direction="decreasing",
            tolerance=tolerance,
        )
        return max(increasing, decreasing)
    return float(concordant / pair_count)
