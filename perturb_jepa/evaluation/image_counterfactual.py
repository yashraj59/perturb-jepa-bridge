from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.metrics import dose_response_monotonicity
from perturb_jepa.evaluation.retrieval import label_enrichment_at_k, ranks_from_scores

DEFAULT_CONDITION_COLUMNS = ("perturbation", "dose", "time", "cell_line")


def image_counterfactual_metrics(
    predicted_embeddings: np.ndarray,
    observed_embeddings: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "condition_key",
    condition_cols: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
    normalize: bool = True,
    enrichment_k: int = 10,
    prefix: str = "image_counterfactual",
) -> dict[str, float]:
    """Evaluate counterfactual image embedding predictions."""

    predicted = _as_2d_float_array(predicted_embeddings, name="predicted_embeddings")
    observed = _as_2d_float_array(observed_embeddings, name="observed_embeddings")
    if predicted.shape != observed.shape:
        raise ValueError("predicted_embeddings and observed_embeddings must have the same shape")
    frame = _as_metadata_frame(metadata, n_samples=predicted.shape[0])
    labels = _labels_from_metadata(frame, label_col=label_col, condition_cols=condition_cols)
    if normalize:
        predicted = _l2_normalize_rows(predicted)
        observed = _l2_normalize_rows(observed)

    observed_bags = _bag_centroids(observed, labels)
    bag_labels = list(observed_bags)
    gallery = np.vstack([observed_bags[label] for label in bag_labels])
    scores = predicted @ gallery.T
    ranks = ranks_from_scores(scores, labels, bag_labels)
    matching_centroids = np.vstack([observed_bags[label] for label in labels])
    distances = np.linalg.norm(predicted - matching_centroids, axis=1)

    metrics = {
        f"{prefix}_distance_to_observed_bag": float(distances.mean()),
        f"{prefix}_median_true_condition_rank": float(np.median(ranks)),
        f"{prefix}_recall@1_true_condition": float(np.mean(ranks <= 1)),
        f"{prefix}_replicate_correlation": replicate_correlation(predicted, observed, labels),
    }
    if "dose" in frame.columns:
        groups = _ordering_groups(frame)
        metrics[f"{prefix}_dose_ordering_accuracy"] = dose_response_monotonicity(
            predicted,
            frame["dose"].tolist(),
            groups=groups,
            control=matching_centroids,
            direction="auto",
        )
    if "time" in frame.columns:
        groups = _ordering_groups(frame, exclude=("time",))
        metrics[f"{prefix}_time_ordering_accuracy"] = dose_response_monotonicity(
            predicted,
            frame["time"].tolist(),
            groups=groups,
            control=matching_centroids,
            direction="auto",
        )
    if "moa" in frame.columns:
        metrics[f"{prefix}_same_moa_enrichment@{enrichment_k}"] = label_enrichment_at_k(
            scores,
            frame["moa"].tolist(),
            _first_metadata_value_by_label(frame, labels, "moa", bag_labels),
            k=enrichment_k,
        )
    return metrics


def replicate_correlation(
    predicted_embeddings: np.ndarray,
    observed_embeddings: np.ndarray,
    labels: Sequence[object],
) -> float:
    predicted = _as_2d_float_array(predicted_embeddings, name="predicted_embeddings")
    observed = _as_2d_float_array(observed_embeddings, name="observed_embeddings")
    labels = [_string_value(label) for label in labels]
    values: list[float] = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(np.asarray(labels, dtype=object) == label)
        if idx.size < 2:
            continue
        pred_d = _pairwise_distances(predicted[idx])
        obs_d = _pairwise_distances(observed[idx])
        iu = np.triu_indices(idx.size, k=1)
        values.append(_safe_corr(pred_d[iu], obs_d[iu]))
    return float(np.mean(values)) if values else 0.0


def _bag_centroids(embeddings: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    frame = pd.DataFrame(embeddings)
    frame["label"] = list(labels)
    grouped = frame.groupby("label", sort=True).mean()
    return {str(label): row.to_numpy(dtype=float) for label, row in grouped.iterrows()}


def _ordering_groups(frame: pd.DataFrame, *, exclude: Sequence[str] = ("dose",)) -> list[str]:
    columns = [
        column
        for column in ("perturbation", "cell_line", "time", "dose")
        if column in frame.columns and column not in set(exclude)
    ]
    if not columns:
        return ["overall"] * len(frame)
    return [
        "|".join(_string_value(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def _first_metadata_value_by_label(
    frame: pd.DataFrame,
    labels: Sequence[str],
    column: str,
    bag_labels: Sequence[str],
) -> list[object]:
    lookup: dict[str, object] = {}
    for label, value in zip(labels, frame[column].tolist(), strict=True):
        lookup.setdefault(label, value)
    return [lookup.get(label, "NA") for label in bag_labels]


def _labels_from_metadata(
    frame: pd.DataFrame,
    *,
    label_col: str,
    condition_cols: Sequence[str],
) -> list[str]:
    if label_col in frame.columns:
        return [_string_value(value) for value in frame[label_col].tolist()]
    columns = [column for column in condition_cols if column in frame.columns]
    if not columns:
        raise ValueError(f"metadata must contain {label_col!r} or at least one condition column")
    return [
        "|".join(_string_value(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("correlation inputs must have the same size")
    if x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _as_2d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    return array


def _as_metadata_frame(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    n_samples: int,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if len(frame) != n_samples:
        raise ValueError("metadata row count must match embedding rows")
    return frame.reset_index(drop=True)


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
