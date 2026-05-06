from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.metrics import (
    centroid_by_condition,
    mean_average_precision,
    recall_at_k,
)

DEFAULT_CONTROL_VALUES = (
    "control",
    "ctrl",
    "dmso",
    "vehicle",
    "untreated",
    "mock",
    "negative_control",
    "negcon",
)


def infer_control_mask(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    control_col: str = "perturbation",
    perturbation_type_col: str = "perturbation_type",
    is_control_col: str = "is_control",
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES,
) -> np.ndarray:
    """Infer control rows from common perturbation metadata conventions."""

    frame = _as_metadata_frame(metadata)
    if is_control_col in frame.columns:
        return frame[is_control_col].astype(bool).to_numpy()

    normalized_controls = {_normalize_token(value) for value in control_values}
    masks: list[np.ndarray] = []
    if control_col in frame.columns:
        masks.append(frame[control_col].map(_normalize_token).isin(normalized_controls).to_numpy())
    if perturbation_type_col in frame.columns:
        masks.append(
            frame[perturbation_type_col].map(_normalize_token).isin(normalized_controls).to_numpy()
        )
    if not masks:
        raise ValueError(
            "metadata must contain an is_control column or perturbation/control type columns"
        )
    return np.logical_or.reduce(masks)


@dataclass
class ControlMeanBaseline:
    """Predict the mean observed profile among control rows for every query."""

    control_col: str = "perturbation"
    perturbation_type_col: str = "perturbation_type"
    is_control_col: str = "is_control"
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES
    mean_: np.ndarray | None = field(init=False, default=None)
    n_features_in_: int | None = field(init=False, default=None)

    def fit(
        self,
        values: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
        *,
        control_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> "ControlMeanBaseline":
        values = _as_2d_float_array(values, name="values")
        frame = _as_metadata_frame(metadata, n_samples=values.shape[0])
        mask = _resolve_control_mask(
            frame,
            control_mask=control_mask,
            control_col=self.control_col,
            perturbation_type_col=self.perturbation_type_col,
            is_control_col=self.is_control_col,
            control_values=self.control_values,
        )
        if not mask.any():
            raise ValueError("cannot fit control mean baseline without control rows")
        self.mean_ = values[mask].mean(axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, metadata_or_n: pd.DataFrame | Mapping[str, Sequence[object]] | int) -> np.ndarray:
        self._check_fitted()
        n_samples = _prediction_count(metadata_or_n)
        return np.repeat(self.mean_[None, :], n_samples, axis=0)

    def fit_predict(
        self,
        values: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
        *,
        predict_metadata: pd.DataFrame | Mapping[str, Sequence[object]] | int | None = None,
        control_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> np.ndarray:
        self.fit(values, metadata, control_mask=control_mask)
        return self.predict(values.shape[0] if predict_metadata is None else predict_metadata)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("ControlMeanBaseline must be fit before predict")


@dataclass
class PerturbationMeanBaseline:
    """Predict a training-set mean profile for each perturbation metadata group."""

    groupby: str | Sequence[str] = "perturbation"
    fallback: str = "global"
    control_col: str = "perturbation"
    perturbation_type_col: str = "perturbation_type"
    is_control_col: str = "is_control"
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES
    group_means_: dict[tuple[str, ...], np.ndarray] = field(init=False, default_factory=dict)
    global_mean_: np.ndarray | None = field(init=False, default=None)
    control_mean_: np.ndarray | None = field(init=False, default=None)
    groupby_: tuple[str, ...] | None = field(init=False, default=None)
    n_features_in_: int | None = field(init=False, default=None)

    def fit(
        self,
        values: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]],
        *,
        control_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> "PerturbationMeanBaseline":
        if self.fallback not in {"global", "control"}:
            raise ValueError("fallback must be 'global' or 'control'")

        values = _as_2d_float_array(values, name="values")
        frame = _as_metadata_frame(metadata, n_samples=values.shape[0])
        groupby = _as_groupby_tuple(self.groupby)
        keys = _metadata_group_keys(frame, groupby)

        self.group_means_ = {}
        for key in sorted(set(keys)):
            rows = np.asarray([row_key == key for row_key in keys], dtype=bool)
            self.group_means_[key] = values[rows].mean(axis=0)

        self.global_mean_ = values.mean(axis=0)
        self.control_mean_ = None
        try:
            mask = _resolve_control_mask(
                frame,
                control_mask=control_mask,
                control_col=self.control_col,
                perturbation_type_col=self.perturbation_type_col,
                is_control_col=self.is_control_col,
                control_values=self.control_values,
            )
        except ValueError:
            mask = np.zeros(values.shape[0], dtype=bool)
        if mask.any():
            self.control_mean_ = values[mask].mean(axis=0)
        elif self.fallback == "control":
            raise ValueError("fallback='control' requires at least one control row")

        self.groupby_ = groupby
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, metadata: pd.DataFrame | Mapping[str, Sequence[object]]) -> np.ndarray:
        self._check_fitted()
        frame = _as_metadata_frame(metadata)
        keys = _metadata_group_keys(frame, self.groupby_)
        fallback = self._fallback_mean()
        return np.vstack([self.group_means_.get(key, fallback) for key in keys])

    def fit_predict(
        self,
        values: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]],
        *,
        predict_metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
        control_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> np.ndarray:
        self.fit(values, metadata, control_mask=control_mask)
        return self.predict(metadata if predict_metadata is None else predict_metadata)

    def _fallback_mean(self) -> np.ndarray:
        if self.fallback == "control":
            if self.control_mean_ is None:
                raise RuntimeError("control fallback was requested but no control mean is available")
            return self.control_mean_
        if self.global_mean_ is None:
            raise RuntimeError("global fallback is unavailable before fit")
        return self.global_mean_

    def _check_fitted(self) -> None:
        if self.global_mean_ is None or self.groupby_ is None:
            raise RuntimeError("PerturbationMeanBaseline must be fit before predict")


@dataclass
class CentroidRetrievalBaseline:
    """Retrieve condition labels by nearest gallery centroid under dot-product scoring."""

    label_col: str = "condition_key"
    normalize: bool = True
    centroids_: np.ndarray | None = field(init=False, default=None)
    labels_: list[str] | None = field(init=False, default=None)

    def fit(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
        *,
        labels: Sequence[str] | None = None,
    ) -> "CentroidRetrievalBaseline":
        embeddings = _as_2d_float_array(embeddings, name="embeddings")
        labels = _labels_from_metadata(labels, metadata, self.label_col, embeddings.shape[0])
        centroids, centroid_labels = centroid_by_condition(embeddings, labels)
        if self.normalize:
            centroids = _l2_normalize_rows(centroids)
        self.centroids_ = centroids
        self.labels_ = centroid_labels
        return self

    def score(self, query_embeddings: np.ndarray) -> np.ndarray:
        self._check_fitted()
        query = _as_2d_float_array(query_embeddings, name="query_embeddings")
        if self.normalize:
            query = _l2_normalize_rows(query)
        return query @ self.centroids_.T

    def predict(self, query_embeddings: np.ndarray) -> list[str]:
        scores = self.score(query_embeddings)
        return [self.labels_[idx] for idx in np.argmax(scores, axis=1)]

    def evaluate(
        self,
        query_embeddings: np.ndarray,
        metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None = None,
        *,
        labels: Sequence[str] | None = None,
        ks: Sequence[int] = (1, 5),
        prefix: str = "centroid",
    ) -> dict[str, float]:
        self._check_fitted()
        query = _as_2d_float_array(query_embeddings, name="query_embeddings")
        query_labels = _labels_from_metadata(labels, metadata, self.label_col, query.shape[0])
        if self.normalize:
            query = _l2_normalize_rows(query)
        return _retrieval_metric_dict(query, self.centroids_, query_labels, self.labels_, ks, prefix=prefix)

    def _check_fitted(self) -> None:
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("CentroidRetrievalBaseline must be fit before use")


def centroid_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5),
    normalize: bool = True,
    prefix: str = "centroid",
) -> dict[str, float]:
    """Evaluate query-to-gallery retrieval after collapsing gallery rows to label centroids."""

    query = _as_2d_float_array(query_embeddings, name="query_embeddings")
    gallery = _as_2d_float_array(gallery_embeddings, name="gallery_embeddings")
    query_labels = _coerce_labels(query_labels, query.shape[0], name="query_labels")
    gallery_labels = _coerce_labels(gallery_labels, gallery.shape[0], name="gallery_labels")
    centroids, centroid_labels = centroid_by_condition(gallery, gallery_labels)
    if normalize:
        query = _l2_normalize_rows(query)
        centroids = _l2_normalize_rows(centroids)
    return _retrieval_metric_dict(query, centroids, query_labels, centroid_labels, ks, prefix=prefix)


def shuffle_labels(
    labels: Sequence[str],
    *,
    random_state: int | np.random.Generator | None = 0,
    max_attempts: int = 256,
) -> list[str]:
    """Return a deterministic label shuffle that minimizes unchanged sample labels."""

    labels_array = np.asarray(list(labels), dtype=object)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels_array.size < 2 or np.unique(labels_array).size < 2:
        return [str(label) for label in labels_array]

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    best = labels_array.copy()
    best_matches = labels_array.size + 1
    for _ in range(max_attempts):
        candidate = labels_array[rng.permutation(labels_array.size)]
        matches = int(np.sum(candidate == labels_array))
        if matches < best_matches:
            best = candidate.copy()
            best_matches = matches
        if matches == 0:
            break
    return [str(label) for label in best]


def label_shuffle_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5),
    random_state: int | np.random.Generator | None = 0,
    normalize: bool = True,
    prefix: str = "label_shuffle",
) -> dict[str, float]:
    """Evaluate retrieval after shuffling query labels as a negative control."""

    query = _as_2d_float_array(query_embeddings, name="query_embeddings")
    gallery = _as_2d_float_array(gallery_embeddings, name="gallery_embeddings")
    query_labels = _coerce_labels(query_labels, query.shape[0], name="query_labels")
    gallery_labels = _coerce_labels(gallery_labels, gallery.shape[0], name="gallery_labels")
    shuffled_query_labels = shuffle_labels(query_labels, random_state=random_state)
    if normalize:
        query = _l2_normalize_rows(query)
        gallery = _l2_normalize_rows(gallery)
    return _retrieval_metric_dict(query, gallery, shuffled_query_labels, gallery_labels, ks, prefix=prefix)


def label_shuffle_centroid_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5),
    random_state: int | np.random.Generator | None = 0,
    normalize: bool = True,
    prefix: str = "centroid",
) -> dict[str, float]:
    """Evaluate centroid retrieval after shuffling query labels as a negative control."""

    query_labels = _coerce_labels(query_labels, len(query_labels), name="query_labels")
    shuffled_query_labels = shuffle_labels(query_labels, random_state=random_state)
    return centroid_retrieval_metrics(
        query_embeddings,
        gallery_embeddings,
        shuffled_query_labels,
        gallery_labels,
        ks=ks,
        normalize=normalize,
        prefix=prefix,
    )


def _retrieval_metric_dict(
    query: np.ndarray,
    gallery: np.ndarray,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    ks: Sequence[int],
    *,
    prefix: str,
) -> dict[str, float]:
    metrics = {f"{prefix}_map": mean_average_precision(query, gallery, list(query_labels), list(gallery_labels))}
    for k in ks:
        if k <= 0:
            raise ValueError("retrieval k values must be positive")
        metrics[f"{prefix}_recall@{k}"] = recall_at_k(
            query,
            gallery,
            list(query_labels),
            list(gallery_labels),
            k=k,
        )
    return metrics


def _resolve_control_mask(
    metadata: pd.DataFrame,
    *,
    control_mask: Sequence[bool] | np.ndarray | None,
    control_col: str,
    perturbation_type_col: str,
    is_control_col: str,
    control_values: Sequence[str],
) -> np.ndarray:
    if control_mask is None:
        mask = infer_control_mask(
            metadata,
            control_col=control_col,
            perturbation_type_col=perturbation_type_col,
            is_control_col=is_control_col,
            control_values=control_values,
        )
    else:
        mask = np.asarray(control_mask, dtype=bool)
    if mask.shape != (len(metadata),):
        raise ValueError("control_mask length must match metadata rows")
    return mask


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
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None,
    *,
    n_samples: int | None = None,
) -> pd.DataFrame:
    if metadata is None:
        if n_samples is None:
            raise ValueError("metadata is required when n_samples is not provided")
        frame = pd.DataFrame(index=np.arange(n_samples))
    elif isinstance(metadata, pd.DataFrame):
        frame = metadata.reset_index(drop=True).copy()
    else:
        frame = pd.DataFrame(metadata).reset_index(drop=True)
    if n_samples is not None and len(frame) != n_samples:
        raise ValueError("metadata row count must match array rows")
    return frame


def _prediction_count(metadata_or_n: pd.DataFrame | Mapping[str, Sequence[object]] | int) -> int:
    if isinstance(metadata_or_n, int):
        if metadata_or_n < 0:
            raise ValueError("prediction row count must be non-negative")
        return metadata_or_n
    return len(_as_metadata_frame(metadata_or_n))


def _as_groupby_tuple(groupby: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(groupby, str):
        columns = (groupby,)
    else:
        columns = tuple(groupby)
    if not columns:
        raise ValueError("groupby must contain at least one column")
    return columns


def _metadata_group_keys(frame: pd.DataFrame, groupby: tuple[str, ...]) -> list[tuple[str, ...]]:
    missing = [column for column in groupby if column not in frame.columns]
    if missing:
        raise ValueError(f"metadata is missing groupby columns: {missing}")
    return [
        tuple(_string_key_value(value) for value in row)
        for row in frame.loc[:, list(groupby)].itertuples(index=False, name=None)
    ]


def _labels_from_metadata(
    labels: Sequence[str] | None,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]] | None,
    label_col: str,
    n_samples: int,
) -> list[str]:
    if labels is not None:
        return _coerce_labels(labels, n_samples, name="labels")
    frame = _as_metadata_frame(metadata, n_samples=n_samples)
    if label_col not in frame.columns:
        raise ValueError(f"metadata is missing label column: {label_col}")
    return _coerce_labels(frame[label_col].tolist(), n_samples, name=label_col)


def _coerce_labels(labels: Sequence[str], n_samples: int, *, name: str) -> list[str]:
    labels = [str(label) for label in labels]
    if len(labels) != n_samples:
        raise ValueError(f"{name} length must match array rows")
    return labels


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _normalize_token(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _string_key_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
