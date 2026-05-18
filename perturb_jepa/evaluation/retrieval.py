from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_RETRIEVAL_KS = (1, 5, 10)
DEFAULT_STRATA = ("batch", "perturbation", "dose", "time", "cell_line")
DEFAULT_CONDITION_COLUMNS = ("perturbation", "dose", "time", "cell_line")


def cross_modal_retrieval_metrics(
    rna_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    rna_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    image_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "condition_key",
    condition_cols: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    normalize: bool = True,
    stratify_by: Sequence[str] = DEFAULT_STRATA,
    enrichment_k: int = 10,
) -> dict[str, float]:
    """Evaluate RNA->image and image->RNA retrieval using embeddings for scoring.

    Metadata is used only to define relevance labels, enrichment labels, and
    query strata. Similarity scores are computed from the supplied embeddings.
    """

    rna = _as_2d_float_array(rna_embeddings, name="rna_embeddings")
    image = _as_2d_float_array(image_embeddings, name="image_embeddings")
    if rna.shape[1] != image.shape[1]:
        raise ValueError("rna_embeddings and image_embeddings must have matching dimensions")
    rna_frame = _as_metadata_frame(rna_metadata, n_samples=rna.shape[0])
    image_frame = _as_metadata_frame(image_metadata, n_samples=image.shape[0])
    if normalize:
        rna = _l2_normalize_rows(rna)
        image = _l2_normalize_rows(image)

    metrics: dict[str, float] = {}
    metrics.update(
        directional_retrieval_metrics(
            rna,
            image,
            rna_frame,
            image_frame,
            label_col=label_col,
            condition_cols=condition_cols,
            ks=ks,
            prefix="rna_to_image",
            enrichment_k=enrichment_k,
            stratify_by=stratify_by,
        )
    )
    metrics.update(
        directional_retrieval_metrics(
            image,
            rna,
            image_frame,
            rna_frame,
            label_col=label_col,
            condition_cols=condition_cols,
            ks=ks,
            prefix="image_to_rna",
            enrichment_k=enrichment_k,
            stratify_by=stratify_by,
        )
    )
    return metrics


def directional_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    gallery_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "condition_key",
    condition_cols: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    prefix: str = "retrieval",
    enrichment_k: int = 10,
    stratify_by: Sequence[str] = DEFAULT_STRATA,
) -> dict[str, float]:
    query = _as_2d_float_array(query_embeddings, name="query_embeddings")
    gallery = _as_2d_float_array(gallery_embeddings, name="gallery_embeddings")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query_embeddings and gallery_embeddings must have matching dimensions")
    query_frame = _as_metadata_frame(query_metadata, n_samples=query.shape[0])
    gallery_frame = _as_metadata_frame(gallery_metadata, n_samples=gallery.shape[0])
    query_labels = _labels_from_metadata(query_frame, label_col=label_col, condition_cols=condition_cols)
    gallery_labels = _labels_from_metadata(gallery_frame, label_col=label_col, condition_cols=condition_cols)
    scores = query @ gallery.T

    metrics = _rank_metrics(scores, query_labels, gallery_labels, ks=ks, prefix=prefix)
    metrics[f"{prefix}_same_perturbation_enrichment@{enrichment_k}"] = label_enrichment_at_k(
        scores,
        _optional_column(query_frame, "perturbation"),
        _optional_column(gallery_frame, "perturbation"),
        k=enrichment_k,
    )
    if "moa" in query_frame.columns and "moa" in gallery_frame.columns:
        metrics[f"{prefix}_same_moa_enrichment@{enrichment_k}"] = label_enrichment_at_k(
            scores,
            query_frame["moa"].tolist(),
            gallery_frame["moa"].tolist(),
            k=enrichment_k,
        )

    for column in stratify_by:
        if column not in query_frame.columns:
            continue
        for value, indices in _strata_indices(query_frame[column]).items():
            if len(indices) == 0:
                continue
            stratum_prefix = f"{prefix}_by_{column}={_metric_key(value)}"
            metrics.update(
                _rank_metrics(
                    scores[indices],
                    [query_labels[idx] for idx in indices],
                    gallery_labels,
                    ks=ks,
                    prefix=stratum_prefix,
                )
            )
    return metrics


def retrieval_ranks(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: Sequence[object],
    gallery_labels: Sequence[object],
) -> np.ndarray:
    query = _as_2d_float_array(query_embeddings, name="query_embeddings")
    gallery = _as_2d_float_array(gallery_embeddings, name="gallery_embeddings")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query_embeddings and gallery_embeddings must have matching dimensions")
    return ranks_from_scores(query @ gallery.T, query_labels, gallery_labels)


def ranks_from_scores(
    scores: np.ndarray,
    query_labels: Sequence[object],
    gallery_labels: Sequence[object],
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    query_labels = _coerce_label_list(query_labels, scores.shape[0], name="query_labels")
    gallery_labels = _coerce_label_list(gallery_labels, scores.shape[1], name="gallery_labels")
    ranks: list[float] = []
    for row, label in zip(scores, query_labels, strict=True):
        order = np.argsort(-row, kind="mergesort")
        hit_positions = np.flatnonzero(np.asarray([gallery_labels[idx] == label for idx in order], dtype=bool))
        ranks.append(float(hit_positions[0] + 1) if hit_positions.size else float(scores.shape[1] + 1))
    return np.asarray(ranks, dtype=float)


def label_enrichment_at_k(
    scores: np.ndarray,
    query_values: Sequence[object] | None,
    gallery_values: Sequence[object] | None,
    *,
    k: int,
) -> float:
    """Return observed top-k same-label rate divided by gallery marginal expectation."""

    if query_values is None or gallery_values is None:
        return 0.0
    if k <= 0:
        raise ValueError("k must be positive")
    scores = np.asarray(scores, dtype=float)
    query = _coerce_label_list(query_values, scores.shape[0], name="query_values")
    gallery = _coerce_label_list(gallery_values, scores.shape[1], name="gallery_values")
    actual_k = min(k, scores.shape[1])
    enrichments: list[float] = []
    for row, label in zip(scores, query, strict=True):
        if label == "" or label.lower() == "nan":
            continue
        gallery_matches = np.asarray([value == label for value in gallery], dtype=bool)
        expected = float(gallery_matches.mean())
        if expected == 0.0:
            continue
        top = np.argsort(-row, kind="mergesort")[:actual_k]
        observed = float(gallery_matches[top].mean())
        enrichments.append(observed / expected)
    return float(np.mean(enrichments)) if enrichments else 0.0


def _rank_metrics(
    scores: np.ndarray,
    query_labels: Sequence[object],
    gallery_labels: Sequence[object],
    *,
    ks: Sequence[int],
    prefix: str,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    ranks = ranks_from_scores(scores, query_labels, gallery_labels)
    metrics = {
        f"{prefix}_map": average_precision_from_scores(scores, query_labels, gallery_labels),
        f"{prefix}_median_rank": float(np.median(ranks)) if ranks.size else 0.0,
    }
    for k in ks:
        if k <= 0:
            raise ValueError("retrieval k values must be positive")
        metrics[f"{prefix}_recall@{k}"] = float(np.mean(ranks <= k)) if ranks.size else 0.0
    return metrics


def average_precision_from_scores(
    scores: np.ndarray,
    query_labels: Sequence[object],
    gallery_labels: Sequence[object],
) -> float:
    scores = np.asarray(scores, dtype=float)
    query_labels = _coerce_label_list(query_labels, scores.shape[0], name="query_labels")
    gallery_labels = _coerce_label_list(gallery_labels, scores.shape[1], name="gallery_labels")
    aps: list[float] = []
    for row, label in zip(scores, query_labels, strict=True):
        order = np.argsort(-row, kind="mergesort")
        hits = np.asarray([gallery_labels[idx] == label for idx in order], dtype=bool)
        if not hits.any():
            continue
        precision_at_hits = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)
        aps.append(float(precision_at_hits.mean()))
    return float(np.mean(aps)) if aps else 0.0


def _labels_from_metadata(
    frame: pd.DataFrame,
    *,
    label_col: str,
    condition_cols: Sequence[str],
) -> list[str]:
    if label_col in frame.columns:
        return _coerce_label_list(frame[label_col].tolist(), len(frame), name=label_col)
    columns = [column for column in condition_cols if column in frame.columns]
    if not columns:
        raise ValueError(f"metadata must contain {label_col!r} or at least one condition column")
    return [
        "|".join(_string_value(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    ]


def _optional_column(frame: pd.DataFrame, column: str) -> list[object] | None:
    return frame[column].tolist() if column in frame.columns else None


def _strata_indices(values: pd.Series) -> dict[object, np.ndarray]:
    result: dict[object, np.ndarray] = {}
    for value in values.drop_duplicates().tolist():
        result[value] = np.flatnonzero((values == value).to_numpy())
    return result


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


def _coerce_label_list(values: Sequence[object], n_samples: int, *, name: str) -> list[str]:
    labels = [_string_value(value) for value in list(values)]
    if len(labels) != n_samples:
        raise ValueError(f"{name} length must match rows")
    return labels


def _l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)


def _metric_key(value: object) -> str:
    return _string_value(value).replace(" ", "_").replace("/", "_")
