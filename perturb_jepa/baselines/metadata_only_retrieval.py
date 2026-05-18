from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.retrieval import average_precision_from_scores, ranks_from_scores

METADATA_RETRIEVAL_COLUMNS = ("perturbation", "dose", "time", "cell_line")


def metadata_only_retrieval_metrics(
    query_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    gallery_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "condition_key",
    columns: Sequence[str] = METADATA_RETRIEVAL_COLUMNS,
    ks: Sequence[int] = (1, 5, 10),
    prefix: str = "metadata_only",
) -> dict[str, float]:
    """Retrieval baseline that scores only biological condition metadata."""

    query = _as_metadata_frame(query_metadata)
    gallery = _as_metadata_frame(gallery_metadata)
    scores = metadata_similarity_scores(query, gallery, columns=columns)
    query_labels = _labels(query, label_col=label_col, columns=columns)
    gallery_labels = _labels(gallery, label_col=label_col, columns=columns)
    return _metrics_from_scores(scores, query_labels, gallery_labels, ks=ks, prefix=prefix)


def metadata_similarity_scores(
    query_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    gallery_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    columns: Sequence[str],
) -> np.ndarray:
    query = _as_metadata_frame(query_metadata)
    gallery = _as_metadata_frame(gallery_metadata)
    available = [column for column in columns if column in query.columns and column in gallery.columns]
    if not available:
        raise ValueError("no requested metadata columns are present in both query and gallery")
    scores = np.zeros((len(query), len(gallery)), dtype=float)
    for weight, column in zip(_column_weights(available), available, strict=True):
        q_values = query[column].map(_string_value).to_numpy(dtype=object)
        g_values = gallery[column].map(_string_value).to_numpy(dtype=object)
        scores += weight * (q_values[:, None] == g_values[None, :])
    return scores


def _metrics_from_scores(
    scores: np.ndarray,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    ks: Sequence[int],
    prefix: str,
) -> dict[str, float]:
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


def _labels(frame: pd.DataFrame, *, label_col: str, columns: Sequence[str]) -> list[str]:
    if label_col in frame.columns:
        return [_string_value(value) for value in frame[label_col].tolist()]
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise ValueError(f"metadata must contain {label_col!r} or baseline metadata columns")
    return [
        "|".join(_string_value(value) for value in row)
        for row in frame.loc[:, available].itertuples(index=False, name=None)
    ]


def _column_weights(columns: Sequence[str]) -> list[float]:
    weights = {"perturbation": 4.0, "dose": 2.0, "time": 1.0, "cell_line": 1.0}
    return [weights.get(column, 1.0) for column in columns]


def _as_metadata_frame(metadata: pd.DataFrame | Mapping[str, Sequence[object]]) -> pd.DataFrame:
    return metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
