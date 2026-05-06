from __future__ import annotations

import numpy as np
import pandas as pd


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
