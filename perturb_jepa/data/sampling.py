from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from perturb_jepa.data.schema import normalize_value


NUISANCE_COLUMNS = ("cell_line", "batch", "plate", "time")
MISSING_NEGATIVE_INDEX = -1


@dataclass(frozen=True)
class HardNegativeCandidates:
    indices: np.ndarray
    matched_nuisance_columns: tuple[str, ...]
    used_fallback: bool


@dataclass(frozen=True)
class HardNegativeSamples:
    indices: np.ndarray
    candidate_counts: np.ndarray
    matched_nuisance_columns: tuple[tuple[str, ...], ...]
    used_fallback: np.ndarray


def _available_columns(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    available: list[str] = []
    for column in columns:
        if column in frame.columns and column not in seen:
            available.append(column)
            seen.add(column)
    return tuple(available)


def _normalize_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        normalized[column] = normalized[column].map(normalize_value)
    return normalized


def stratified_hard_negative_candidates(
    frame: pd.DataFrame,
    anchor_index: int,
    *,
    perturbation_col: str = "perturbation",
    nuisance_cols: Sequence[str] = NUISANCE_COLUMNS,
    fallback_to_any: bool = True,
) -> HardNegativeCandidates:
    """Return positional negatives with matched nuisance metadata and different perturbation.

    The first pass requires all available nuisance columns to match the anchor.
    If no such row exists, the optional fallback returns any row with a different
    perturbation so callers can keep fixed-size batches without special casing.
    """

    if perturbation_col not in frame.columns:
        raise ValueError(f"{perturbation_col!r} not found in metadata frame")
    if anchor_index < 0 or anchor_index >= len(frame):
        raise IndexError(f"anchor_index {anchor_index} is out of bounds for frame of length {len(frame)}")

    nuisance = _available_columns(frame, nuisance_cols)
    metadata = _normalize_columns(frame, (perturbation_col, *nuisance))
    anchor = metadata.iloc[anchor_index]

    different_perturbation = metadata[perturbation_col] != anchor[perturbation_col]
    matched = different_perturbation.copy()
    for column in nuisance:
        matched &= metadata[column] == anchor[column]

    indices = np.flatnonzero(matched.to_numpy())
    if indices.size > 0 or not fallback_to_any:
        return HardNegativeCandidates(
            indices=indices.astype(np.int64, copy=False),
            matched_nuisance_columns=nuisance,
            used_fallback=False,
        )

    fallback = np.flatnonzero(different_perturbation.to_numpy())
    return HardNegativeCandidates(
        indices=fallback.astype(np.int64, copy=False),
        matched_nuisance_columns=tuple(),
        used_fallback=fallback.size > 0,
    )


def sample_stratified_hard_negatives(
    frame: pd.DataFrame,
    *,
    n_negatives: int = 1,
    perturbation_col: str = "perturbation",
    nuisance_cols: Sequence[str] = NUISANCE_COLUMNS,
    fallback_to_any: bool = True,
    replace: bool = True,
    seed: int | None = None,
) -> HardNegativeSamples:
    """Sample positional hard-negative indices for every row in a metadata frame."""

    if n_negatives <= 0:
        raise ValueError("n_negatives must be positive")
    if perturbation_col not in frame.columns:
        raise ValueError(f"{perturbation_col!r} not found in metadata frame")

    rng = np.random.default_rng(seed)
    sampled = np.full((len(frame), n_negatives), MISSING_NEGATIVE_INDEX, dtype=np.int64)
    candidate_counts = np.zeros(len(frame), dtype=np.int64)
    matched_columns: list[tuple[str, ...]] = []
    used_fallback = np.zeros(len(frame), dtype=bool)

    for anchor_index in range(len(frame)):
        candidates = stratified_hard_negative_candidates(
            frame,
            anchor_index,
            perturbation_col=perturbation_col,
            nuisance_cols=nuisance_cols,
            fallback_to_any=fallback_to_any,
        )
        candidate_counts[anchor_index] = candidates.indices.size
        matched_columns.append(candidates.matched_nuisance_columns)
        used_fallback[anchor_index] = candidates.used_fallback
        if candidates.indices.size == 0:
            continue

        if replace:
            selected = rng.choice(candidates.indices, size=n_negatives, replace=True)
            sampled[anchor_index] = selected
        else:
            sample_size = min(n_negatives, candidates.indices.size)
            selected = rng.choice(candidates.indices, size=sample_size, replace=False)
            sampled[anchor_index, :sample_size] = selected

    return HardNegativeSamples(
        indices=sampled,
        candidate_counts=candidate_counts,
        matched_nuisance_columns=tuple(matched_columns),
        used_fallback=used_fallback,
    )


def add_stratified_hard_negative_indices(
    frame: pd.DataFrame,
    *,
    output_col: str = "hard_negative_index",
    include_diagnostics: bool = True,
    **sampling_kwargs,
) -> pd.DataFrame:
    samples = sample_stratified_hard_negatives(frame, **sampling_kwargs)
    result = frame.copy()
    if samples.indices.shape[1] == 1:
        result[output_col] = samples.indices[:, 0]
    else:
        result[output_col] = [tuple(indices.tolist()) for indices in samples.indices]
    if include_diagnostics:
        result[f"{output_col}_candidate_count"] = samples.candidate_counts
        result[f"{output_col}_used_fallback"] = samples.used_fallback
    return result
