from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import pandas as pd


def _hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def grouped_hash_split(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    """Assign train/val/test labels without splitting rows from the same group."""

    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    if not fractions:
        raise ValueError("fractions must not be empty")
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"group columns missing from frame: {missing}")

    normalized = {name: value / total for name, value in fractions.items()}
    cutoffs: list[tuple[str, float]] = []
    running = 0.0
    for name, fraction in normalized.items():
        running += fraction
        cutoffs.append((name, running))

    result = frame.copy()

    def assign(row: pd.Series) -> str:
        key = "|".join(str(row[column]) for column in group_cols)
        value = _hash_to_unit_interval(f"{seed}|{key}")
        for name, cutoff in cutoffs:
            if value <= cutoff:
                return name
        return cutoffs[-1][0]

    result[output_col] = result.apply(assign, axis=1)
    return result


def assert_group_split_integrity(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    split_col: str = "split",
) -> None:
    missing = [column for column in (*group_cols, split_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"columns missing from frame: {missing}")
    counts = frame.groupby(list(group_cols))[split_col].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        example = leaked.index[0]
        raise ValueError(f"group split leakage detected for group {example!r}")
