from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def condition_batch_confounding_report(
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    condition_col: str = "condition_key",
    batch_col: str = "batch",
    split_col: str | None = "split",
) -> pd.DataFrame:
    """Summarize whether biological conditions are replicated across batches."""

    frame = _as_metadata_frame(metadata)
    for column in (condition_col, batch_col):
        if column not in frame.columns:
            raise ValueError(f"{column!r} is missing from metadata")
    group_cols = [split_col] if split_col and split_col in frame.columns else [None]
    rows: list[dict[str, float | str]] = []
    for split_value, group in _iter_groups(frame, group_cols[0]):
        condition_batches = group.groupby(condition_col, dropna=False)[batch_col].nunique(dropna=False)
        batch_conditions = group.groupby(batch_col, dropna=False)[condition_col].nunique(dropna=False)
        if condition_batches.empty:
            continue
        rows.append(
            {
                "split": split_value,
                "n_rows": float(len(group)),
                "n_conditions": float(condition_batches.shape[0]),
                "n_batches": float(batch_conditions.shape[0]),
                "condition_batches_min": float(condition_batches.min()),
                "condition_batches_mean": float(condition_batches.mean()),
                "condition_batches_median": float(condition_batches.median()),
                "condition_batches_max": float(condition_batches.max()),
                "single_batch_condition_fraction": float(np.mean(condition_batches.to_numpy() <= 1)),
                "batch_conditions_min": float(batch_conditions.min()),
                "batch_conditions_mean": float(batch_conditions.mean()),
                "batch_conditions_median": float(batch_conditions.median()),
                "batch_conditions_max": float(batch_conditions.max()),
            }
        )
    return pd.DataFrame(rows)


def _iter_groups(frame: pd.DataFrame, split_col: str | None):
    if split_col is None:
        yield "all", frame
        return
    for split_value, group in frame.groupby(split_col, sort=True, dropna=False):
        yield _string_value(split_value), group


def _as_metadata_frame(metadata: pd.DataFrame | Mapping[str, Sequence[object]]) -> pd.DataFrame:
    return metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
