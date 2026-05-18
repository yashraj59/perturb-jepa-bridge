from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import pandas as pd

from perturb_jepa.data.schema import normalize_value


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


def random_sample_split(
    frame: pd.DataFrame,
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    """Assign train/val/test labels independently per row."""

    if fractions is None:
        fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    if not fractions:
        raise ValueError("fractions must not be empty")
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive value")

    labels = list(fractions)
    probabilities = [fractions[label] / total for label in labels]
    result = frame.copy()
    rng = pd.Series(range(len(result))).sample(frac=1.0, random_state=seed).index
    assignments = pd.Series(index=result.index, dtype=object)
    start = 0
    for label, probability in zip(labels, probabilities, strict=True):
        stop = start + round(probability * len(result))
        assignments.iloc[rng[start:stop]] = label
        start = stop
    if start < len(result):
        assignments.iloc[rng[start:]] = labels[-1]
    assignments = assignments.fillna(labels[-1])
    result[output_col] = assignments.to_numpy()
    return result


def _stable_groups(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"group columns missing from frame: {missing}")
    if len(group_cols) == 1:
        return frame[group_cols[0]].map(normalize_value)
    return frame.apply(lambda row: "|".join(normalize_value(row[column]) for column in group_cols), axis=1)


def _select_heldout_groups(
    groups: Sequence[str],
    *,
    heldout_groups: Sequence[object] | None,
    heldout_fraction: float,
    seed: int,
) -> set[str]:
    unique_groups = sorted({normalize_value(group) for group in groups})
    if not unique_groups:
        return set()
    if heldout_groups is not None:
        heldout = {normalize_value(group) for group in heldout_groups}
        missing = sorted(heldout.difference(unique_groups))
        if missing:
            raise ValueError(f"heldout groups are not present in frame: {missing}")
        return heldout
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between 0 and 1")
    count = max(1, round(len(unique_groups) * heldout_fraction))
    ranked = sorted(unique_groups, key=lambda group: _hash_to_unit_interval(f"{seed}|{group}"))
    return set(ranked[:count])


def _heldout_group_split(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    heldout_groups: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
    train_label: str = "train",
    heldout_label: str = "test",
) -> pd.DataFrame:
    groups = _stable_groups(frame, group_cols)
    normalized_heldout_groups = heldout_groups
    if heldout_groups is not None and len(group_cols) > 1:
        normalized_heldout_groups = [
            "|".join(normalize_value(value) for value in group)
            if isinstance(group, (tuple, list))
            else normalize_value(group)
            for group in heldout_groups
        ]
    heldout = _select_heldout_groups(
        groups.tolist(),
        heldout_groups=normalized_heldout_groups,
        heldout_fraction=heldout_fraction,
        seed=seed,
    )
    result = frame.copy()
    result["_split_group_key"] = groups.to_numpy()
    result[output_col] = result["_split_group_key"].map(lambda group: heldout_label if group in heldout else train_label)
    assert_group_split_integrity(result, ["_split_group_key"], split_col=output_col)
    return result.drop(columns=["_split_group_key"])


def heldout_batch_split(
    frame: pd.DataFrame,
    *,
    heldout_batches: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    return _heldout_group_split(
        frame,
        ["batch"],
        heldout_groups=heldout_batches,
        heldout_fraction=heldout_fraction,
        seed=seed,
        output_col=output_col,
    )


def heldout_dose_time_split(
    frame: pd.DataFrame,
    *,
    heldout_dose_times: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    return _heldout_group_split(
        frame,
        ["dose", "time"],
        heldout_groups=heldout_dose_times,
        heldout_fraction=heldout_fraction,
        seed=seed,
        output_col=output_col,
    )


def heldout_perturbation_split(
    frame: pd.DataFrame,
    *,
    heldout_perturbations: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    return _heldout_group_split(
        frame,
        ["perturbation"],
        heldout_groups=heldout_perturbations,
        heldout_fraction=heldout_fraction,
        seed=seed,
        output_col=output_col,
    )


def heldout_cell_line_split(
    frame: pd.DataFrame,
    *,
    heldout_cell_lines: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    return _heldout_group_split(
        frame,
        ["cell_line"],
        heldout_groups=heldout_cell_lines,
        heldout_fraction=heldout_fraction,
        seed=seed,
        output_col=output_col,
    )


def heldout_moa_split(
    frame: pd.DataFrame,
    *,
    heldout_moas: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
    output_col: str = "split",
) -> pd.DataFrame:
    if "moa" not in frame.columns:
        raise ValueError("moa column is required for heldout_moa_split")
    labels = frame["moa"].map(normalize_value)
    if labels.eq("NA").all():
        raise ValueError("heldout_moa_split requires at least one MoA label")
    return _heldout_group_split(
        frame,
        ["moa"],
        heldout_groups=heldout_moas,
        heldout_fraction=heldout_fraction,
        seed=seed,
        output_col=output_col,
    )


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
