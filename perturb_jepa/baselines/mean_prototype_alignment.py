from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics

DEFAULT_GROUPBY = ("perturbation", "dose", "time", "cell_line")


@dataclass
class MeanPrototypeAlignment:
    """Map source conditions to target-space mean prototypes by metadata key."""

    groupby: str | Sequence[str] = DEFAULT_GROUPBY
    fallback: str = "global"
    prototypes_: dict[tuple[str, ...], np.ndarray] = field(init=False, default_factory=dict)
    global_prototype_: np.ndarray | None = field(init=False, default=None)
    groupby_: tuple[str, ...] | None = field(init=False, default=None)

    def fit(
        self,
        target_embeddings: np.ndarray,
        target_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    ) -> "MeanPrototypeAlignment":
        if self.fallback != "global":
            raise ValueError("only fallback='global' is currently supported")
        target = _as_2d_float_array(target_embeddings, name="target_embeddings")
        frame = _as_metadata_frame(target_metadata, n_samples=target.shape[0])
        groupby = _as_groupby_tuple(self.groupby)
        keys = _metadata_group_keys(frame, groupby)
        self.prototypes_ = {}
        for key in sorted(set(keys)):
            mask = np.asarray([candidate == key for candidate in keys], dtype=bool)
            self.prototypes_[key] = target[mask].mean(axis=0)
        self.global_prototype_ = target.mean(axis=0)
        self.groupby_ = groupby
        return self

    def predict(
        self,
        source_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    ) -> np.ndarray:
        self._check_fitted()
        frame = _as_metadata_frame(source_metadata)
        keys = _metadata_group_keys(frame, self.groupby_)
        return np.vstack([self.prototypes_.get(key, self.global_prototype_) for key in keys])

    def evaluate(
        self,
        source_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
        target_embeddings: np.ndarray,
        target_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
        *,
        label_col: str = "condition_key",
        ks: Sequence[int] = (1, 5, 10),
        prefix: str = "mean_prototype_alignment",
    ) -> dict[str, float]:
        predicted = self.predict(source_metadata)
        return directional_retrieval_metrics(
            predicted,
            target_embeddings,
            source_metadata,
            target_metadata,
            label_col=label_col,
            condition_cols=self.groupby_,
            ks=ks,
            prefix=prefix,
        )

    def _check_fitted(self) -> None:
        if self.global_prototype_ is None or self.groupby_ is None:
            raise RuntimeError("MeanPrototypeAlignment must be fit before predict/evaluate")


def mean_prototype_alignment_metrics(
    source_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    target_embeddings: np.ndarray,
    target_metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    groupby: str | Sequence[str] = DEFAULT_GROUPBY,
    label_col: str = "condition_key",
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    baseline = MeanPrototypeAlignment(groupby=groupby).fit(target_embeddings, target_metadata)
    return baseline.evaluate(
        source_metadata,
        target_embeddings,
        target_metadata,
        label_col=label_col,
        ks=ks,
    )


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
    n_samples: int | None = None,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if n_samples is not None and len(frame) != n_samples:
        raise ValueError("metadata row count must match embedding rows")
    return frame.reset_index(drop=True)


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
        tuple(_string_value(value) for value in row)
        for row in frame.loc[:, list(groupby)].itertuples(index=False, name=None)
    ]


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
