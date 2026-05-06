from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from perturb_jepa.data.schema import add_condition_key, normalize_value


_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_metadata_float(value: object, *, default: float = 0.0) -> float:
    """Parse compact dose/time metadata such as ``10uM`` or ``48h``."""

    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return default
        return float(value)
    text = normalize_value(value).replace(",", "")
    if text == "NA":
        return default
    match = _NUMERIC_RE.search(text)
    if match is None:
        return default
    return float(match.group(0))


def _category_map(values: Iterable[object], *, default: str = "unknown") -> dict[str, int]:
    normalized = {normalize_value(value) for value in values}
    keys = [default]
    keys.extend(sorted(value for value in normalized if value != default))
    return {value: index for index, value in enumerate(keys)}


@dataclass(frozen=True)
class MetadataVocab:
    perturbation_to_id: dict[str, int]
    perturbation_type_to_id: dict[str, int]
    cell_line_to_id: dict[str, int]
    batch_to_id: dict[str, int]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MetadataVocab":
        return cls.from_frames([frame])

    @classmethod
    def from_frames(cls, frames: Sequence[pd.DataFrame]) -> "MetadataVocab":
        if not frames:
            frames = [pd.DataFrame()]

        def collect(column: str) -> list[object]:
            values: list[object] = []
            for frame in frames:
                if column in frame.columns:
                    values.extend(frame[column].tolist())
            return values

        return cls(
            perturbation_to_id=_category_map(collect("perturbation")),
            perturbation_type_to_id=_category_map(collect("perturbation_type")),
            cell_line_to_id=_category_map(collect("cell_line")),
            batch_to_id=_category_map(collect("batch")),
        )

    @property
    def num_perturbations(self) -> int:
        return len(self.perturbation_to_id)

    @property
    def num_types(self) -> int:
        return len(self.perturbation_type_to_id)

    @property
    def num_cell_lines(self) -> int:
        return len(self.cell_line_to_id)

    @property
    def num_batches(self) -> int:
        return len(self.batch_to_id)

    def to_config_kwargs(self) -> dict[str, int]:
        return {
            "num_perturbations": self.num_perturbations,
            "num_types": self.num_types,
            "num_cell_lines": self.num_cell_lines,
            "num_batches": self.num_batches,
        }

    def _lookup(self, mapping: Mapping[str, int], value: object) -> int:
        return mapping.get(normalize_value(value), 0)

    def encode_row(self, row: Mapping[str, object]) -> dict[str, int | float]:
        return {
            "perturbation_id": self._lookup(self.perturbation_to_id, row.get("perturbation", "unknown")),
            "perturbation_type_id": self._lookup(
                self.perturbation_type_to_id,
                row.get("perturbation_type", "unknown"),
            ),
            "cell_line_id": self._lookup(self.cell_line_to_id, row.get("cell_line", "unknown")),
            "batch_id": self._lookup(self.batch_to_id, row.get("batch", "unknown")),
            "dose": parse_metadata_float(row.get("dose", "NA")),
            "time": parse_metadata_float(row.get("time", "NA")),
        }

    def encode_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        rows = [self.encode_row(row) for row in encoded.to_dict(orient="records")]
        encoded_values = pd.DataFrame(rows, index=encoded.index)
        for column in encoded_values.columns:
            encoded[column] = encoded_values[column]
        return encoded


@dataclass(frozen=True)
class ConditionBags:
    keys: list[str]
    indices: dict[str, np.ndarray]
    counts: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.indices[key]


def build_condition_bags(
    frame: pd.DataFrame,
    *,
    key_col: str = "condition_key",
    sort: bool = True,
) -> ConditionBags:
    if key_col not in frame.columns:
        frame = add_condition_key(frame, output_col=key_col)
    grouped = frame.reset_index(drop=True).groupby(key_col, sort=sort).indices
    indices = {normalize_value(key): np.asarray(value, dtype=np.int64) for key, value in grouped.items()}
    keys = list(indices)
    counts = np.asarray([len(indices[key]) for key in keys], dtype=np.int64)
    return ConditionBags(keys=keys, indices=indices, counts=counts)


@dataclass(frozen=True)
class ConditionPrototypes:
    keys: list[str]
    values: np.ndarray
    counts: np.ndarray

    @property
    def key_to_index(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.keys)}

    def lookup(self, condition_keys: Sequence[str]) -> np.ndarray:
        key_to_index = self.key_to_index
        missing = [key for key in condition_keys if key not in key_to_index]
        if missing:
            raise KeyError(f"condition keys missing from prototypes: {missing}")
        return self.values[[key_to_index[key] for key in condition_keys]]


def compute_condition_prototypes(
    values: np.ndarray,
    condition_keys: Sequence[str],
    *,
    reducer: str = "mean",
    sort: bool = True,
) -> ConditionPrototypes:
    array = np.asarray(values, dtype=np.float32)
    keys = [normalize_value(key) for key in condition_keys]
    if array.shape[0] != len(keys):
        raise ValueError("values and condition_keys must have the same first dimension")
    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'")

    frame = pd.DataFrame({"condition_key": keys})
    bags = build_condition_bags(frame, sort=sort)
    prototypes: list[np.ndarray] = []
    for key in bags.keys:
        members = array[bags.indices[key]]
        if reducer == "mean":
            prototypes.append(members.mean(axis=0))
        else:
            prototypes.append(np.median(members, axis=0))
    stacked = np.stack(prototypes, axis=0).astype(np.float32, copy=False)
    return ConditionPrototypes(keys=bags.keys, values=stacked, counts=bags.counts)


def prototype_lookup_indices(
    condition_keys: Sequence[str],
    prototype_keys: Sequence[str],
) -> np.ndarray:
    key_to_index = {key: index for index, key in enumerate(prototype_keys)}
    missing = [key for key in condition_keys if key not in key_to_index]
    if missing:
        raise KeyError(f"condition keys missing from prototypes: {missing}")
    return np.asarray([key_to_index[key] for key in condition_keys], dtype=np.int64)
