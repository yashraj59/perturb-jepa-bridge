from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from perturb_jepa.data.schema import (
    DEFAULT_METADATA_SCHEMA,
    make_bio_key,
    make_condition_id,
    make_tech_key,
    normalize_value,
    validate_metadata_columns,
)


SplitName = Literal["train", "val", "test"]
TechSummaryStrategy = Literal["mode", "first_non_na", "set"]


def _condition_id_from_row(row: Mapping[str, object], *, key_col: str) -> str:
    if key_col in row:
        return normalize_value(row.get(key_col))
    return make_condition_id(row)


def _metadata_with_condition_ids(frame: pd.DataFrame, *, key_col: str = "condition_key") -> pd.DataFrame:
    validate_metadata_columns(frame)
    metadata = frame.copy()
    for column in DEFAULT_METADATA_SCHEMA.all_biological_keys:
        if column in metadata.columns:
            metadata[column] = metadata[column].map(normalize_value)
    for column in DEFAULT_METADATA_SCHEMA.technical_keys:
        if column in metadata.columns:
            metadata[column] = metadata[column].map(normalize_value)
    has_key_col = key_col in metadata.columns
    if has_key_col:
        metadata[key_col] = metadata[key_col].map(normalize_value)
    metadata["condition_id"] = [
        _condition_id_from_row(row, key_col=key_col) for row in metadata.to_dict(orient="records")
    ]
    metadata["bio_key"] = [
        str(row["condition_id"]) if has_key_col else make_bio_key(row)
        for row in metadata.to_dict(orient="records")
    ]
    return metadata


def _sample_ids(metadata: pd.DataFrame, sample_id_col: str | None) -> np.ndarray:
    if sample_id_col is not None:
        if sample_id_col not in metadata.columns:
            raise ValueError(f"sample_id_col {sample_id_col!r} is missing from metadata")
        return metadata[sample_id_col].astype(str).to_numpy()
    if "sample_id" in metadata.columns:
        return metadata["sample_id"].astype(str).to_numpy()
    return metadata.index.astype(str).to_numpy()


def _sample_groups(metadata: pd.DataFrame, balanced_sample_col: str | None) -> np.ndarray | None:
    if balanced_sample_col is None or balanced_sample_col == "":
        return None
    if balanced_sample_col not in metadata.columns:
        raise ValueError(f"balanced_sample_col {balanced_sample_col!r} is missing from metadata")
    return metadata[balanced_sample_col].map(normalize_value).to_numpy(dtype=object)


def _condition_indices(metadata: pd.DataFrame, min_bag_size: int) -> dict[str, np.ndarray]:
    if min_bag_size <= 0:
        raise ValueError("min_bag_size must be positive")
    grouped = metadata.reset_index(drop=True).groupby("condition_id", sort=True).indices
    indices = {
        str(condition_id): np.asarray(row_indices, dtype=np.int64)
        for condition_id, row_indices in grouped.items()
        if len(row_indices) >= min_bag_size
    }
    if not indices:
        raise ValueError("no condition bags satisfy the minimum bag size")
    return indices


def _choose_indices(
    indices: np.ndarray,
    *,
    bag_size: int | None,
    split: SplitName,
    rng: np.random.Generator,
    sample_groups: np.ndarray | None = None,
) -> np.ndarray:
    if bag_size is not None and bag_size <= 0:
        raise ValueError("bag_size must be positive")
    if bag_size is None or bag_size >= len(indices):
        return np.asarray(indices, dtype=np.int64)
    if split == "train":
        if sample_groups is not None:
            selected = _balanced_group_sample(indices, sample_groups, bag_size=bag_size, rng=rng)
            if selected is not None:
                return selected
        return np.sort(rng.choice(indices, size=bag_size, replace=False)).astype(np.int64, copy=False)
    return np.asarray(indices[:bag_size], dtype=np.int64)


def _balanced_group_sample(
    indices: np.ndarray,
    sample_groups: np.ndarray,
    *,
    bag_size: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    labels = np.asarray(sample_groups, dtype=object)[indices]
    groups: dict[str, np.ndarray] = {}
    for label in sorted({normalize_value(value) for value in labels}):
        members = indices[labels == label]
        if len(members) > 0:
            groups[label] = rng.permutation(members).astype(np.int64, copy=False)
    if len(groups) <= 1:
        return None

    group_names = list(rng.permutation(np.asarray(sorted(groups), dtype=object)))
    selected: list[int] = []
    cursors = {name: 0 for name in group_names}
    while len(selected) < bag_size:
        progressed = False
        for name in group_names:
            cursor = cursors[name]
            members = groups[name]
            if cursor >= len(members):
                continue
            selected.append(int(members[cursor]))
            cursors[name] = cursor + 1
            progressed = True
            if len(selected) >= bag_size:
                break
        if not progressed:
            break
    if len(selected) < bag_size:
        remaining = np.setdiff1d(indices, np.asarray(selected, dtype=np.int64), assume_unique=False)
        if len(remaining) > 0:
            needed = min(bag_size - len(selected), len(remaining))
            selected.extend(int(value) for value in rng.choice(remaining, size=needed, replace=False))
    if len(selected) != bag_size:
        return None
    return np.sort(np.asarray(selected, dtype=np.int64))


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.to_dict(orient="records")


def _tech_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(dict(zip(DEFAULT_METADATA_SCHEMA.technical_keys, make_tech_key(row), strict=True)))
    return rows


def summarize_technical_metadata(
    records: Sequence[Mapping[str, object]],
    *,
    strategy: TechSummaryStrategy = "mode",
    columns: Sequence[str] = DEFAULT_METADATA_SCHEMA.technical_keys,
) -> dict[str, str]:
    """Summarize per-cell/per-image technical metadata into deterministic bag labels.

    ``mode`` excludes NA-like values when possible and breaks ties lexicographically.
    ``first_non_na`` returns the first non-NA-like value in row order.
    ``set`` returns a semicolon-joined sorted set of non-NA-like values.
    """

    if strategy not in {"mode", "first_non_na", "set"}:
        raise ValueError("strategy must be one of 'mode', 'first_non_na', or 'set'")
    summary: dict[str, str] = {}
    for column in columns:
        values = [normalize_value(row.get(column, "NA")) for row in records]
        non_na = [value for value in values if value != "NA"]
        candidates = non_na or values or ["NA"]
        if strategy == "first_non_na":
            summary[column] = candidates[0]
        elif strategy == "set":
            summary[column] = ";".join(sorted(set(candidates)))
        else:
            counts: dict[str, int] = {}
            for value in candidates:
                counts[value] = counts.get(value, 0) + 1
            summary[column] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return summary


def _condition_record(row: Mapping[str, object], *, key_col: str = "condition_key") -> dict[str, object]:
    record = {
        column: normalize_value(row.get(column, "NA"))
        for column in DEFAULT_METADATA_SCHEMA.all_biological_keys
        if column in row or column in DEFAULT_METADATA_SCHEMA.biological_keys
    }
    for column in (
        "condition_key_coarse",
        "condition_key_medium",
        "condition_key_fine",
        "condition_key_with_type",
    ):
        if column in row:
            record[column] = normalize_value(row.get(column))
    condition_id = normalize_value(row.get("condition_id", _condition_id_from_row(row, key_col=key_col)))
    record["bio_key"] = condition_id
    record["condition_id"] = condition_id
    record["condition_key"] = condition_id
    return record


class RNAConditionBagDataset(Dataset):
    """Condition-level bags of scRNA rows."""

    def __init__(
        self,
        matrix: np.ndarray,
        metadata: pd.DataFrame,
        *,
        rna_bag_size: int | None = 8,
        min_rna_bag_size: int = 1,
        split: SplitName = "train",
        seed: int | None = None,
        sample_id_col: str | None = None,
        condition_key_col: str = "condition_key",
        balanced_sample_col: str | None = None,
    ) -> None:
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("matrix must have shape [cells, features]")
        if len(metadata) != values.shape[0]:
            raise ValueError("metadata rows must match matrix cells")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        self.matrix = values
        self.condition_key_col = condition_key_col
        self.metadata = _metadata_with_condition_ids(metadata, key_col=condition_key_col)
        self.sample_ids = _sample_ids(self.metadata, sample_id_col)
        self.sample_groups = _sample_groups(self.metadata, balanced_sample_col)
        self.rna_bag_size = rna_bag_size
        self.split: SplitName = split
        self.rng = np.random.default_rng(seed)
        self.condition_to_indices = _condition_indices(self.metadata, min_rna_bag_size)
        self.condition_ids = sorted(self.condition_to_indices)

    def __len__(self) -> int:
        return len(self.condition_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.get_condition(self.condition_ids[index])

    def get_condition(self, condition_id: str) -> dict[str, object]:
        if condition_id not in self.condition_to_indices:
            raise KeyError(f"unknown condition_id {condition_id!r}")
        selected = _choose_indices(
            self.condition_to_indices[condition_id],
            bag_size=self.rna_bag_size,
            split=self.split,
            rng=self.rng,
            sample_groups=self.sample_groups,
        )
        rows = self.metadata.iloc[selected]
        row = rows.iloc[0].to_dict()
        return {
            "bio_key": row["bio_key"],
            "condition": _condition_record(row, key_col=self.condition_key_col),
            "condition_id": condition_id,
            "rna": self.matrix[selected].copy(),
            "sample_ids": self.sample_ids[selected].tolist(),
            "tech_metadata": _tech_records(rows),
            "cell_meta": _records(rows),
        }


class ImageConditionBagDataset(Dataset):
    """Condition-level bags of image rows or preloaded image arrays."""

    def __init__(
        self,
        images: np.ndarray | Sequence[np.ndarray] | pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        *,
        image_bag_size: int | None = 8,
        min_image_bag_size: int = 1,
        split: SplitName = "train",
        seed: int | None = None,
        sample_id_col: str | None = None,
        image_root: str | Path = "",
        image_path_col: str = "image_path",
        channels: int | None = None,
        resize: tuple[int, int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        condition_key_col: str = "condition_key",
        balanced_sample_col: str | None = None,
    ) -> None:
        if isinstance(images, pd.DataFrame) and metadata is None:
            metadata = images
            image_values = None
        else:
            if metadata is None:
                raise ValueError("metadata is required when images are provided separately")
            image_values = np.asarray(images)
            if image_values.shape[0] != len(metadata):
                raise ValueError("metadata rows must match images")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        self.images = image_values
        self.condition_key_col = condition_key_col
        self.metadata = _metadata_with_condition_ids(metadata, key_col=condition_key_col)
        if self.images is None and image_path_col not in self.metadata.columns:
            raise ValueError(f"{image_path_col!r} is required when images are loaded from paths")
        self.sample_ids = _sample_ids(self.metadata, sample_id_col)
        self.sample_groups = _sample_groups(self.metadata, balanced_sample_col)
        self.image_bag_size = image_bag_size
        self.split: SplitName = split
        self.rng = np.random.default_rng(seed)
        self.condition_to_indices = _condition_indices(self.metadata, min_image_bag_size)
        self.condition_ids = sorted(self.condition_to_indices)
        self.image_root = Path(image_root)
        self.image_path_col = image_path_col
        self.channels = channels
        self.resize = resize
        self.transform = transform

    def __len__(self) -> int:
        return len(self.condition_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.get_condition(self.condition_ids[index])

    def get_condition(self, condition_id: str) -> dict[str, object]:
        if condition_id not in self.condition_to_indices:
            raise KeyError(f"unknown condition_id {condition_id!r}")
        selected = _choose_indices(
            self.condition_to_indices[condition_id],
            bag_size=self.image_bag_size,
            split=self.split,
            rng=self.rng,
            sample_groups=self.sample_groups,
        )
        rows = self.metadata.iloc[selected]
        row = rows.iloc[0].to_dict()
        images = np.stack([self._load_image(index) for index in selected]).astype(np.float32, copy=False)
        return {
            "bio_key": row["bio_key"],
            "condition": _condition_record(row, key_col=self.condition_key_col),
            "condition_id": condition_id,
            "image": images,
            "sample_ids": self.sample_ids[selected].tolist(),
            "tech_metadata": _tech_records(rows),
            "cell_meta": _records(rows),
        }

    def _load_image(self, index: int) -> np.ndarray:
        if self.images is not None:
            image = np.asarray(self.images[index], dtype=np.float32)
        else:
            from perturb_jepa.data.images import load_image_array

            path = Path(str(self.metadata.iloc[index][self.image_path_col]))
            if not path.is_absolute() and str(self.image_root):
                path = self.image_root / path
            image = load_image_array(path, channels=self.channels, resize=self.resize)
        if self.transform is not None:
            image = self.transform(image)
        return np.asarray(image, dtype=np.float32)


class PairedConditionBagDataset(Dataset):
    """Paired RNA and image bags matched by biological condition."""

    def __init__(
        self,
        rna_dataset: RNAConditionBagDataset,
        image_dataset: ImageConditionBagDataset,
    ) -> None:
        self.rna_dataset = rna_dataset
        self.image_dataset = image_dataset
        shared = set(rna_dataset.condition_ids).intersection(image_dataset.condition_ids)
        if not shared:
            raise ValueError("RNA and image datasets have no overlapping biological conditions")
        self.condition_ids = sorted(shared)

    def __len__(self) -> int:
        return len(self.condition_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        condition_id = self.condition_ids[index]
        rna = self.rna_dataset.get_condition(condition_id)
        image = self.image_dataset.get_condition(condition_id)
        return {
            "bio_key": rna["bio_key"],
            "condition": rna["condition"],
            "condition_id": condition_id,
            "rna": {
                "x": rna["rna"],
                "sample_ids": rna["sample_ids"],
                "tech": rna["tech_metadata"],
                "cell_meta": rna["cell_meta"],
            },
            "image": {
                "x": image["image"],
                "sample_ids": image["sample_ids"],
                "tech": image["tech_metadata"],
                "cell_meta": image["cell_meta"],
            },
        }
