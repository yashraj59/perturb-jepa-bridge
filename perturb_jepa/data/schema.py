from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

SCRNA_OBS_COLUMNS = (
    "perturbation",
    "perturbation_type",
    "dose",
    "time",
    "cell_line",
    "batch",
)

SCRNA_VAR_COLUMNS = ("gene_id", "gene_symbol")

IMAGE_COLUMNS = (
    "image_path",
    "plate",
    "well",
    "site",
    "channel_or_z",
    "perturbation",
    "compound",
    "moa",
    "target_gene",
    "dose",
    "time",
    "cell_line",
    "batch",
)

CONDITION_COLUMNS = (
    "perturbation",
    "perturbation_type",
    "dose",
    "time",
    "cell_line",
)

DEFAULT_METADATA = {
    "perturbation": "unknown",
    "perturbation_type": "unknown",
    "dose": "NA",
    "time": "NA",
    "cell_line": "unknown",
    "batch": "unknown",
    "compound": "",
    "moa": "",
    "target_gene": "",
}


@dataclass(frozen=True)
class ConditionKey:
    perturbation: str
    perturbation_type: str
    dose: str
    time: str
    cell_line: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ConditionKey":
        return cls(
            perturbation=normalize_value(row.get("perturbation", "unknown")),
            perturbation_type=normalize_value(row.get("perturbation_type", "unknown")),
            dose=normalize_value(row.get("dose", "NA")),
            time=normalize_value(row.get("time", "NA")),
            cell_line=normalize_value(row.get("cell_line", "unknown")),
        )

    def as_string(self) -> str:
        return "|".join(
            (
                self.perturbation,
                self.perturbation_type,
                self.dose,
                self.time,
                self.cell_line,
            )
        )


def normalize_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "NA"
    return text


def assert_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_metadata_frame(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    defaults: Mapping[str, object] | None = None,
    name: str = "metadata",
) -> pd.DataFrame:
    defaults = {**DEFAULT_METADATA, **(defaults or {})}
    normalized = frame.copy()
    for column in required:
        if column not in normalized.columns and column in defaults:
            normalized[column] = defaults[column]
    assert_columns(normalized, required, name=name)
    for column in required:
        normalized[column] = normalized[column].map(normalize_value)
    return normalized


def add_condition_key(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = CONDITION_COLUMNS,
    output_col: str = "condition_key",
) -> pd.DataFrame:
    assert_columns(frame, columns, name="condition metadata")
    keyed = frame.copy()
    keyed[output_col] = keyed.apply(
        lambda row: ConditionKey.from_mapping({column: row[column] for column in columns}).as_string(),
        axis=1,
    )
    return keyed


def normalize_scrna_obs(obs: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_metadata_frame(obs, SCRNA_OBS_COLUMNS, name="AnnData.obs")
    return add_condition_key(normalized)


def normalize_scrna_var(var: pd.DataFrame) -> pd.DataFrame:
    normalized = var.copy()
    if "gene_id" not in normalized.columns:
        normalized["gene_id"] = normalized.index.astype(str)
    if "gene_symbol" not in normalized.columns:
        normalized["gene_symbol"] = normalized.index.astype(str)
    return normalize_metadata_frame(
        normalized,
        SCRNA_VAR_COLUMNS,
        defaults={"gene_id": "unknown", "gene_symbol": "unknown"},
        name="AnnData.var",
    )


def normalize_image_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_metadata_frame(
        frame,
        (*IMAGE_COLUMNS, "perturbation_type"),
        defaults={"perturbation_type": "unknown"},
        name="image manifest",
    )
    return add_condition_key(normalized)
