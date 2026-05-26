"""Metadata-only adapter helpers for condition-paired scGeneScope validation.

The adapter deliberately starts at the manifest level. It should prove split,
modality, and condition-pair contracts before any large image or count payload is
downloaded or opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import urllib.parse
import urllib.request

import pandas as pd


SCGENESCOPE_HF_REPO = "altoslabs/scGeneScope"
REQUIRED_SCGENESCOPE_MANIFEST_COLUMNS = (
    "modality",
    "treatment_id",
    "round",
    "batch",
    "replicate",
    "split",
    "uri",
)
SCGENESCOPE_OPTIONAL_DEFAULT_COLUMNS = {"dose": "fixed"}
SCGENESCOPE_PAIR_COLUMNS = ("treatment_id", "dose", "round")
SCGENESCOPE_FORBIDDEN_MODEL_INPUT_COLUMNS = (
    "condition_key",
    "biological_key",
    "target_key",
    "exact_target_key",
)
SCGENESCOPE_VALID_MODALITIES = ("rna", "image")
SCGENESCOPE_COLUMN_ALIASES = {
    "Treatment": "treatment_id",
    "treatment": "treatment_id",
    "Replicate": "replicate",
    "Batch": "batch",
    "Group": "group",
    "cell_id": "cell_id",
}
SCGENESCOPE_LIGHT_METADATA_PATTERNS = (
    "metadata",
    "manifest",
    "split",
    "treatment",
    "condition",
    "sample",
)


@dataclass(frozen=True)
class ScGeneScopeAudit:
    root: str
    root_exists: bool
    manifest_path: str | None
    manifest_valid: bool
    paired_condition_count: int
    rna_record_count: int
    image_record_count: int
    split_count: int
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "root_exists": self.root_exists,
            "manifest_path": self.manifest_path,
            "manifest_valid": self.manifest_valid,
            "paired_condition_count": self.paired_condition_count,
            "rna_record_count": self.rna_record_count,
            "image_record_count": self.image_record_count,
            "split_count": self.split_count,
            "message": self.message,
        }


def scgenescope_manifest_template() -> pd.DataFrame:
    columns = list(REQUIRED_SCGENESCOPE_MANIFEST_COLUMNS) + list(SCGENESCOPE_OPTIONAL_DEFAULT_COLUMNS)
    return pd.DataFrame(
        [
            {
                "modality": "rna",
                "treatment_id": "control",
                "dose": "0",
                "round": "round_1",
                "batch": "batch_1",
                "replicate": "replicate_1",
                "split": "train",
                "uri": "relative/or/remote/rna_feature_file.h5ad",
            },
            {
                "modality": "image",
                "treatment_id": "control",
                "dose": "0",
                "round": "round_1",
                "batch": "batch_1",
                "replicate": "replicate_1",
                "split": "train",
                "uri": "relative/or/remote/image_feature_file.parquet",
            },
        ],
        columns=columns,
    )


def load_scgenescope_manifest(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        frame = pd.read_csv(path)
    return validate_scgenescope_manifest(frame)


def validate_scgenescope_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _rename_scgenescope_aliases(frame)
    if "split" not in frame.columns and "replicate" in frame.columns:
        frame = frame.copy()
        frame["split"] = frame["replicate"].map(scgenescope_split_from_replicate)
    for column, default_value in SCGENESCOPE_OPTIONAL_DEFAULT_COLUMNS.items():
        if column not in frame.columns:
            frame = frame.copy()
            frame[column] = default_value
        else:
            frame = frame.copy()
            frame[column] = frame[column].fillna(default_value).replace("", default_value)
    missing = [column for column in REQUIRED_SCGENESCOPE_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"scGeneScope manifest missing required columns: {missing}")
    forbidden_present = [column for column in SCGENESCOPE_FORBIDDEN_MODEL_INPUT_COLUMNS if column in frame.columns]
    if forbidden_present:
        raise ValueError(f"scGeneScope manifest contains forbidden shortcut columns: {forbidden_present}")
    out = frame.copy()
    for column in REQUIRED_SCGENESCOPE_MANIFEST_COLUMNS:
        out[column] = out[column].astype(str)
    out["modality"] = out["modality"].str.lower()
    invalid_modalities = sorted(set(out["modality"]) - set(SCGENESCOPE_VALID_MODALITIES))
    if invalid_modalities:
        raise ValueError(f"scGeneScope manifest has invalid modalities: {invalid_modalities}")
    if out["uri"].str.len().eq(0).any():
        raise ValueError("scGeneScope manifest has empty uri values")
    return out.reset_index(drop=True)


def scgenescope_split_from_replicate(replicate: object) -> str:
    text = str(replicate).strip().lower().replace("replicate_", "")
    if text in {"3", "rep3"}:
        return "train"
    if text in {"5", "rep5"}:
        return "validation"
    if text in {"4", "rep4"}:
        return "test"
    if text in {"1", "2", "rep1", "rep2"}:
        return "alternate_test"
    return "unknown"


def _rename_scgenescope_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for source, target in SCGENESCOPE_COLUMN_ALIASES.items():
        if source in frame.columns and target not in frame.columns:
            rename[source] = target
    return frame.rename(columns=rename)


def validate_scgenescope_croissant_contract(fields: pd.DataFrame, split_contract: dict[str, object]) -> dict[str, object]:
    field_names = set(fields["field"].astype(str)) if "field" in fields.columns else set()
    required_fields = {"cell_id", "Treatment", "Replicate", "batch", "Group"}
    missing_fields = sorted(required_fields - field_names)
    replicate_to_split = split_contract.get("replicate_to_split", {})
    if not isinstance(replicate_to_split, dict):
        replicate_to_split = {}
    expected_mapping = {
        "1": "alternate_test",
        "2": "alternate_test",
        "3": "train",
        "4": "test",
        "5": "validation",
    }
    mapping_mismatches = {
        replicate: {
            "expected": expected_split,
            "observed": str(replicate_to_split.get(replicate, "")),
        }
        for replicate, expected_split in expected_mapping.items()
        if str(replicate_to_split.get(replicate, "")) != expected_split
    }
    return {
        "required_fields_present": not missing_fields,
        "missing_fields": missing_fields,
        "replicate_split_mapping_valid": not mapping_mismatches,
        "mapping_mismatches": mapping_mismatches,
        "dose_fixed_or_manifest_supplied": True,
        "target_column": str(split_contract.get("target_column", "")),
        "control_column": str(split_contract.get("control_column", "")),
        "batch_column": str(split_contract.get("batch_column", "")),
        "adapter_contract_valid": not missing_fields and not mapping_mismatches,
    }


def scgenescope_croissant_dry_run_manifest(
    split_contract: dict[str, object],
    *,
    treatments: Iterable[str] = ("DMSO", "treatment_a"),
    rounds: Iterable[str] = ("round_1", "round_2"),
    modalities: Iterable[str] = SCGENESCOPE_VALID_MODALITIES,
) -> pd.DataFrame:
    replicate_to_split = split_contract.get("replicate_to_split", {})
    if not isinstance(replicate_to_split, dict):
        replicate_to_split = {}
    rows: list[dict[str, str]] = []
    for round_id in rounds:
        for treatment in treatments:
            for replicate, split in sorted(replicate_to_split.items(), key=lambda item: str(item[0])):
                for modality in modalities:
                    rows.append(
                        {
                            "modality": str(modality),
                            "treatment_id": str(treatment),
                            "dose": "fixed",
                            "round": str(round_id),
                            "batch": f"{round_id}_batch",
                            "replicate": str(replicate),
                            "split": str(split),
                            "uri": f"hf://altoslabs/scGeneScope/features/{modality}/{round_id}_{treatment}_rep{replicate}.h5ad",
                        }
                    )
    return validate_scgenescope_manifest(pd.DataFrame(rows))


def build_scgenescope_condition_pairs(
    manifest: pd.DataFrame,
    *,
    split: str | None = None,
    pair_columns: Iterable[str] = SCGENESCOPE_PAIR_COLUMNS,
) -> pd.DataFrame:
    frame = validate_scgenescope_manifest(manifest)
    if split is not None:
        frame = frame[frame["split"] == str(split)].copy()
    pair_columns = tuple(pair_columns)
    missing_pair_columns = [column for column in pair_columns if column not in frame.columns]
    if missing_pair_columns:
        raise ValueError(f"scGeneScope pair columns missing: {missing_pair_columns}")
    grouped = (
        frame.groupby(list(pair_columns) + ["split", "modality"], dropna=False)
        .size()
        .rename("record_count")
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=list(pair_columns) + ["split"],
        columns="modality",
        values="record_count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for modality in SCGENESCOPE_VALID_MODALITIES:
        if modality not in pivot.columns:
            pivot[modality] = 0
    pivot = pivot.rename(columns={"rna": "rna_record_count", "image": "image_record_count"})
    pivot["condition_pair_ready"] = (pivot["rna_record_count"] > 0) & (pivot["image_record_count"] > 0)
    pair_id = pivot[list(pair_columns) + ["split"]].astype(str).agg("::".join, axis=1)
    pivot.insert(0, "pair_id", pair_id)
    return pivot.sort_values(["split", "pair_id"]).reset_index(drop=True)


def audit_scgenescope_local_root(root: Path, *, manifest_name: str | None = None) -> ScGeneScopeAudit:
    root = Path(root)
    if not root.exists():
        return ScGeneScopeAudit(
            root=str(root),
            root_exists=False,
            manifest_path=None,
            manifest_valid=False,
            paired_condition_count=0,
            rna_record_count=0,
            image_record_count=0,
            split_count=0,
            message="Local scGeneScope root does not exist; no data was downloaded.",
        )
    manifest_path = _find_scgenescope_manifest(root, manifest_name=manifest_name)
    if manifest_path is None:
        return ScGeneScopeAudit(
            root=str(root),
            root_exists=True,
            manifest_path=None,
            manifest_valid=False,
            paired_condition_count=0,
            rna_record_count=0,
            image_record_count=0,
            split_count=0,
            message="Local root exists but no manifest file was found.",
        )
    try:
        manifest = load_scgenescope_manifest(manifest_path)
        pairs = build_scgenescope_condition_pairs(manifest)
    except Exception as exc:
        return ScGeneScopeAudit(
            root=str(root),
            root_exists=True,
            manifest_path=str(manifest_path),
            manifest_valid=False,
            paired_condition_count=0,
            rna_record_count=0,
            image_record_count=0,
            split_count=0,
            message=f"Manifest failed validation: {exc}",
        )
    return ScGeneScopeAudit(
        root=str(root),
        root_exists=True,
        manifest_path=str(manifest_path),
        manifest_valid=True,
        paired_condition_count=int(pairs["condition_pair_ready"].sum()),
        rna_record_count=int((manifest["modality"] == "rna").sum()),
        image_record_count=int((manifest["modality"] == "image").sum()),
        split_count=int(manifest["split"].nunique()),
        message="Manifest validates and condition-pair table was built.",
    )


def _find_scgenescope_manifest(root: Path, *, manifest_name: str | None = None) -> Path | None:
    if manifest_name:
        path = root / manifest_name
        return path if path.exists() else None
    for name in (
        "scgenescope_manifest.tsv",
        "scgenescope_manifest.csv",
        "manifest.tsv",
        "manifest.csv",
        "metadata.tsv",
        "metadata.csv",
    ):
        path = root / name
        if path.exists():
            return path
    return None


def fetch_huggingface_dataset_tree(
    repo_id: str = SCGENESCOPE_HF_REPO,
    *,
    path: str = "",
    revision: str = "main",
    recursive: bool = False,
    timeout_seconds: float = 30.0,
) -> list[dict[str, object]]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    clean_path = path.strip("/")
    suffix = f"/{urllib.parse.quote(clean_path, safe='/')}" if clean_path else ""
    query = "?recursive=1" if recursive else "?recursive=0"
    url = f"https://huggingface.co/api/datasets/{quoted_repo}/tree/{quoted_revision}{suffix}{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "perturb-jepa-bridge-metadata-audit"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f"Expected Hugging Face tree list for {path!r}, got {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def classify_scgenescope_remote_path(path: str, *, size: int | None = None, entry_type: str | None = None) -> dict[str, object]:
    normalized = path.strip("/")
    lower = normalized.lower()
    tokens = lower.split("/")
    modality = "unknown"
    if "/rnaseq/" in f"/{lower}/" or lower.endswith("/rnaseq") or "rnaseq" in tokens:
        modality = "rna"
    elif "/imaging/" in f"/{lower}/" or lower.endswith("/imaging") or "imaging" in tokens:
        modality = "image"
    representation = "unknown"
    if lower.startswith("features/"):
        representation = "feature"
    elif lower.startswith("measured/"):
        representation = "measured"
    extension = Path(normalized).suffix.lower()
    size_int = int(size or 0)
    is_file = entry_type == "file"
    is_light_metadata = bool(
        is_file
        and size_int <= 50_000_000
        and any(pattern in lower for pattern in SCGENESCOPE_LIGHT_METADATA_PATTERNS)
        and extension in {".csv", ".tsv", ".json", ".parquet", ".txt", ".yaml", ".yml"}
    )
    is_large_payload = bool(is_file and size_int > 1_000_000_000)
    is_feature_h5ad = bool(is_file and lower.startswith("features/") and extension == ".h5ad")
    round_id = "unknown"
    for token in tokens:
        if token.startswith("round_"):
            round_id = Path(token).stem
            break
    return {
        "path": normalized,
        "type": entry_type or "unknown",
        "size_bytes": size_int,
        "modality": modality,
        "representation": representation,
        "extension": extension,
        "round": round_id,
        "is_light_metadata_candidate": is_light_metadata,
        "is_large_payload": is_large_payload,
        "is_feature_h5ad": is_feature_h5ad,
    }


def summarize_scgenescope_remote_entries(entries: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = [
        classify_scgenescope_remote_path(
            str(item.get("path", "")),
            size=int(item.get("size", 0) or 0),
            entry_type=str(item.get("type", "unknown")),
        )
        for item in entries
        if str(item.get("path", ""))
    ]
    return pd.DataFrame(rows)


def scgenescope_feature_pair_candidates(remote_summary: pd.DataFrame) -> pd.DataFrame:
    """Pair RNA and image feature H5AD files by released round.

    This is metadata-only: callers pass a previously fetched Hugging Face tree
    summary and the function estimates candidate paired feature footprints
    without opening or downloading payload files.
    """

    columns = [
        "round",
        "rna_path",
        "rna_size_bytes",
        "image_path",
        "image_size_bytes",
        "paired_size_bytes",
    ]
    if remote_summary.empty:
        return pd.DataFrame(columns=columns)
    required = {"path", "modality", "round", "size_bytes", "is_feature_h5ad"}
    missing = sorted(required - set(remote_summary.columns))
    if missing:
        raise ValueError(f"scGeneScope remote summary missing columns: {missing}")
    frame = remote_summary.copy()
    feature_mask = frame["is_feature_h5ad"].astype(bool)
    feature_frame = frame[feature_mask & frame["modality"].isin(["rna", "image"])].copy()
    rna = (
        feature_frame[feature_frame["modality"] == "rna"][["round", "path", "size_bytes"]]
        .rename(columns={"path": "rna_path", "size_bytes": "rna_size_bytes"})
        .reset_index(drop=True)
    )
    image = (
        feature_frame[feature_frame["modality"] == "image"][["round", "path", "size_bytes"]]
        .rename(columns={"path": "image_path", "size_bytes": "image_size_bytes"})
        .reset_index(drop=True)
    )
    if rna.empty or image.empty:
        return pd.DataFrame(columns=columns)
    pairs = rna.merge(image, on="round", how="inner")
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    pairs["rna_size_bytes"] = pairs["rna_size_bytes"].astype("int64")
    pairs["image_size_bytes"] = pairs["image_size_bytes"].astype("int64")
    pairs["paired_size_bytes"] = pairs["rna_size_bytes"] + pairs["image_size_bytes"]
    return pairs[columns].sort_values(["paired_size_bytes", "round", "rna_path", "image_path"]).reset_index(drop=True)
