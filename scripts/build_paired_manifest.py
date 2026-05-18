from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BIOLOGICAL_KEYS = ["perturbation", "dose", "time", "cell_line"]
TECHNICAL_EXCLUSIONS = ["batch", "plate", "well", "site", "run", "sequencing_lane", "library_id"]
PAIRING_KEY_TYPES = ("auto", "cell", "spot", "tile", "well", "sample", "condition")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a condition-level paired RNA/image manifest.")
    parser.add_argument("--rna-metadata", required=True, type=Path, help="RNA metadata CSV from build_rna_manifest.py.")
    parser.add_argument("--image-manifest", required=True, type=Path, help="Normalized image manifest CSV.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Output paired manifest CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for copied normalized manifests.")
    parser.add_argument("--output-rna-metadata", type=Path, default=None, help="Copied/normalized RNA metadata CSV.")
    parser.add_argument("--output-image-manifest", type=Path, default=None, help="Copied/normalized image manifest CSV.")
    parser.add_argument("--pairing-table", type=Path, default=None, help="Detailed pairing table CSV.")
    parser.add_argument("--pairing-key-type", choices=PAIRING_KEY_TYPES, default="auto")
    parser.add_argument("--pairing-tier", default=None, help="Override inferred pairing tier label.")
    parser.add_argument("--qc-json", type=Path, default=None, help="QC JSON path. Defaults next to output manifest.")
    parser.add_argument("--qc-csv", type=Path, default=None, help="Pairing-tier QC CSV path. Defaults next to output manifest.")
    return parser.parse_args(argv)


def _load_runtime_deps() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to build paired manifests: pip install -e .") from exc

    from perturb_jepa.data.schema import add_hierarchical_condition_keys, normalize_value

    return pd, add_hierarchical_condition_keys, normalize_value


def _read_csv(pd: Any, path: Path, name: str) -> Any:
    if not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    return pd.read_csv(path)


def _normalize_metadata(frame: Any, add_hierarchical_condition_keys: Any, normalize_value: Any, name: str) -> Any:
    missing = [column for column in BIOLOGICAL_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required biological columns: {missing}")
    normalized = frame.copy()
    for column in BIOLOGICAL_KEYS:
        normalized[column] = normalized[column].map(normalize_value)
    if "perturbation_type" not in normalized.columns:
        normalized["perturbation_type"] = "unknown"
    if "condition_key" not in normalized.columns:
        normalized = add_hierarchical_condition_keys(normalized)
    else:
        expected = normalized.apply(lambda row: "|".join(str(row[column]) for column in BIOLOGICAL_KEYS), axis=1)
        bad = normalized["condition_key"].astype(str) != expected.astype(str)
        if bool(bad.any()):
            raise ValueError(
                f"{name} has condition_key values that do not equal perturbation|dose|time|cell_line; "
                f"first bad row index: {normalized.index[bad][0]}"
            )
        normalized = add_hierarchical_condition_keys(normalized)
    return normalized


def _with_pairing_aliases(frame: Any) -> Any:
    aliased = frame.copy()
    aliases = {
        "plate_id": "plate",
        "well_id": "well",
        "field_id": "site",
        "cell_id": "obs_index",
    }
    for target, source in aliases.items():
        if target not in aliased.columns and source in aliased.columns:
            aliased[target] = aliased[source]
    return aliased


def _condition_counts(frame: Any, count_name: str) -> Any:
    return (
        frame.groupby(["condition_key", "perturbation", "dose", "time", "cell_line"], sort=True, dropna=False)
        .size()
        .reset_index(name=count_name)
    )


def _infer_pairing_key_type(rna: Any, image: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if _has_pairing_columns(rna, image, ["sample_id", "cell_id"]):
        return "cell"
    if _has_pairing_columns(rna, image, ["sample_id", "spot_id"]):
        return "spot"
    if _has_pairing_columns(rna, image, ["sample_id", "tile_id"]):
        return "tile"
    if _has_pairing_columns(rna, image, ["plate_id", "well_id"]):
        return "well"
    if _has_pairing_columns(rna, image, ["sample_id"]):
        return "sample"
    return "condition"


def _has_pairing_columns(rna: Any, image: Any, columns: list[str]) -> bool:
    return all(column in rna.columns and column in image.columns for column in columns)


def _pairing_columns(pairing_key_type: str) -> list[str]:
    if pairing_key_type == "cell":
        return ["sample_id", "cell_id"]
    if pairing_key_type == "spot":
        return ["sample_id", "spot_id"]
    if pairing_key_type == "tile":
        return ["sample_id", "tile_id"]
    if pairing_key_type == "well":
        return ["plate_id", "well_id"]
    if pairing_key_type == "sample":
        return ["sample_id"]
    if pairing_key_type == "condition":
        return ["condition_key"]
    raise ValueError(f"unsupported pairing key type: {pairing_key_type}")


def _default_tier(pairing_key_type: str) -> str:
    if pairing_key_type in {"cell", "spot", "tile"}:
        return "tier1_true_paired_image_expression"
    if pairing_key_type in {"well", "sample", "condition"}:
        return "tier2_weakly_paired_sample_well_condition"
    return "not_verified"


def _add_pairing_id(frame: Any, columns: list[str]) -> Any:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"requested pairing columns are missing: {missing}")
    paired = frame.copy()
    paired["pairing_id"] = paired.apply(lambda row: "|".join(str(row[column]) for column in columns), axis=1)
    return paired


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_paired_manifest(args: argparse.Namespace) -> dict[str, Any]:
    pd, add_hierarchical_condition_keys, normalize_value = _load_runtime_deps()
    qc_json = args.qc_json or args.output_manifest.with_name(args.output_manifest.stem + "_qc_summary.json")
    qc_csv = args.qc_csv or args.output_manifest.with_name(args.output_manifest.stem + "_pairing_qc.csv")
    requested_key_type = getattr(args, "pairing_key_type", "auto")
    requested_tier = getattr(args, "pairing_tier", None)

    rna = _with_pairing_aliases(_normalize_metadata(
        _read_csv(pd, args.rna_metadata, "RNA metadata"),
        add_hierarchical_condition_keys,
        normalize_value,
        "RNA metadata",
    ))
    image = _with_pairing_aliases(_normalize_metadata(
        _read_csv(pd, args.image_manifest, "image manifest"),
        add_hierarchical_condition_keys,
        normalize_value,
        "image manifest",
    ))
    pairing_key_type = _infer_pairing_key_type(rna, image, requested_key_type)
    pairing_columns = _pairing_columns(pairing_key_type)
    pairing_tier = requested_tier or _default_tier(pairing_key_type)
    rna = _add_pairing_id(rna, pairing_columns)
    image = _add_pairing_id(image, pairing_columns)
    rna_counts = _condition_counts(rna, "n_rna_cells")
    image_counts = _condition_counts(image, "n_images")
    paired = rna_counts.merge(
        image_counts,
        on=["condition_key", "perturbation", "dose", "time", "cell_line"],
        how="inner",
        validate="one_to_one",
    )
    pairing_table = _build_pairing_table(pd, rna, image, pairing_tier=pairing_tier, pairing_key_type=pairing_key_type)
    paired["pairing_key"] = paired["condition_key"]
    paired["pairing_key_perturbation"] = paired["perturbation"]
    paired["pairing_key_dose_time"] = paired["perturbation"] + "|" + paired["dose"] + "|" + paired["time"]
    paired["pairing_tier"] = pairing_tier
    paired = paired[
        [
            "pairing_tier",
            "pairing_key",
            "condition_key",
            "pairing_key_perturbation",
            "pairing_key_dose_time",
            "perturbation",
            "dose",
            "time",
            "cell_line",
            "n_rna_cells",
            "n_images",
        ]
    ].sort_values("condition_key")

    if paired.empty:
        raise ValueError("RNA metadata and image manifest have no overlapping perturbation|dose|time|cell_line conditions")

    output_dir = getattr(args, "output_dir", None)
    pairing_table_path = getattr(args, "pairing_table", None) or args.output_manifest.with_name("pairing_table.csv")
    output_rna = getattr(args, "output_rna_metadata", None) or (
        output_dir / "rna_metadata.csv" if output_dir else args.output_manifest.with_name("rna_metadata.csv")
    )
    output_image = getattr(args, "output_image_manifest", None) or (
        output_dir / "image_manifest.csv" if output_dir else args.output_manifest.with_name("image_manifest.csv")
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pairing_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_rna.parent.mkdir(parents=True, exist_ok=True)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_manifest, index=False)
    pairing_table.to_csv(pairing_table_path, index=False)
    rna.to_csv(output_rna, index=False)
    image.to_csv(output_image, index=False)
    tier_qc = paired.groupby("pairing_tier", sort=True).agg(
        n_pairs=("condition_key", "size"),
        n_rna_cells=("n_rna_cells", "sum"),
        n_images=("n_images", "sum"),
    )
    tier_qc.reset_index().to_csv(qc_csv, index=False)

    rna_conditions = set(rna_counts["condition_key"].astype(str))
    image_conditions = set(image_counts["condition_key"].astype(str))
    qc = {
        "rna_metadata": str(args.rna_metadata),
        "image_manifest": str(args.image_manifest),
        "output_manifest": str(args.output_manifest),
        "pairing_table": str(pairing_table_path),
        "normalized_rna_metadata": str(output_rna),
        "normalized_image_manifest": str(output_image),
        "n_paired_conditions": int(len(paired)),
        "n_pairing_rows": int(len(pairing_table)),
        "n_rna_conditions": int(len(rna_conditions)),
        "n_image_conditions": int(len(image_conditions)),
        "n_rna_only_conditions": int(len(rna_conditions - image_conditions)),
        "n_image_only_conditions": int(len(image_conditions - rna_conditions)),
        "condition_key_columns": BIOLOGICAL_KEYS,
        "excluded_from_condition_key": TECHNICAL_EXCLUSIONS,
        "pairing_key_type": pairing_key_type,
        "pairing_key_columns": pairing_columns,
        "pairing_tiers": sorted(paired["pairing_tier"].unique().tolist()),
        "pairing_keys": ["pairing_key", "pairing_key_perturbation", "pairing_key_dose_time"],
    }
    _write_json(qc_json, qc)
    return qc


def _build_pairing_table(pd: Any, rna: Any, image: Any, *, pairing_tier: str, pairing_key_type: str) -> Any:
    rna_grouped = rna.groupby("pairing_id", sort=True, dropna=False)
    image_grouped = image.groupby("pairing_id", sort=True, dropna=False)
    shared_ids = sorted(set(rna_grouped.groups).intersection(set(image_grouped.groups)))
    rows = []
    for pairing_id in shared_ids:
        rna_group = rna_grouped.get_group(pairing_id)
        image_group = image_grouped.get_group(pairing_id)
        first = rna_group.iloc[0]
        image_first = image_group.iloc[0]
        rows.append(
            {
                "pairing_id": pairing_id,
                "pairing_tier": pairing_tier,
                "pairing_key_type": pairing_key_type,
                "condition_key": first.get("condition_key", image_first.get("condition_key", "")),
                "rna_obs_ids": json.dumps(rna_group.get("obs_index", rna_group.index.to_series()).astype(str).tolist()),
                "rna_file": "",
                "image_paths": json.dumps(image_group.get("image_path", image_group.index.to_series()).astype(str).tolist()),
                "sample_id": first.get("sample_id", image_first.get("sample_id", "")),
                "well_id": first.get("well_id", image_first.get("well_id", "")),
                "spot_id": first.get("spot_id", image_first.get("spot_id", "")),
                "tile_id": first.get("tile_id", image_first.get("tile_id", "")),
                "plate_id": first.get("plate_id", image_first.get("plate_id", "")),
                "cell_line": first.get("cell_line", image_first.get("cell_line", "")),
                "perturbation": first.get("perturbation", image_first.get("perturbation", "")),
                "dose": first.get("dose", image_first.get("dose", "")),
                "time": first.get("time", image_first.get("time", "")),
                "batch": first.get("batch", image_first.get("batch", "")),
                "notes": _pairing_note(pairing_key_type),
            }
        )
    return pd.DataFrame.from_records(rows)


def _pairing_note(pairing_key_type: str) -> str:
    if pairing_key_type in {"cell", "spot", "tile"}:
        return "explicit same-cell/spot/tile key present; eligible for true paired evaluation if source metadata verifies it"
    if pairing_key_type in {"well", "sample"}:
        return "weak sample/well-level pairing; do not claim same-cell image-expression pairing"
    return "condition-level overlap only; use as weak condition-bag pairing, not true paired data"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        qc = build_paired_manifest(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote paired manifest with {qc['n_paired_conditions']:,} shared biological conditions.")


if __name__ == "__main__":
    main()
