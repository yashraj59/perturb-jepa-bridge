from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.schema import normalize_image_manifest, normalize_scrna_obs, normalize_value


STANDARD_DOSE_BUCKETS_UM = (0.04, 0.12, 0.37, 1.11, 3.33, 10.0, 20.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare condition-level LINCS L1000 and LINCS Cell Painting profile inputs. "
            "Cell Painting profiles are stored as small .npy feature-grid arrays so the "
            "existing image manifest loader can train on processed morphology profiles."
        )
    )
    parser.add_argument("--l1000-gctx", type=Path, default=Path("data/raw/lincs_l1000/level_5_modz_n9482x978.gctx"))
    parser.add_argument(
        "--l1000-metadata",
        type=Path,
        default=Path("data/raw/lincs_l1000/col_meta_level_5_REP.A_A549_only_n9482.txt"),
    )
    parser.add_argument("--l1000-pert-info", type=Path, default=Path("data/raw/lincs_l1000/REP.A_A549_pert_info.txt"))
    parser.add_argument(
        "--cell-painting-release",
        type=Path,
        default=Path("data/raw/lincs_cell_painting/extracted/broadinstitute-lincs-cell-painting-74f7a0d"),
    )
    parser.add_argument("--cell-painting-profile-dir", type=Path, default=Path("data/raw/lincs_cell_painting/level4b"))
    parser.add_argument("--dose-um", type=float, default=10.0, help="Use one dose bucket; pass <=0 to keep all nonzero doses.")
    parser.add_argument("--include-controls", action="store_true", help="Include DMSO controls as condition dmso|0uM.")
    parser.add_argument("--max-conditions", type=int, default=None, help="Optional deterministic cap for small smoke runs.")
    parser.add_argument("--image-size", type=int, default=36, help="Square feature-grid side length for Cell Painting profiles.")
    parser.add_argument("--download-cell-painting", action="store_true", help="Download missing Level4b profile files.")
    parser.add_argument("--min-shared-conditions", type=int, default=6)
    parser.add_argument("--rna-output", type=Path, default=Path("data/processed/lincs/l1000_a549_10uM_24h.h5ad"))
    parser.add_argument(
        "--image-output",
        type=Path,
        default=Path("data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv"),
    )
    parser.add_argument("--image-array-dir", type=Path, default=Path("data/processed/lincs/cell_painting_profile_arrays"))
    parser.add_argument("--summary-output", type=Path, default=Path("metrics/lincs/compound_intersection.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("metrics/lincs/prepare_lincs_summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require(args.l1000_gctx, "L1000 GCTX")
    _require(args.l1000_metadata, "L1000 metadata")
    _require(args.l1000_pert_info, "L1000 perturbation info")
    _require(args.cell_painting_release, "LINCS Cell Painting release")

    manifest_path = args.cell_painting_release / "metadata/clue_manifest/cell_painting_lincs_pilot_1_manifest.txt"
    _require(manifest_path, "LINCS Cell Painting clue manifest")
    cp_manifest = pd.read_csv(manifest_path, sep="\t")
    level4b = cp_manifest.loc[cp_manifest["data_level"].eq("Level4b")].copy()
    if level4b.empty:
        raise SystemExit("No Level4b Cell Painting profile files were listed in the release manifest")
    if args.download_cell_painting:
        _download_level4b_profiles(level4b, args.cell_painting_profile_dir)
    profile_paths = sorted(args.cell_painting_profile_dir.glob("*_normalized_feature_select.csv.gz"))
    if not profile_paths:
        raise SystemExit(
            f"No downloaded Cell Painting Level4b profiles found in {args.cell_painting_profile_dir}; "
            "rerun with --download-cell-painting"
        )

    l1000_matrix, gene_ids, l1000_metadata = _load_l1000(args.l1000_gctx, args.l1000_metadata)
    pert_info = _load_pert_info(args.l1000_pert_info)
    l1000_metadata = _standardize_l1000_metadata(l1000_metadata, pert_info, include_controls=args.include_controls)

    cp_metadata, cp_features, feature_names = _load_cell_painting_profiles(profile_paths, include_controls=args.include_controls)
    cp_metadata = _standardize_cell_painting_metadata(cp_metadata, pert_info, include_controls=args.include_controls)

    if args.dose_um > 0:
        dose = _format_dose(args.dose_um)
        l1000_mask = l1000_metadata["dose"].eq(dose)
        cp_mask = cp_metadata["dose"].eq(dose)
        if args.include_controls:
            l1000_mask = l1000_mask | l1000_metadata["perturbation"].eq("dmso")
            cp_mask = cp_mask | cp_metadata["perturbation"].eq("dmso")
        l1000_metadata = l1000_metadata.loc[l1000_mask].copy()
        l1000_matrix = l1000_matrix[l1000_metadata["_matrix_index"].to_numpy()]
        l1000_metadata["_matrix_index"] = np.arange(len(l1000_metadata))
        cp_features = cp_features[cp_mask.to_numpy()]
        cp_metadata = cp_metadata.loc[cp_mask].reset_index(drop=True)

    shared_conditions = sorted(set(l1000_metadata["condition_key_lincs"]) & set(cp_metadata["condition_key_lincs"]))
    if args.max_conditions is not None and args.max_conditions > 0:
        shared_conditions = shared_conditions[: int(args.max_conditions)]
    if len(shared_conditions) < args.min_shared_conditions:
        raise SystemExit(
            f"Only {len(shared_conditions)} shared LINCS conditions remain; "
            f"minimum required is {args.min_shared_conditions}"
        )

    l1000_keep = l1000_metadata["condition_key_lincs"].isin(shared_conditions)
    cp_keep = cp_metadata["condition_key_lincs"].isin(shared_conditions)
    l1000_metadata = l1000_metadata.loc[l1000_keep].copy()
    l1000_matrix = l1000_matrix[l1000_metadata["_matrix_index"].to_numpy()]
    cp_metadata = cp_metadata.loc[cp_keep].reset_index(drop=True)
    cp_features = cp_features[cp_keep.to_numpy()]

    rna_output = _write_l1000_h5ad(l1000_matrix, gene_ids, l1000_metadata, args.rna_output)
    image_manifest = _write_cell_painting_arrays(
        cp_metadata,
        cp_features,
        feature_names,
        array_dir=args.image_array_dir,
        image_size=args.image_size,
    )
    args.image_output.parent.mkdir(parents=True, exist_ok=True)
    image_manifest.to_csv(args.image_output, index=False)

    summary = _summary_table(l1000_metadata, image_manifest)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)

    payload = {
        "status": "prepared",
        "condition_key": "condition_key_lincs",
        "condition_key_definition": "perturbation|dose",
        "l1000_time": sorted(l1000_metadata["time"].unique().tolist()),
        "cell_painting_time": sorted(image_manifest["time"].unique().tolist()),
        "time_mismatch_limitation": "L1000 A549 profiles are 24h; LINCS Cell Painting pilot profiles are 48h.",
        "n_shared_conditions": len(shared_conditions),
        "n_shared_perturbations": int(summary["perturbation"].nunique()),
        "n_l1000_profiles": int(len(l1000_metadata)),
        "n_cell_painting_profiles": int(len(image_manifest)),
        "n_l1000_genes": int(l1000_matrix.shape[1]),
        "n_cell_painting_features": int(len(feature_names)),
        "feature_grid_shape": [1, int(args.image_size), int(args.image_size)],
        "rna_output": str(rna_output),
        "image_output": str(args.image_output),
        "summary_output": str(args.summary_output),
        "source_files": {
            "l1000_gctx": str(args.l1000_gctx),
            "l1000_metadata": str(args.l1000_metadata),
            "l1000_pert_info": str(args.l1000_pert_info),
            "cell_painting_manifest": str(manifest_path),
            "cell_painting_profile_dir": str(args.cell_painting_profile_dir),
        },
        "sha256": {
            "l1000_gctx": _sha256(args.l1000_gctx),
            "l1000_metadata": _sha256(args.l1000_metadata),
            "l1000_pert_info": _sha256(args.l1000_pert_info),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(summary.head(30).to_string(index=False))
    print(f"Shared LINCS conditions: {len(shared_conditions)}")
    print(f"Shared perturbations: {summary['perturbation'].nunique()}")
    print(f"Wrote L1000 AnnData: {rna_output}")
    print(f"Wrote Cell Painting profile manifest: {args.image_output}")
    print(f"Wrote summary: {args.summary_output}")
    return 0


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def _download_level4b_profiles(level4b: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in level4b.itertuples(index=False):
        url = str(row.file_name)
        target = output_dir / url.rsplit("/", 1)[-1]
        expected_size = int(row.size)
        if target.exists() and target.stat().st_size == expected_size:
            continue
        print(f"Downloading {target.name}")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            temp = target.with_suffix(target.suffix + ".tmp")
            with temp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temp.replace(target)


def _load_l1000(gctx_path: Path, metadata_path: Path) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    with h5py.File(gctx_path, "r") as handle:
        matrix = np.asarray(handle["0/DATA/0/matrix"], dtype=np.float32)
        sig_ids = [_decode(value) for value in handle["0/META/COL/id"][:]]
        gene_ids = [_decode(value) for value in handle["0/META/ROW/id"][:]]
    metadata = pd.read_csv(metadata_path, sep="\t")
    if "sig_id" not in metadata.columns:
        raise SystemExit("L1000 metadata is missing sig_id")
    order = pd.DataFrame({"sig_id": sig_ids, "_gctx_index": np.arange(len(sig_ids))})
    metadata = order.merge(metadata, on="sig_id", how="left", validate="one_to_one")
    if metadata["pert_id"].isna().any():
        missing = int(metadata["pert_id"].isna().sum())
        raise SystemExit(f"L1000 metadata failed to align for {missing} GCTX columns")
    matrix = matrix[metadata["_gctx_index"].to_numpy()]
    metadata = metadata.drop(columns=["_gctx_index"]).copy()
    return matrix, gene_ids, metadata


def _load_pert_info(path: Path) -> pd.DataFrame:
    info = pd.read_csv(path, sep="\t")
    for column in ("pert_id", "pert_iname", "moa"):
        if column not in info.columns:
            raise SystemExit(f"L1000 perturbation info is missing {column!r}")
    if "target" not in info.columns:
        info["target"] = ""
    return info.drop_duplicates("pert_id")


def _standardize_l1000_metadata(metadata: pd.DataFrame, pert_info: pd.DataFrame, *, include_controls: bool) -> pd.DataFrame:
    frame = metadata.copy()
    is_control = frame["pert_type"].astype(str).eq("ctl_vehicle")
    is_compound = frame["pert_type"].astype(str).eq("trt_cp")
    frame = frame.loc[is_compound | (include_controls & is_control)].copy()
    frame["_is_control"] = frame["pert_type"].astype(str).eq("ctl_vehicle")
    frame = frame.merge(pert_info[["pert_id", "pert_iname", "moa", "target"]], on="pert_id", how="left", suffixes=("", "_info"))
    is_control = frame["_is_control"]
    frame["perturbation"] = np.where(is_control, "dmso", frame["pert_id"].astype(str))
    frame["perturbation_type"] = np.where(is_control, "control", "compound")
    frame["dose"] = [
        "0uM" if control else _format_dose(value)
        for control, value in zip(is_control.tolist(), frame["nearest_dose"].tolist(), strict=False)
    ]
    frame["time"] = frame["pert_time"].map(lambda value: f"{_number(value):g}h")
    frame["cell_line"] = frame["cell_id"].map(normalize_value)
    frame["batch"] = frame["batch"].map(normalize_value)
    frame["compound"] = np.where(is_control, "DMSO", frame["pert_iname"].fillna(frame["pert_id"]).astype(str))
    moa_values = frame["moa_info"] if "moa_info" in frame.columns else frame.get("moa", "")
    target_values = frame["target_info"] if "target_info" in frame.columns else frame.get("target", "")
    frame["moa"] = np.where(is_control, "control", pd.Series(moa_values, index=frame.index).fillna("").astype(str))
    frame["target_gene"] = np.where(is_control, "", pd.Series(target_values, index=frame.index).fillna("").astype(str))
    frame["condition_key_lincs"] = frame["perturbation"].astype(str) + "|" + frame["dose"].astype(str)
    frame["_matrix_index"] = np.arange(len(frame), dtype=np.int64)
    normalized = normalize_scrna_obs(frame)
    normalized["condition_key_lincs"] = frame["condition_key_lincs"].values
    normalized["condition_key"] = normalized["condition_key_lincs"]
    return normalized


def _load_cell_painting_profiles(
    paths: Iterable[Path],
    *,
    include_controls: bool,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    tables: list[pd.DataFrame] = []
    feature_union: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path)
        if "Metadata_pert_type" not in frame.columns:
            raise SystemExit(f"Cell Painting profile file is missing Metadata_pert_type: {path}")
        keep = frame["Metadata_pert_type"].astype(str).eq("trt")
        if include_controls:
            keep = keep | frame["Metadata_pert_type"].astype(str).eq("control")
        frame = frame.loc[keep].copy()
        frame["_source_file"] = path.name
        feature_union.update(column for column in frame.columns if not column.startswith("Metadata_") and not column.startswith("_"))
        tables.append(frame)
    if not tables:
        raise SystemExit("No Cell Painting profile tables were loaded")
    feature_names = sorted(feature_union)
    metadata_columns = sorted({column for table in tables for column in table.columns if column.startswith("Metadata_")})
    selected_frames = []
    feature_frames = []
    for table in tables:
        selected_frames.append(table[[*metadata_columns, "_source_file"]].copy())
        feature_frames.append(table.reindex(columns=feature_names, fill_value=0.0))
    metadata = pd.concat(selected_frames, ignore_index=True, sort=False)
    features = pd.concat(feature_frames, ignore_index=True, sort=False).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return metadata, features.to_numpy(dtype=np.float32), feature_names


def _standardize_cell_painting_metadata(
    metadata: pd.DataFrame,
    pert_info: pd.DataFrame,
    *,
    include_controls: bool,
) -> pd.DataFrame:
    frame = metadata.copy()
    is_control = frame["Metadata_pert_type"].astype(str).eq("control")
    if not include_controls:
        frame = frame.loc[~is_control].copy()
        is_control = frame["Metadata_pert_type"].astype(str).eq("control")
    frame["pert_id"] = np.where(is_control, "dmso", frame["Metadata_broad_id"].astype(str))
    frame = frame.merge(pert_info[["pert_id", "pert_iname", "moa", "target"]], on="pert_id", how="left", suffixes=("", "_info"))
    dose_values = [_nearest_dose_bucket(value) for value in frame["Metadata_mmoles_per_liter"].tolist()]
    frame["perturbation"] = np.where(is_control, "dmso", frame["pert_id"].astype(str))
    frame["perturbation_type"] = np.where(is_control, "control", "compound")
    frame["dose"] = ["0uM" if control else _format_dose(value) for control, value in zip(is_control.tolist(), dose_values, strict=False)]
    frame["time"] = "48h"
    frame["cell_line"] = frame.get("Metadata_cell_id", "A549")
    frame["batch"] = frame.get("Metadata_Plate", frame.get("Metadata_Assay_Plate_Barcode", "unknown")).map(normalize_value)
    frame["plate"] = frame.get("Metadata_Plate", frame.get("Metadata_Assay_Plate_Barcode", "unknown")).map(normalize_value)
    frame["well"] = frame.get("Metadata_Well", "unknown").map(normalize_value)
    frame["site"] = "profile"
    frame["channel_or_z"] = "CP_Level4b_profile"
    frame["compound"] = np.where(is_control, "DMSO", frame["pert_iname"].fillna(frame["pert_id"]).astype(str))
    moa_values = frame["moa_info"] if "moa_info" in frame.columns else frame.get("moa", frame.get("Metadata_moa", ""))
    target_values = frame["target_info"] if "target_info" in frame.columns else frame.get("target", frame.get("Metadata_target", ""))
    frame["moa"] = np.where(
        is_control,
        "control",
        pd.Series(moa_values, index=frame.index).fillna(frame.get("Metadata_moa", "")).fillna("").astype(str),
    )
    frame["target_gene"] = np.where(
        is_control,
        "",
        pd.Series(target_values, index=frame.index).fillna(frame.get("Metadata_target", "")).fillna("").astype(str),
    )
    frame["condition_key_lincs"] = frame["perturbation"].astype(str) + "|" + frame["dose"].astype(str)
    return frame.reset_index(drop=True)


def _write_l1000_h5ad(matrix: np.ndarray, gene_ids: list[str], metadata: pd.DataFrame, output: Path) -> Path:
    import anndata as ad

    output.parent.mkdir(parents=True, exist_ok=True)
    obs = metadata.drop(columns=["_matrix_index"], errors="ignore").copy()
    var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_ids}, index=gene_ids)
    adata = ad.AnnData(X=matrix.astype(np.float32, copy=False), obs=obs, var=var)
    adata.write_h5ad(output)
    return output


def _write_cell_painting_arrays(
    metadata: pd.DataFrame,
    features: np.ndarray,
    feature_names: list[str],
    *,
    array_dir: Path,
    image_size: int,
) -> pd.DataFrame:
    array_dir.mkdir(parents=True, exist_ok=True)
    grid_features = image_size * image_size
    if len(feature_names) > grid_features:
        raise SystemExit(
            f"image-size {image_size} is too small for {len(feature_names)} Cell Painting features; "
            f"need at least {math.ceil(math.sqrt(len(feature_names)))}"
        )
    padded = np.zeros((features.shape[0], grid_features), dtype=np.float32)
    padded[:, : features.shape[1]] = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        condition = str(row["condition_key_lincs"]).replace("/", "_")
        plate = normalize_value(row.get("plate", "unknown"))
        well = normalize_value(row.get("well", "unknown"))
        path = array_dir / f"{index:06d}_{plate}_{well}_{condition}.npy"
        array = padded[index].reshape(1, image_size, image_size)
        np.save(path, array)
        rows.append(
            {
                "image_path": str(path),
                "plate": plate,
                "well": well,
                "site": "profile",
                "channel_or_z": "CP_Level4b_profile",
                "perturbation": row["perturbation"],
                "compound": row["compound"],
                "moa": row["moa"],
                "target_gene": row["target_gene"],
                "dose": row["dose"],
                "time": row["time"],
                "cell_line": row["cell_line"],
                "batch": row["batch"],
                "perturbation_type": row["perturbation_type"],
                "condition_key_lincs": row["condition_key_lincs"],
            }
        )
    manifest = normalize_image_manifest(pd.DataFrame(rows))
    manifest["condition_key_lincs"] = [row["condition_key_lincs"] for row in rows]
    manifest["condition_key"] = manifest["condition_key_lincs"]
    feature_path = array_dir.parent / "cell_painting_feature_order.txt"
    feature_path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    return manifest


def _summary_table(rna: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    rna_counts = rna.groupby(["perturbation", "compound", "dose"], dropna=False).size().rename("n_l1000_profiles")
    image_counts = image.groupby(["perturbation", "compound", "dose"], dropna=False).size().rename("n_cell_painting_profiles")
    moa = image.groupby(["perturbation", "compound", "dose"], dropna=False)["moa"].agg(
        lambda values: ";".join(sorted(set(map(str, values))))
    )
    summary = pd.concat([rna_counts, image_counts, moa.rename("moa")], axis=1).reset_index()
    summary = summary.fillna({"n_l1000_profiles": 0, "n_cell_painting_profiles": 0, "moa": ""})
    return summary.sort_values(["perturbation", "dose"]).reset_index(drop=True)


def _decode(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _number(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def _nearest_dose_bucket(value: object) -> float:
    number = _number(value)
    if number <= 0:
        return 0.0
    return min(STANDARD_DOSE_BUCKETS_UM, key=lambda bucket: abs(bucket - number))


def _format_dose(value: object) -> str:
    number = _nearest_dose_bucket(value)
    return f"{number:g}uM"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
