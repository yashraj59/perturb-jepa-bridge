from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.program_bootstrap_jepa import ProgramBootstrapJEPAConfig
from perturb_jepa.training.biospectral_operator import (
    LatentOperatorBundle,
    bundle_features,
    bundle_transition_metrics,
    fit_ridge_numpy,
    predict_ridge_numpy,
)
from scripts.run_bioguard_wm_total_autonomy import (
    _evaluate_delta_calibrated_source_delta_rank_jepa,
    _train_source_delta_rank_program_jepa,
)


DEFAULT_RAW_DIR = Path("data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F097_cpg0003_rosetta_external_confirmation"
)
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "F097_CPG0003_ROSETTA_EXTERNAL_CONFIRMATION.md"
CP_RELATIVE_PATH = Path("CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz")
L1K_RELATIVE_PATH = Path("L1000/replicate_level_l1k.csv.gz")
CP_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/"
    "cpg0003-rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/"
    "CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz"
)
L1K_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/"
    "cpg0003-rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/"
    "L1000/replicate_level_l1k.csv.gz"
)
EXPECTED_BYTES = {
    str(CP_RELATIVE_PATH): 112_143_618,
    str(L1K_RELATIVE_PATH): 24_681_738,
}
SMILES_HASH_DIM = 768
CHEM_SCALAR_COLUMNS = (
    "dose_um",
    "log10_dose_um",
    "smiles_length",
    "atom_C",
    "atom_N",
    "atom_O",
    "atom_S",
    "atom_P",
    "atom_F",
    "atom_Cl",
    "atom_Br",
    "atom_I",
    "ring_digit_count",
    "branch_count",
    "double_bond_count",
    "triple_bond_count",
    "aromatic_char_count",
    "charge_marker_count",
    "is_control_descriptor",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fresh cpg0003 Rosetta external confirmation for the frozen F096/F082 path."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="*", type=int, default=[37, 38, 39])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--download-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-shared-noncontrol", type=int, default=300)
    parser.add_argument("--pca-dim", type=int, default=24)
    parser.add_argument("--hash-dim", type=int, default=SMILES_HASH_DIM)
    parser.add_argument("--split-mode", choices=["compound_holdout", "same_condition_replicate"], default="compound_holdout")
    parser.add_argument("--experiment-id", default="F097")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    env = _environment_summary()
    download_status = _ensure_files(args.raw_dir, download_missing=args.download_missing)
    paired = _build_condition_pairs(args.raw_dir, output_dir=args.output_dir, split_mode=args.split_mode)
    tables = _build_tables(
        paired,
        output_dir=args.output_dir,
        min_shared_noncontrol=args.min_shared_noncontrol,
        pca_dim=args.pca_dim,
        hash_dim=args.hash_dim,
    )
    preflight = {
        "environment": env,
        "download_status": download_status,
        "pairing_summary": tables["pairing_summary"],
        "descriptor_summary": tables["descriptor_summary"],
        "control_summary": tables["control_summary"],
        "preflight_pass": bool(tables["preflight_pass"]),
        "preflight_reason": tables.get("reason", ""),
        "split_mode": args.split_mode,
        "experiment_id": args.experiment_id,
        "raw_files_not_for_git": True,
        "validator_caveat": "cpg0003 Rosetta is L1000 expression plus Cell Painting morphology, not scRNA.",
    }
    _write_json(args.output_dir / "preflight_summary.json", preflight)
    if not tables["preflight_pass"]:
        payload = {
            "decision": "FAIL_CPG0003_PREFLIGHT_NON_PROMOTING",
            "preflight": preflight,
            "model_promoted": False,
        }
        _write_json(args.output_dir / "metrics_eval.json", payload)
        _write_report(args.report_path, payload)
        return 1

    per_seed_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        model, trace = _train_source_delta_rank_program_jepa(
            tables["train_table"],
            tables["config"],
            seed=int(seed) + 9700,
            device=args.device,
            output_dir=args.output_dir / f"seed_{seed}",
            steps=args.steps,
        )
        trace_rows.append({"seed": seed, **trace})
        for split, eval_table in tables["eval_tables"].items():
            eval_payload = _evaluate_delta_calibrated_source_delta_rank_jepa(
                model,
                tables["train_table"],
                eval_table,
                device=args.device,
            )
            per_seed_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": f"{args.experiment_id}_frozen_smiles_delta_calibrated",
                    **eval_payload,
                }
            )
            baseline_rows.extend(_baseline_rows(tables["train_table"], eval_table, seed=seed, split=split))
            baseline_rows.append(_raw_uncalibrated_row(model, eval_table, seed=seed, split=split, device=args.device))

    per_seed = pd.DataFrame(per_seed_rows)
    baselines = pd.DataFrame(baseline_rows)
    trace = pd.DataFrame(trace_rows)
    summary = _summary_metrics(per_seed, baselines)
    file_prefix = args.experiment_id.lower()
    per_seed.to_csv(args.output_dir / f"{file_prefix}_seed_split_metrics.tsv", sep="\t", index=False)
    baselines.to_csv(args.output_dir / f"{file_prefix}_baseline_metrics.tsv", sep="\t", index=False)
    trace.to_csv(args.output_dir / f"{file_prefix}_train_trace.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / f"{file_prefix}_summary_metrics.tsv", sep="\t", index=False)

    decision = _decision(summary, preflight, candidate_method=f"{args.experiment_id}_frozen_smiles_delta_calibrated")
    payload = {
        "decision": decision,
        "candidate_method": f"{args.experiment_id}_frozen_smiles_delta_calibrated",
        "preflight": preflight,
        "per_seed_split_metrics": per_seed.to_dict("records"),
        "baseline_metrics": baselines.to_dict("records"),
        "summary_metrics": summary.to_dict("records"),
        "train_trace": trace.to_dict("records"),
        "model_promoted": False,
        "leakage_controls": (
            f"{args.experiment_id} uses compound IDs only for pairing/grouping/retrieval labels. "
            "Model action tensors are non-exact SMILES hash descriptors, simple chemistry counts, and dose. "
            "PCA, scaling, full-ridge floor, and delta calibration are fit on train split only. "
            "No condition_key, biological_key, exact target one-hot, held-out target means, pooled train+test "
            "statistics, or floor predictions are candidate model inputs."
        ),
        "promotion_caveat": (
            "No model is promoted. cpg0003 Rosetta is fresh external L1000 plus Cell Painting validation, "
            "not strict scRNA plus imaging validation."
        ),
    }
    _write_json(args.output_dir / "metrics_eval.json", payload)
    _write_report(args.report_path, payload)
    return 0 if decision == "PASS_FRESH_EXTERNAL_CONFIRMATION_NON_PROMOTING" else 1


def _environment_summary() -> dict[str, Any]:
    import os

    fs_stat = os.statvfs(".")
    free = int(fs_stat.f_bavail * fs_stat.f_frsize)
    return {
        "disk_free_bytes": free,
        "disk_free_gib": free / 1024**3,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }


def _ensure_files(raw_dir: Path, *, download_missing: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, url in ((CP_RELATIVE_PATH, CP_URL), (L1K_RELATIVE_PATH, L1K_URL)):
        path = raw_dir / relative
        expected = EXPECTED_BYTES[str(relative)]
        if path.exists() and path.stat().st_size >= int(0.95 * expected):
            status = "present"
        elif download_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            _download_file(url, path)
            status = "downloaded"
        else:
            status = "missing"
        rows.append(
            {
                "relative_path": str(relative),
                "local_path": str(path),
                "expected_bytes": expected,
                "actual_bytes": int(path.stat().st_size) if path.exists() else 0,
                "status": status,
            }
        )
    return rows


def _download_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _build_condition_pairs(raw_dir: Path, *, output_dir: Path, split_mode: str) -> dict[str, Any]:
    cp_path = raw_dir / CP_RELATIVE_PATH
    l1k_path = raw_dir / L1K_RELATIVE_PATH
    cp = pd.read_csv(cp_path, compression="gzip", low_memory=False)
    l1k = pd.read_csv(l1k_path, compression="gzip", low_memory=False)
    cp = _prepare_cp_frame(cp)
    l1k = _prepare_l1k_frame(l1k)

    cp_features = _numeric_feature_columns(
        cp,
        metadata_columns={
            "condition_key",
            "compound_id",
            "compound_core",
            "dose_um",
            "split",
            "is_control",
            "cpd_name",
            "smiles",
        },
    )
    l1k_features = _numeric_feature_columns(
        l1k,
        metadata_columns={
            "condition_key",
            "compound_id",
            "compound_core",
            "dose_um",
            "split",
            "is_control",
            "cpd_name",
            "smiles",
        },
    )
    cp_non = cp.loc[~cp["is_control"]].copy()
    l1k_non = l1k.loc[~l1k["is_control"]].copy()
    shared_keys = sorted(set(cp_non["condition_key"]) & set(l1k_non["condition_key"]))
    if split_mode == "same_condition_replicate":
        return _build_replicate_holdout_pairs(
            cp_non,
            l1k_non,
            cp_features,
            l1k_features,
            shared_keys,
            cp_controls=_control_values(cp, cp_features, split_col="Metadata_Plate"),
            l1k_controls=_control_values(l1k, l1k_features, split_col="det_plate"),
            output_dir=output_dir,
        )
    cp_shared = cp_non[cp_non["condition_key"].isin(shared_keys)].copy()
    l1k_shared = l1k_non[l1k_non["condition_key"].isin(shared_keys)].copy()
    cp_meta = _first_metadata(cp_shared, ["condition_key", "compound_id", "compound_core", "dose_um", "cpd_name", "smiles"])
    l1k_meta = _first_metadata(l1k_shared, ["condition_key", "compound_id", "compound_core", "dose_um", "cpd_name", "smiles"])
    metadata = cp_meta.merge(l1k_meta, on="condition_key", suffixes=("_cp", "_l1k"), how="inner")
    metadata["compound_id"] = metadata["compound_id_l1k"].fillna(metadata["compound_id_cp"])
    metadata["compound_core"] = metadata["compound_core_l1k"].fillna(metadata["compound_core_cp"])
    metadata["dose_um"] = pd.to_numeric(metadata["dose_um_l1k"].fillna(metadata["dose_um_cp"]), errors="coerce")
    metadata["cpd_name"] = metadata["cpd_name_l1k"].fillna(metadata["cpd_name_cp"]).fillna(metadata["compound_id"])
    metadata["smiles"] = metadata["smiles_l1k"].fillna(metadata["smiles_cp"]).fillna("")
    metadata["split"] = metadata["compound_id"].map(_split_for_compound)
    metadata["is_control"] = False
    metadata = metadata[
        ["condition_key", "compound_id", "compound_core", "dose_um", "cpd_name", "smiles", "split", "is_control"]
    ].sort_values(["split", "compound_id", "dose_um"]).reset_index(drop=True)

    cp_means = _condition_means(cp_shared, cp_features, shared_keys)
    l1k_means = _condition_means(l1k_shared, l1k_features, shared_keys)
    key_index = pd.Index(shared_keys)
    order = key_index.get_indexer(metadata["condition_key"])
    if np.any(order < 0):
        raise RuntimeError("Internal pairing error: metadata condition not found in mean matrix.")
    cp_values = cp_means[order]
    l1k_values = l1k_means[order]
    cp_train_controls = _control_values(cp, cp_features, split_col="Metadata_Plate")
    l1k_train_controls = _control_values(l1k, l1k_features, split_col="det_plate")
    cp_counts = cp_shared.groupby("condition_key", sort=False).size().reindex(metadata["condition_key"]).fillna(0).astype(int)
    l1k_counts = l1k_shared.groupby("condition_key", sort=False).size().reindex(metadata["condition_key"]).fillna(0).astype(int)
    metadata["n_cp"] = cp_counts.to_numpy(dtype=int)
    metadata["n_l1k"] = l1k_counts.to_numpy(dtype=int)
    metadata.to_csv(output_dir / "paired_condition_metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"cp_feature": cp_features}).to_csv(output_dir / "cp_feature_columns.tsv", sep="\t", index=False)
    pd.DataFrame({"l1k_feature": l1k_features}).to_csv(output_dir / "l1k_feature_columns.tsv", sep="\t", index=False)
    return {
        "metadata": metadata,
        "cp": cp_values.astype(np.float64),
        "l1k": l1k_values.astype(np.float64),
        "cp_train_controls": cp_train_controls.astype(np.float64),
        "l1k_train_controls": l1k_train_controls.astype(np.float64),
        "cp_feature_count": len(cp_features),
        "l1k_feature_count": len(l1k_features),
        "cp_shape": tuple(cp.shape),
        "l1k_shape": tuple(l1k.shape),
    }


def _build_replicate_holdout_pairs(
    cp_non: pd.DataFrame,
    l1k_non: pd.DataFrame,
    cp_features: list[str],
    l1k_features: list[str],
    shared_keys: list[str],
    *,
    cp_controls: np.ndarray,
    l1k_controls: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cp_values: list[np.ndarray] = []
    l1k_values: list[np.ndarray] = []
    for key in shared_keys:
        cp_group = cp_non.loc[cp_non["condition_key"].eq(key)].sort_values(["Metadata_Plate", "Metadata_Well"])
        l1k_group = l1k_non.loc[l1k_non["condition_key"].eq(key)].sort_values(["det_plate"])
        if cp_group.empty or l1k_group.empty:
            continue
        base = l1k_group.iloc[0]
        meta = {
            "condition_key": key,
            "compound_id": str(base["compound_id"]),
            "compound_core": str(base["compound_core"]),
            "dose_um": float(base["dose_um"]),
            "cpd_name": str(base.get("cpd_name", base["compound_id"])),
            "smiles": str(base.get("smiles", "")),
            "is_control": False,
        }
        cp_train = cp_group.iloc[:-1] if len(cp_group) >= 2 else cp_group
        cp_eval = cp_group.iloc[-1:]
        l1k_train = l1k_group.iloc[:-1] if len(l1k_group) >= 2 else l1k_group
        l1k_eval = l1k_group.iloc[-1:]
        rows.append(
            {
                **meta,
                "split": "train",
                "n_cp": int(len(cp_train)),
                "n_l1k": int(len(l1k_train)),
                "_value_index": len(cp_values),
            }
        )
        cp_values.append(_frame_feature_mean(cp_train, cp_features))
        l1k_values.append(_frame_feature_mean(l1k_train, l1k_features))
        if len(cp_group) >= 2 and len(l1k_group) >= 2:
            rows.append(
                {
                    **meta,
                    "split": _eval_split_for_condition(key),
                    "n_cp": int(len(cp_eval)),
                    "n_l1k": int(len(l1k_eval)),
                    "_value_index": len(cp_values),
                }
            )
            cp_values.append(_frame_feature_mean(cp_eval, cp_features))
            l1k_values.append(_frame_feature_mean(l1k_eval, l1k_features))
    metadata = pd.DataFrame(rows).sort_values(["split", "compound_id", "dose_um"]).reset_index(drop=True)
    order = metadata.pop("_value_index").to_numpy(dtype=int)
    cp_array = np.vstack(cp_values).astype(np.float64)[order]
    l1k_array = np.vstack(l1k_values).astype(np.float64)[order]
    metadata.to_csv(output_dir / "paired_condition_metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"cp_feature": cp_features}).to_csv(output_dir / "cp_feature_columns.tsv", sep="\t", index=False)
    pd.DataFrame({"l1k_feature": l1k_features}).to_csv(output_dir / "l1k_feature_columns.tsv", sep="\t", index=False)
    return {
        "metadata": metadata,
        "cp": cp_array,
        "l1k": l1k_array,
        "cp_train_controls": cp_controls,
        "l1k_train_controls": l1k_controls,
        "cp_feature_count": len(cp_features),
        "l1k_feature_count": len(l1k_features),
        "cp_shape": tuple(cp_non.shape),
        "l1k_shape": tuple(l1k_non.shape),
    }


def _prepare_cp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_control"] = ~out["Metadata_pert_type"].astype(str).str.lower().eq("trt")
    out["compound_id"] = out["Metadata_pert_mfc_id"].map(_normalize_id)
    out["compound_core"] = out["Metadata_pert_id"].map(_normalize_id)
    out["dose_um"] = pd.to_numeric(out["Metadata_mmoles_per_liter2"], errors="coerce").round(4)
    out.loc[out["is_control"], "compound_id"] = "DMSO"
    out.loc[out["is_control"], "compound_core"] = "DMSO"
    out.loc[out["is_control"], "dose_um"] = 0.0
    out["condition_key"] = out["compound_id"] + "::" + out["dose_um"].map(_dose_text)
    out["cpd_name"] = out.get("Metadata_pert_iname2", out["compound_id"]).astype(object)
    out["smiles"] = ""
    return out


def _prepare_l1k_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_control"] = ~out["pert_type"].astype(str).str.lower().eq("trt")
    out["compound_id"] = out["pert_id"].map(_normalize_id)
    out["compound_core"] = out["compound_id"].str.replace(r"-\d{3}-\d{2}-\d+$", "", regex=True)
    out["dose_um"] = pd.to_numeric(out["pert_dose"], errors="coerce").round(4)
    out.loc[out["is_control"], "compound_id"] = "DMSO"
    out.loc[out["is_control"], "compound_core"] = "DMSO"
    out.loc[out["is_control"], "dose_um"] = 0.0
    out["condition_key"] = out["compound_id"] + "::" + out["dose_um"].map(_dose_text)
    out["cpd_name"] = out.get("CPD_NAME", out["compound_id"]).astype(object)
    out["smiles"] = out.get("CPD_SMILES", "").fillna("").astype(str)
    return out


def _numeric_feature_columns(frame: pd.DataFrame, *, metadata_columns: set[str]) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        text = str(column)
        if text in metadata_columns or text.startswith("Metadata_"):
            continue
        if text in {"pert_type", "control_type", "pert_id", "pert_dose", "det_plate", "BROAD_CPD_ID", "CPD_NAME", "CPD_TYPE",
                    "CPD_SMILES", "pert_id_dose", "pert_sample_dose", "cpd_name", "smiles", "compound_id", "compound_core",
                    "condition_key", "is_control", "dose_um"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(text)
    if not columns:
        raise ValueError("No numeric feature columns found.")
    return columns


def _first_metadata(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].drop_duplicates("condition_key").reset_index(drop=True)


def _condition_means(frame: pd.DataFrame, feature_columns: list[str], keys: list[str]) -> np.ndarray:
    clean = frame[["condition_key", *feature_columns]].copy()
    clean[feature_columns] = clean[feature_columns].replace([np.inf, -np.inf], np.nan)
    grouped = clean.groupby("condition_key", sort=False)[feature_columns].mean()
    grouped = grouped.reindex(keys)
    if grouped.isna().any().any():
        grouped = grouped.fillna(0.0)
    return _finite_matrix(grouped.to_numpy(dtype=np.float64))


def _control_values(frame: pd.DataFrame, feature_columns: list[str], *, split_col: str) -> np.ndarray:
    controls = frame.loc[frame["is_control"]].copy()
    if controls.empty:
        raise ValueError("No control rows found for source-state estimation.")
    if split_col in controls.columns:
        train = controls[controls[split_col].map(lambda value: _hash_fraction(str(value)) < 0.70)]
        if train.empty:
            train = controls
    else:
        train = controls
    values = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _finite_matrix(values.astype(float).to_numpy(dtype=np.float64))


def _frame_feature_mean(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    values = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _finite_matrix(values.astype(float).to_numpy(dtype=np.float64)).mean(axis=0)


def _build_tables(
    paired: dict[str, Any],
    *,
    output_dir: Path,
    min_shared_noncontrol: int,
    pca_dim: int,
    hash_dim: int,
) -> dict[str, Any]:
    metadata = paired["metadata"].copy()
    l1k = np.asarray(paired["l1k"], dtype=np.float64)
    cp = np.asarray(paired["cp"], dtype=np.float64)
    split_counts = (
        metadata.groupby("split", sort=True)
        .agg(
            paired_conditions=("condition_key", "nunique"),
            compounds=("compound_id", "nunique"),
            min_cp_replicates=("n_cp", "min"),
            median_cp_replicates=("n_cp", "median"),
            max_cp_replicates=("n_cp", "max"),
            min_l1k_replicates=("n_l1k", "min"),
            median_l1k_replicates=("n_l1k", "median"),
            max_l1k_replicates=("n_l1k", "max"),
        )
        .reset_index()
    )
    split_counts.to_csv(output_dir / "split_pairing_summary.tsv", sep="\t", index=False)
    train_mask = metadata["split"].eq("train").to_numpy()
    expected = {"train", "validation", "test", "alternate_test"}
    if not expected.issubset(set(metadata["split"])):
        return {
            "preflight_pass": False,
            "reason": "missing expected split",
            "pairing_summary": split_counts.to_dict("records"),
            "descriptor_summary": {},
            "control_summary": {},
        }
    if int(train_mask.sum()) < int(min_shared_noncontrol):
        return {
            "preflight_pass": False,
            "reason": "insufficient train non-control condition pairs",
            "pairing_summary": split_counts.to_dict("records"),
            "descriptor_summary": {},
            "control_summary": {},
        }
    cp_pca, source_cp_pca = _train_only_pca_with_source(
        cp,
        train_mask,
        paired["cp_train_controls"],
        dim=pca_dim,
    )
    source_l1k = np.asarray(paired["l1k_train_controls"], dtype=np.float64).mean(axis=0, keepdims=True)
    descriptors, descriptor_columns, descriptor_summary = _action_descriptors(metadata, train_mask, hash_dim=hash_dim)
    descriptor_frame = pd.DataFrame(descriptors, columns=descriptor_columns)
    descriptor_frame.insert(0, "condition_key", metadata["condition_key"].to_numpy())
    descriptor_frame.head(20).to_csv(output_dir / "action_descriptor_matrix_head.tsv", sep="\t", index=False)
    tables: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test", "alternate_test"):
        mask = metadata["split"].eq(split).to_numpy()
        split_metadata = metadata.loc[mask].reset_index(drop=True).copy()
        split_metadata["perturbation"] = split_metadata["compound_id"]
        split_metadata["dose"] = split_metadata["dose_um"].map(_dose_text)
        tables[split] = {
            "control_expression": np.repeat(source_l1k, int(mask.sum()), axis=0),
            "target_expression": l1k[mask],
            "target_image_flat": cp[mask],
            "pca_target": cp_pca[mask],
            "source_pca_target": np.repeat(source_cp_pca, int(mask.sum()), axis=0),
            "action": descriptors[mask],
            "metadata": split_metadata,
        }
    control_summary = {
        "cp_train_control_rows": int(np.asarray(paired["cp_train_controls"]).shape[0]),
        "l1k_train_control_rows": int(np.asarray(paired["l1k_train_controls"]).shape[0]),
        "source_l1k_dim": int(source_l1k.shape[1]),
        "source_cp_pca_dim": int(source_cp_pca.shape[1]),
    }
    config = ProgramBootstrapJEPAConfig(
        genes=int(l1k.shape[1]),
        image_dim=int(cp.shape[1]),
        action_dim=int(descriptors.shape[1]),
        z_dim=int(pca_dim),
        hidden_dim=112,
        pca_anchor_weight=0.0,
        transition_weight=1.0,
        cross_modal_weight=1.0,
        action_negative_weight=0.1,
        delta_direction_weight=0.0,
        source_improvement_weight=0.0,
        variance_weight=0.0,
    )
    return {
        "preflight_pass": True,
        "train_table": tables["train"],
        "eval_tables": {key: tables[key] for key in ("validation", "test", "alternate_test")},
        "config": config,
        "pairing_summary": split_counts.to_dict("records"),
        "descriptor_summary": descriptor_summary,
        "control_summary": control_summary,
    }


def _train_only_pca_with_source(
    values: np.ndarray,
    train_mask: np.ndarray,
    source_values: np.ndarray,
    *,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_matrix(values)
    source_values = _finite_matrix(source_values)
    train = np.asarray(values[train_mask], dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    centered = train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_dim = max(1, min(int(dim), vt.shape[0]))
    basis = vt[:use_dim].T
    transformed = (np.asarray(values, dtype=np.float64) - mean) @ basis
    source_transformed = (np.asarray(source_values, dtype=np.float64) - mean) @ basis
    if use_dim < dim:
        transformed = np.pad(transformed, ((0, 0), (0, dim - use_dim)))
        source_transformed = np.pad(source_transformed, ((0, 0), (0, dim - use_dim)))
    return transformed.astype(np.float64), source_transformed.mean(axis=0, keepdims=True).astype(np.float64)


def _finite_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _action_descriptors(metadata: pd.DataFrame, train_mask: np.ndarray, *, hash_dim: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    scalar_columns = list(CHEM_SCALAR_COLUMNS)
    hash_columns = [f"smiles_hash_{idx:03d}" for idx in range(int(hash_dim))]
    raw_rows = []
    missing_smiles = 0
    for row in metadata.itertuples(index=False):
        smiles = str(getattr(row, "smiles", "") or "")
        dose = float(getattr(row, "dose_um", 0.0) or 0.0)
        missing_smiles += int(not smiles.strip())
        scalars = _chem_scalar_features(smiles, dose=dose, is_control=False)
        hashed = _hashed_smiles_ngrams(smiles, dim=int(hash_dim))
        raw_rows.append(np.concatenate([scalars, hashed], axis=0))
    raw = np.vstack(raw_rows).astype(np.float64)
    mean = raw[train_mask].mean(axis=0, keepdims=True)
    std = raw[train_mask].std(axis=0, keepdims=True)
    scaled = (raw - mean) / np.maximum(std, 1.0e-6)
    columns = scalar_columns + hash_columns
    return scaled, columns, {
        "action_descriptor": "SMILES-derived chemistry counts, dose, and deterministic hashed SMILES n-grams; not exact treatment one-hot",
        "action_dim": int(scaled.shape[1]),
        "hash_dim": int(hash_dim),
        "missing_noncontrol_smiles_rows": int(missing_smiles),
        "descriptor_columns": columns,
    }


def _chem_scalar_features(smiles: str, *, dose: float, is_control: bool) -> np.ndarray:
    text = str(smiles or "")
    aromatic = sum(1 for char in text if char in "bcnops")
    values = [
        float(dose),
        float(math.log10(max(float(dose), 1.0e-6))),
        float(len(text)),
        float(text.count("C") + aromatic),
        float(text.count("N") + text.count("n")),
        float(text.count("O") + text.count("o")),
        float(text.count("S") + text.count("s")),
        float(text.count("P") + text.count("p")),
        float(text.count("F")),
        float(text.count("Cl")),
        float(text.count("Br")),
        float(text.count("I")),
        float(sum(char.isdigit() for char in text)),
        float(text.count("(") + text.count(")")),
        float(text.count("=")),
        float(text.count("#")),
        float(aromatic),
        float(text.count("+") + text.count("-")),
        float(is_control),
    ]
    return np.asarray(values, dtype=np.float64)


def _hashed_smiles_ngrams(smiles: str, *, dim: int) -> np.ndarray:
    out = np.zeros(int(dim), dtype=np.float64)
    text = str(smiles or "").strip()
    if not text:
        return out
    padded = f"^{text}$"
    for ngram_size in (2, 3, 4, 5):
        if len(padded) < ngram_size:
            continue
        for start in range(0, len(padded) - ngram_size + 1):
            token = padded[start : start + ngram_size]
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % int(dim)
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            out[index] += sign
    norm = np.linalg.norm(out)
    if norm > 0.0:
        out = out / norm
    return out


def _baseline_rows(train_table: dict[str, Any], eval_table: dict[str, Any], *, seed: int, split: str) -> list[dict[str, Any]]:
    bundle = _table_bundle(eval_table, name=f"f097_{split}")
    zeros = np.zeros_like(bundle.delta)
    rows = [_metric_row("source_as_target", zeros, bundle, seed=seed, split=split)]
    train_bundle = _table_bundle(train_table, name="f097_train")
    floor_fit = fit_ridge_numpy(bundle_features(train_bundle), train_bundle.delta, alpha=1.0e-2)
    floor_delta = predict_ridge_numpy(floor_fit, bundle_features(bundle))
    rows.append(_metric_row("protected_full_ridge_floor", floor_delta, bundle, seed=seed, split=split))
    return rows


def _raw_uncalibrated_row(model: Any, eval_table: dict[str, Any], *, seed: int, split: str, device: str) -> dict[str, Any]:
    from scripts.run_bioguard_wm_total_autonomy import _encode_program_jepa_table

    encoded = _encode_program_jepa_table(model, eval_table, device=device)
    bundle = _table_bundle(eval_table, name=f"f097_raw_{split}")
    return _metric_row("no_delta_calibration", encoded["transition"] - bundle.source, bundle, seed=seed, split=split)


def _table_bundle(table: dict[str, Any], *, name: str) -> LatentOperatorBundle:
    return LatentOperatorBundle(
        name=name,
        arrays={
            "source_z_bio_teacher": table["source_pca_target"].astype(np.float64),
            "target_z_bio_teacher": table["pca_target"].astype(np.float64),
            "action_descriptor": table["action"].astype(np.float64),
        },
        metadata=table["metadata"],
    )


def _metric_row(method: str, pred_delta: np.ndarray, bundle: LatentOperatorBundle, *, seed: int, split: str) -> dict[str, Any]:
    metrics = bundle_transition_metrics(bundle, pred_delta)
    return {
        "seed": seed,
        "split": split,
        "method": method,
        "transition_improvement": metrics["transition_source_cosine_improvement"],
        "delta_cosine": metrics["delta_cosine"],
        "recall_at_1": metrics.get("transition_to_target_recall@1", 0.0),
        "delta_rank": metrics["delta_prediction_effective_rank"],
        "magnitude_ratio": metrics["delta_magnitude_ratio"],
    }


def _summary_metrics(per_seed: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    candidate = per_seed.rename(
        columns={
            "calibrated_transition_improvement": "transition_improvement",
            "calibrated_delta_cosine": "delta_cosine",
            "calibrated_recall_at_1": "recall_at_1",
            "calibrated_delta_rank": "delta_rank",
            "calibrated_magnitude_ratio": "magnitude_ratio",
        }
    )
    candidate = candidate[
        [
            "seed",
            "split",
            "method",
            "transition_improvement",
            "delta_cosine",
            "recall_at_1",
            "delta_rank",
            "magnitude_ratio",
            "rna_to_image_recall_at_1",
            "image_to_rna_recall_at_1",
            "identity_violation",
            "leakage_flag",
        ]
    ].copy()
    all_rows = pd.concat([candidate, baselines], ignore_index=True, sort=False)
    summary = (
        all_rows.groupby(["method", "split"], sort=True)
        .agg(
            mean_transition_improvement=("transition_improvement", "mean"),
            mean_delta_cosine=("delta_cosine", "mean"),
            mean_recall_at_1=("recall_at_1", "mean"),
            mean_delta_rank=("delta_rank", "mean"),
            mean_magnitude_ratio=("magnitude_ratio", "mean"),
            mean_rna_to_image_recall_at_1=("rna_to_image_recall_at_1", "mean"),
            mean_image_to_rna_recall_at_1=("image_to_rna_recall_at_1", "mean"),
            max_identity_violation=("identity_violation", "max"),
            max_leakage_flag=("leakage_flag", "max"),
        )
        .reset_index()
    )
    floor = summary[summary["method"] == "protected_full_ridge_floor"][
        ["split", "mean_transition_improvement", "mean_delta_cosine", "mean_recall_at_1"]
    ].rename(
        columns={
            "mean_transition_improvement": "floor_transition_improvement",
            "mean_delta_cosine": "floor_delta_cosine",
            "mean_recall_at_1": "floor_recall_at_1",
        }
    )
    summary = summary.merge(floor, on="split", how="left")
    for metric in ("transition_improvement", "delta_cosine", "recall_at_1"):
        summary[f"floor_gap_{metric}"] = summary[f"mean_{metric}"] - summary[f"floor_{metric}"]
    return summary


def _decision(summary: pd.DataFrame, preflight: dict[str, Any], *, candidate_method: str) -> str:
    if not preflight.get("preflight_pass", False):
        return "FAIL_CPG0003_PREFLIGHT_NON_PROMOTING"
    candidate = summary[summary["method"].eq(candidate_method)]
    if set(candidate["split"]) != {"validation", "test", "alternate_test"}:
        return "FAIL_CPG0003_MISSING_SPLITS_NON_PROMOTING"
    no_leak = bool(
        candidate["max_identity_violation"].fillna(0.0).le(0.0).all()
        and candidate["max_leakage_flag"].fillna(0.0).le(0.0).all()
    )
    beats_floor = bool(
        candidate["floor_gap_transition_improvement"].gt(0.0).all()
        and candidate["floor_gap_delta_cosine"].ge(0.0).all()
        and candidate["floor_gap_recall_at_1"].ge(0.0).all()
    )
    positive = bool(candidate["mean_transition_improvement"].gt(0.0).all() and candidate["mean_delta_cosine"].gt(0.0).all())
    if no_leak and beats_floor and positive:
        return "PASS_FRESH_EXTERNAL_CONFIRMATION_NON_PROMOTING"
    return "FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    summary = pd.DataFrame(payload.get("summary_metrics", []))
    baselines = pd.DataFrame(payload.get("baseline_metrics", []))
    summary_tsv = summary.to_csv(sep="\t", index=False) if not summary.empty else "No summary metrics.\n"
    baseline_summary = (
        baselines.groupby(["method", "split"], sort=True)
        .agg(
            transition_improvement=("transition_improvement", "mean"),
            delta_cosine=("delta_cosine", "mean"),
            recall_at_1=("recall_at_1", "mean"),
            delta_rank=("delta_rank", "mean"),
            magnitude_ratio=("magnitude_ratio", "mean"),
        )
        .reset_index()
        if not baselines.empty
        else pd.DataFrame()
    )
    baseline_tsv = baseline_summary.to_csv(sep="\t", index=False) if not baseline_summary.empty else "No baselines.\n"
    pairing_tsv = pd.DataFrame(preflight.get("pairing_summary", [])).to_csv(sep="\t", index=False)
    descriptor = preflight.get("descriptor_summary", {})
    experiment_id = payload.get("preflight", {}).get("experiment_id", "F097")
    text = f"""# {experiment_id} cpg0003 Rosetta External Confirmation

## Decision
`{payload['decision']}`

No model is promoted. The protected rank-3 train-split-only PLS raw-linear
readout remains the model of record.

## Scope
- candidate family: `F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`
- frozen path: ProgramBootstrapJEPA plus train-only delta calibration
- external validator: cpg0003 Rosetta CDRPBIO-BBBC036-Bray
- modalities: L1000 expression plus Cell Painting morphology
- caveat: this is not scRNA
- split mode: `{preflight.get('split_mode', 'unknown')}`
- action descriptor: {descriptor.get('action_descriptor', '')}
- action dim: `{descriptor.get('action_dim', 0)}`
- missing non-control SMILES rows: `{descriptor.get('missing_noncontrol_smiles_rows', 0)}`

## Split And Pairing
```tsv
{pairing_tsv}
```

## Candidate And Floor Comparison
```tsv
{summary_tsv}
```

## Baselines
```tsv
{baseline_tsv}
```

## Leakage And Identity
{payload.get('leakage_controls', 'Leakage controls unavailable.')}

## Promotion Status
{payload.get('promotion_caveat', 'No model is promoted.')}
"""
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _split_for_compound(value: object) -> str:
    frac = _hash_fraction(str(value))
    if frac < 0.70:
        return "train"
    if frac < 0.80:
        return "validation"
    if frac < 0.90:
        return "test"
    return "alternate_test"


def _eval_split_for_condition(value: object) -> str:
    frac = _hash_fraction(str(value))
    if frac < 1.0 / 3.0:
        return "validation"
    if frac < 2.0 / 3.0:
        return "test"
    return "alternate_test"


def _hash_fraction(value: str) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _normalize_id(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"", "NAN", "NONE", "NULL"}:
        return ""
    return text


def _dose_text(value: object) -> str:
    number = float(value) if value == value else 0.0
    return f"{number:.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
