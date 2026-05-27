from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import urllib.parse
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.schema import normalize_value
from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics
from perturb_jepa.models.program_bootstrap_jepa import ProgramBootstrapJEPAConfig
from perturb_jepa.training.bioguard_wm_calibration import make_action_grouped_folds, select_jepa_calibration_blend_gate
from perturb_jepa.training.biospectral_operator import (
    LatentOperatorBundle,
    bundle_features,
    bundle_transition_metrics,
    effective_rank,
    fit_ridge_numpy,
    predict_ridge_numpy,
)
from scripts.prepare_scgenescope_pairing import CONTROL_TOKENS, OFFICIAL_TREATMENTS
from scripts.run_bioguard_wm_total_autonomy import (
    _encode_program_jepa_table,
    _evaluate_delta_calibrated_source_delta_rank_jepa,
    _evaluate_source_delta_rank_program_jepa,
    _predict_program_jepa_transition,
    _train_source_delta_rank_program_jepa,
)


HF_REPO_ID = "altoslabs/scGeneScope"
DEFAULT_OUTPUT_DIR = Path("outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_f082_scgenescope")
DEFAULT_REPORT_PATH = Path("outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_report.md")
FEATURE_FILES = {
    ("rna", "1"): "features/rnaseq/scvi/n200/round_1.h5ad",
    ("rna", "2"): "features/rnaseq/scvi/n200/round_2.h5ad",
    ("image", "1"): "features/imaging/imagenet/vit-l/round_1.h5ad",
    ("image", "2"): "features/imaging/imagenet/vit-l/round_2.h5ad",
}
EXPECTED_BYTES = {
    "features/rnaseq/scvi/n200/round_1.h5ad": 4_366_952_116,
    "features/rnaseq/scvi/n200/round_2.h5ad": 2_565_764_148,
    "features/imaging/imagenet/vit-l/round_1.h5ad": 18_506_448_104,
    "features/imaging/imagenet/vit-l/round_2.h5ad": 11_023_900_503,
}
REPLICATE_TO_SPLIT = {
    "3": "train",
    "5": "validation",
    "4": "test",
    "1": "alternate_test",
    "2": "alternate_test",
}
PUBCHEM_PROPERTIES = (
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "CovalentUnitCount",
)
PUBCHEM_FINGERPRINT_BITS = 920


@dataclass(frozen=True)
class ConditionFeatures:
    values: np.ndarray
    metadata: pd.DataFrame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run F082 external validation on scGeneScope feature H5ADs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/scgenescope"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="*", type=int, default=[37, 38, 39])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--download-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--rna-obsm-key", default="scvi_n200")
    parser.add_argument("--image-chunk-size", type=int, default=4096)
    parser.add_argument("--pubchem-timeout", type=float, default=12.0)
    parser.add_argument("--min-shared-noncontrol", type=int, default=20)
    parser.add_argument("--descriptor-mode", choices=["pubchem_scalar", "pubchem_fingerprint"], default="pubchem_scalar")
    parser.add_argument("--gate-mode", choices=["calibrated", "split_safe"], default="calibrated")
    parser.add_argument("--reuse-condition-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-cache-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    env = _environment_summary()
    download_status = _ensure_feature_files(args)
    obs_contract = _backed_obs_contract(args.raw_dir, output_dir=args.output_dir)
    condition_cache = args.output_dir / "condition_features.npz"
    metadata_cache = args.output_dir / "condition_metadata.tsv"
    paired, cache_source = _load_or_build_paired_conditions(args, condition_cache=condition_cache, metadata_cache=metadata_cache)

    split_pairing = _split_pairing_summary(paired["metadata"], output_dir=args.output_dir)
    descriptor_frame = _build_action_descriptors(
        paired["metadata"]["treatment"].drop_duplicates().tolist(),
        output_dir=args.output_dir,
        timeout=args.pubchem_timeout,
        descriptor_mode=args.descriptor_mode,
    )
    table_payload = _build_f082_tables(
        paired,
        descriptor_frame,
        min_shared_noncontrol=args.min_shared_noncontrol,
        descriptor_mode=args.descriptor_mode,
    )
    preflight = {
        "environment": env,
        "download_status": download_status,
        "obs_contract": obs_contract,
        "split_pairing": split_pairing,
        "descriptor_summary": _descriptor_summary(descriptor_frame, table_payload),
        "condition_cache": str(condition_cache),
        "metadata_cache": str(metadata_cache),
        "condition_cache_source": cache_source,
    }
    _write_json(args.output_dir / "preflight_summary.json", preflight)

    if not table_payload["preflight_pass"]:
        report_payload = {
            "decision": "FAIL_PREFLIGHT_NON_PROMOTING",
            "preflight": preflight,
            "per_seed_split_metrics": [],
            "summary_metrics": [],
            "baseline_metrics": [],
            "model_promoted": False,
        }
        _write_report(args.report_path, report_payload)
        _write_json(args.output_dir / "metrics_eval.json", report_payload)
        return 1

    per_seed_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        model, trace = _train_source_delta_rank_program_jepa(
            table_payload["train_table"],
            table_payload["config"],
            seed=int(seed) + 9182,
            device=args.device,
            output_dir=args.output_dir / f"seed_{seed}",
            steps=args.steps,
        )
        trace_rows.append({"seed": seed, **trace})
        gate_payload = None
        if args.gate_mode == "split_safe":
            gate_payload = _fit_split_safe_jepa_calibration_gate(
                model,
                table_payload["train_table"],
                seed=seed,
                device=args.device,
                output_dir=args.output_dir / f"seed_{seed}",
            )
            gate_rows.append({"seed": seed, **gate_payload["selection"]})
        for split, eval_table in table_payload["eval_tables"].items():
            eval_payload = _evaluate_delta_calibrated_source_delta_rank_jepa(
                model,
                table_payload["train_table"],
                eval_table,
                device=args.device,
            )
            per_seed_rows.append({"seed": seed, "split": split, "method": "F082_delta_calibrated", **eval_payload})
            if gate_payload is not None:
                gate_eval = _evaluate_split_safe_jepa_calibration_gate(
                    model,
                    table_payload["train_table"],
                    eval_table,
                    fit=gate_payload["fit"],
                    blend_alpha=float(gate_payload["selection"]["blend_alpha"]),
                    train_metrics=gate_payload["train_metrics"],
                    selection=gate_payload["selection"],
                    device=args.device,
                )
                per_seed_rows.append({"seed": seed, "split": split, "method": "F094_split_safe_jepa_calibration_gate", **gate_eval})
            baseline_rows.extend(_baseline_rows(table_payload["train_table"], eval_table, seed=seed, split=split))
            baseline_rows.append(_raw_uncalibrated_row(model, eval_table, seed=seed, split=split, device=args.device))

    per_seed_frame = pd.DataFrame(per_seed_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    trace_frame = pd.DataFrame(trace_rows)
    gate_frame = pd.DataFrame(gate_rows)
    summary_frame = _summary_metrics(per_seed_frame, baseline_frame)
    per_seed_frame.to_csv(args.output_dir / "f082_external_seed_split_metrics.tsv", sep="\t", index=False)
    baseline_frame.to_csv(args.output_dir / "external_baseline_metrics.tsv", sep="\t", index=False)
    trace_frame.to_csv(args.output_dir / "external_train_trace.tsv", sep="\t", index=False)
    if not gate_frame.empty:
        gate_frame.to_csv(args.output_dir / "f094_gate_selection.tsv", sep="\t", index=False)
    summary_frame.to_csv(args.output_dir / "external_summary_metrics.tsv", sep="\t", index=False)

    candidate_method = "F094_split_safe_jepa_calibration_gate" if args.gate_mode == "split_safe" else "F082_delta_calibrated"
    decision = _external_decision(summary_frame, preflight, candidate_method=candidate_method)
    report_payload = {
        "decision": decision,
        "candidate_method": candidate_method,
        "gate_mode": args.gate_mode,
        "preflight": preflight,
        "per_seed_split_metrics": per_seed_frame.to_dict("records"),
        "summary_metrics": summary_frame.to_dict("records"),
        "baseline_metrics": baseline_frame.to_dict("records"),
        "train_trace": trace_frame.to_dict("records"),
        "gate_selection": gate_frame.to_dict("records") if not gate_frame.empty else [],
        "model_promoted": False,
        "leakage_controls": (
            "F082 external validation uses train-only control source centroids, train-only image PCA, "
            "train-only delta calibration, and numeric PubChem property action descriptors. "
            "Treatment labels are used only for grouping, pairing, and relevance labels; no condition_key, "
            "biological_key, exact treatment one-hot, held-out target means, or pooled train+test statistics "
            "are model inputs."
        ),
    }
    _write_json(args.output_dir / "metrics_eval.json", report_payload)
    _write_report(args.report_path, report_payload)
    return 0 if decision == "PASS_EXTERNAL_TIER3_NON_PROMOTING" else 1


def _environment_summary() -> dict[str, Any]:
    stat = os.statvfs(".")
    free = int(stat.f_bavail * stat.f_frsize)
    return {
        "disk_free_bytes": free,
        "disk_free_gib": free / 1024**3,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }


def _ensure_feature_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    expected_total = sum(EXPECTED_BYTES.values())
    stat = os.statvfs(args.raw_dir.parent if args.raw_dir.parent.exists() else ".")
    free = int(stat.f_bavail * stat.f_frsize)
    if free < expected_total * 2:
        raise SystemExit(f"Insufficient free space for quota-safe scGeneScope validation: {free} < {expected_total * 2}")
    for (modality, round_id), repo_path in FEATURE_FILES.items():
        local_path = args.raw_dir / repo_path
        expected = EXPECTED_BYTES[repo_path]
        if local_path.exists() and local_path.stat().st_size >= int(0.95 * expected):
            status = "present"
        elif args.download_missing:
            local_path = Path(
                hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=args.raw_dir,
                )
            )
            status = "downloaded"
        else:
            status = "missing"
        rows.append(
            {
                "modality": modality,
                "round": round_id,
                "repo_path": repo_path,
                "local_path": str(local_path),
                "expected_bytes": expected,
                "actual_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
                "status": status,
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "download_status.tsv", sep="\t", index=False)
    return rows


def _load_or_build_paired_conditions(
    args: argparse.Namespace,
    *,
    condition_cache: Path,
    metadata_cache: Path,
) -> tuple[dict[str, Any], str]:
    search_dirs = [args.output_dir]
    if args.condition_cache_dir is not None:
        search_dirs.append(args.condition_cache_dir)
    if DEFAULT_OUTPUT_DIR not in search_dirs:
        search_dirs.append(DEFAULT_OUTPUT_DIR)
    if args.reuse_condition_cache:
        for cache_dir in search_dirs:
            cache_path = cache_dir / "condition_features.npz"
            meta_path = cache_dir / "condition_metadata.tsv"
            if cache_path.exists() and meta_path.exists():
                with np.load(cache_path) as arrays:
                    paired = {
                        "rna": np.asarray(arrays["rna_values"], dtype=np.float64),
                        "image": np.asarray(arrays["image_values"], dtype=np.float64),
                        "metadata": pd.read_csv(meta_path, sep="\t"),
                    }
                if cache_path.resolve() != condition_cache.resolve():
                    paired["metadata"].to_csv(metadata_cache, sep="\t", index=False)
                    np.savez_compressed(condition_cache, rna_values=paired["rna"], image_values=paired["image"])
                return paired, str(cache_path)
    rna_conditions = _condition_means_all_rounds(args, modality="rna")
    image_conditions = _condition_means_all_rounds(args, modality="image")
    paired = _pair_condition_features(rna_conditions, image_conditions)
    paired["metadata"].to_csv(metadata_cache, sep="\t", index=False)
    np.savez_compressed(condition_cache, rna_values=paired["rna"], image_values=paired["image"])
    return paired, "rebuilt_from_backed_h5ad"


def _backed_obs_contract(raw_dir: Path, *, output_dir: Path) -> dict[str, Any]:
    import anndata as ad

    rows: list[dict[str, Any]] = []
    required = {"Treatment", "Replicate", "batch", "Group"}
    for (modality, round_id), repo_path in FEATURE_FILES.items():
        path = raw_dir / repo_path
        obj = ad.read_h5ad(path, backed="r")
        obs = obj.obs
        columns = set(map(str, obs.columns))
        replicate_values = sorted({_normalize_integerish(value) for value in obs["Replicate"].dropna().unique()})
        rows.append(
            {
                "modality": modality,
                "round": round_id,
                "path": str(path),
                "n_obs": int(obj.n_obs),
                "n_vars": int(obj.n_vars),
                "required_obs_columns_present": bool(required.issubset(columns)),
                "replicate_values": ",".join(replicate_values),
                "split_values": ",".join(sorted({REPLICATE_TO_SPLIT.get(value, "unknown") for value in replicate_values})),
                "treatment_unique_count": int(obs["Treatment"].nunique(dropna=True)),
                "batch_unique_count": int(obs["batch"].nunique(dropna=True)),
                "group_unique_count": int(obs["Group"].nunique(dropna=True)),
            }
        )
        obj.file.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "all_rounds_backed_obs_contract.tsv", sep="\t", index=False)
    return {
        "all_required_obs_columns_present": bool(frame["required_obs_columns_present"].all()),
        "rows": rows,
    }


def _condition_means_all_rounds(args: argparse.Namespace, *, modality: str) -> ConditionFeatures:
    parts = []
    for round_id in ("1", "2"):
        path = args.raw_dir / FEATURE_FILES[(modality, round_id)]
        if modality == "rna":
            parts.append(_rna_condition_means(path, round_id=round_id, obsm_key=args.rna_obsm_key))
        else:
            parts.append(_image_condition_means(path, round_id=round_id, chunk_size=args.image_chunk_size))
    values = np.concatenate([part.values for part in parts], axis=0).astype(np.float64)
    metadata = pd.concat([part.metadata for part in parts], ignore_index=True, sort=False)
    return ConditionFeatures(values=values, metadata=metadata)


def _rna_condition_means(path: Path, *, round_id: str, obsm_key: str) -> ConditionFeatures:
    import anndata as ad

    obj = ad.read_h5ad(path, backed="r")
    if obsm_key not in obj.obsm:
        raise KeyError(f"{path} missing RNA obsm key {obsm_key!r}")
    values = np.asarray(obj.obsm[obsm_key], dtype=np.float64)
    metadata, row_codes = _condition_metadata_and_codes(obj.obs, round_id=round_id, modality="rna")
    keep = row_codes >= 0
    values = values[keep]
    codes = row_codes[keep]
    sums = np.zeros((len(metadata), values.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=len(metadata)).astype(np.float64)
    metadata["n_rna"] = counts.astype(int)
    obj.file.close()
    return ConditionFeatures(values=sums / np.maximum(counts[:, None], 1.0), metadata=metadata)


def _image_condition_means(path: Path, *, round_id: str, chunk_size: int) -> ConditionFeatures:
    import anndata as ad

    obj = ad.read_h5ad(path, backed="r")
    metadata, row_codes = _condition_metadata_and_codes(obj.obs, round_id=round_id, modality="image")
    sums = np.zeros((len(metadata), obj.n_vars), dtype=np.float64)
    counts = np.bincount(row_codes[row_codes >= 0], minlength=len(metadata)).astype(np.float64)
    x = obj.X
    for start in range(0, int(obj.n_obs), int(chunk_size)):
        stop = min(start + int(chunk_size), int(obj.n_obs))
        chunk_codes = row_codes[start:stop]
        keep = chunk_codes >= 0
        if not np.any(keep):
            continue
        block = np.asarray(x[start:stop], dtype=np.float64)[keep]
        np.add.at(sums, chunk_codes[keep], block)
    metadata["n_image"] = counts.astype(int)
    obj.file.close()
    return ConditionFeatures(values=sums / np.maximum(counts[:, None], 1.0), metadata=metadata)


def _condition_metadata_and_codes(obs: pd.DataFrame, *, round_id: str, modality: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(obs).reset_index(drop=True)
    frame["round"] = str(round_id)
    frame["replicate"] = frame["Replicate"].map(_normalize_integerish).astype(str)
    frame["split"] = frame["replicate"].map(REPLICATE_TO_SPLIT).astype(object).fillna("unknown")
    frame["treatment"] = frame["Treatment"].map(normalize_value)
    frame["is_control"] = frame["treatment"].map(_is_control)
    valid = frame["split"].ne("unknown").to_numpy()
    valid_frame = frame.loc[valid].copy()
    key_cols = ["round", "replicate", "split", "treatment", "is_control"]
    keys = valid_frame[key_cols].astype(str).agg("||".join, axis=1)
    unique_keys = pd.Index(sorted(keys.unique()))
    code_map = {key: idx for idx, key in enumerate(unique_keys)}
    row_codes = np.full(len(frame), -1, dtype=int)
    row_codes[np.flatnonzero(valid)] = keys.map(code_map).to_numpy(dtype=int)
    metadata = pd.DataFrame([key.split("||") for key in unique_keys], columns=key_cols)
    metadata["is_control"] = metadata["is_control"].map(lambda value: str(value) == "True")
    metadata["modality"] = modality
    return metadata, row_codes


def _pair_condition_features(rna: ConditionFeatures, image: ConditionFeatures) -> dict[str, Any]:
    key_cols = ["round", "replicate", "split", "treatment", "is_control"]
    rna_meta = rna.metadata.reset_index(drop=False).rename(columns={"index": "rna_index"})
    image_meta = image.metadata.reset_index(drop=False).rename(columns={"index": "image_index"})
    paired = rna_meta.merge(image_meta, on=key_cols, suffixes=("_rna", "_image"), how="inner")
    if paired.empty:
        raise SystemExit("No paired scGeneScope RNA/image condition rows were found.")
    paired = paired.sort_values(["split", "round", "replicate", "treatment"]).reset_index(drop=True)
    metadata = paired[key_cols + ["n_rna", "n_image"]].copy()
    metadata["condition_key"] = metadata[["round", "replicate", "treatment"]].astype(str).agg("::".join, axis=1)
    return {
        "rna": rna.values[paired["rna_index"].to_numpy(dtype=int)],
        "image": image.values[paired["image_index"].to_numpy(dtype=int)],
        "metadata": metadata,
    }


def _split_pairing_summary(metadata: pd.DataFrame, *, output_dir: Path) -> dict[str, Any]:
    frame = metadata.copy()
    frame["noncontrol"] = ~frame["is_control"].astype(bool)
    summary = (
        frame.groupby("split", sort=True)
        .agg(
            paired_conditions=("condition_key", "nunique"),
            noncontrol_conditions=("noncontrol", "sum"),
            treatments=("treatment", "nunique"),
            min_rna_profiles=("n_rna", "min"),
            min_image_profiles=("n_image", "min"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "split_pairing_summary.tsv", sep="\t", index=False)
    expected = {"train", "validation", "test", "alternate_test"}
    return {
        "all_expected_splits_present": bool(expected.issubset(set(summary["split"].astype(str)))),
        "rows": summary.to_dict("records"),
    }


def _build_action_descriptors(treatments: list[str], *, output_dir: Path, timeout: float, descriptor_mode: str) -> pd.DataFrame:
    cache_name = "pubchem_action_descriptors.tsv" if descriptor_mode == "pubchem_scalar" else "pubchem_fingerprint_action_descriptors.tsv"
    cache_path = output_dir / cache_name
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    fingerprint_columns = [f"pubchem_fp2d_{idx:03d}" for idx in range(PUBCHEM_FINGERPRINT_BITS)] if descriptor_mode == "pubchem_fingerprint" else []
    rows = []
    for treatment in sorted(set(map(str, treatments))):
        is_control = _is_control(treatment)
        properties = {key: 0.0 for key in PUBCHEM_PROPERTIES}
        fingerprint = np.zeros(PUBCHEM_FINGERPRINT_BITS, dtype=np.float64)
        cid = ""
        matched_name = ""
        found = False
        if not is_control:
            for candidate in _pubchem_name_candidates(treatment):
                payload = _fetch_pubchem_properties(candidate, timeout=timeout, include_fingerprint=descriptor_mode == "pubchem_fingerprint")
                if payload:
                    properties.update(payload["properties"])
                    if descriptor_mode == "pubchem_fingerprint":
                        fingerprint = np.asarray(payload.get("fingerprint_bits", fingerprint), dtype=np.float64)
                    cid = str(payload["cid"])
                    matched_name = candidate
                    found = True
                    break
        rows.append(
            {
                "treatment": treatment,
                "pubchem_cid": cid,
                "pubchem_matched_name": matched_name,
                "pubchem_found": found,
                "is_control_descriptor": is_control,
                **properties,
                **{column: float(fingerprint[idx]) for idx, column in enumerate(fingerprint_columns)},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, sep="\t", index=False)
    return frame


def _fetch_pubchem_properties(name: str, *, timeout: float, include_fingerprint: bool = False) -> dict[str, Any] | None:
    requested = list(PUBCHEM_PROPERTIES)
    if include_fingerprint:
        requested.append("Fingerprint2D")
    properties = ",".join(requested)
    quoted = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quoted}/property/{properties}/JSON"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        data = response.json()
        rows = data.get("PropertyTable", {}).get("Properties", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "cid": row.get("CID", ""),
            "properties": {key: float(row.get(key, 0.0) or 0.0) for key in PUBCHEM_PROPERTIES},
            "fingerprint_bits": _decode_pubchem_fingerprint(row.get("Fingerprint2D", "")) if include_fingerprint else None,
        }
    except Exception:
        return None


def _decode_pubchem_fingerprint(value: object) -> np.ndarray:
    text = str(value or "").strip()
    if not text:
        return np.zeros(PUBCHEM_FINGERPRINT_BITS, dtype=np.float64)
    try:
        decoded = base64.b64decode(text)
    except Exception:
        return np.zeros(PUBCHEM_FINGERPRINT_BITS, dtype=np.float64)
    bits = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8), bitorder="big").astype(np.float64)
    if bits.size < PUBCHEM_FINGERPRINT_BITS:
        bits = np.pad(bits, (0, PUBCHEM_FINGERPRINT_BITS - bits.size))
    return bits[:PUBCHEM_FINGERPRINT_BITS]


def _pubchem_name_candidates(name: str) -> list[str]:
    raw = str(name).strip()
    candidates = [raw]
    candidates.extend(part.strip() for part in re.split(r"/|;", raw) if part.strip())
    stripped = re.sub(r"\([^)]*\)", "", raw).strip()
    if stripped:
        candidates.append(stripped)
    cleaned = raw.replace("(hydrochloride)", "hydrochloride").replace(" / ", " ")
    candidates.append(cleaned)
    candidates.append(raw.replace("-", " "))
    candidates.append(raw.replace(" ", "-"))
    candidates.append(re.sub(r"([A-Za-z]+)(\d+)$", r"\1 \2", raw))
    candidates.append(re.sub(r"([A-Za-z]+)(\d+)$", r"\1-\2", raw))
    if raw.upper() == "SKII":
        candidates.extend(["SKI II", "SKI-II", "Ski II"])
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _build_f082_tables(
    paired: dict[str, Any],
    descriptor_frame: pd.DataFrame,
    *,
    min_shared_noncontrol: int,
    descriptor_mode: str,
) -> dict[str, Any]:
    metadata = paired["metadata"].copy()
    rna = np.asarray(paired["rna"], dtype=np.float64)
    image = np.asarray(paired["image"], dtype=np.float64)
    train_mask = metadata["split"].eq("train").to_numpy()
    control_train_mask = train_mask & metadata["is_control"].astype(bool).to_numpy()
    noncontrol = ~metadata["is_control"].astype(bool).to_numpy()
    if not control_train_mask.any():
        return {"preflight_pass": False, "reason": "no train DMSO/control condition"}
    if int((train_mask & noncontrol).sum()) < min_shared_noncontrol:
        return {"preflight_pass": False, "reason": "insufficient train non-control paired conditions"}
    image_pca = _train_only_pca(image, train_mask, dim=24)
    source_rna = rna[control_train_mask].mean(axis=0, keepdims=True)
    source_image_pca = image_pca[control_train_mask].mean(axis=0, keepdims=True)
    actions, action_columns, action_summary = _scaled_action_matrix(
        metadata["treatment"].tolist(),
        descriptor_frame,
        train_mask,
        descriptor_mode=descriptor_mode,
    )
    tables = {}
    for split in ("train", "validation", "test", "alternate_test"):
        mask = metadata["split"].eq(split).to_numpy() & noncontrol
        if not mask.any():
            continue
        split_metadata = metadata.loc[mask].reset_index(drop=True).copy()
        tables[split] = {
            "control_expression": np.repeat(source_rna, int(mask.sum()), axis=0),
            "target_expression": rna[mask],
            "target_image_flat": image[mask],
            "pca_target": image_pca[mask],
            "source_pca_target": np.repeat(source_image_pca, int(mask.sum()), axis=0),
            "action": actions[mask],
            "metadata": split_metadata,
        }
    required_eval = {"validation", "test", "alternate_test"}
    preflight_pass = "train" in tables and required_eval.issubset(set(tables))
    config = ProgramBootstrapJEPAConfig(
        genes=int(rna.shape[1]),
        image_dim=int(image.shape[1]),
        action_dim=int(actions.shape[1]),
        z_dim=24,
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
        "preflight_pass": preflight_pass,
        "train_table": tables.get("train"),
        "eval_tables": {key: tables[key] for key in ("validation", "test", "alternate_test") if key in tables},
        "config": config,
        "action_columns": action_columns,
        "action_summary": action_summary,
    }


def _train_only_pca(values: np.ndarray, train_mask: np.ndarray, *, dim: int) -> np.ndarray:
    train = np.asarray(values[train_mask], dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    centered = train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_dim = max(1, min(int(dim), vt.shape[0]))
    basis = vt[:use_dim].T
    out = (np.asarray(values, dtype=np.float64) - mean) @ basis
    if use_dim < dim:
        out = np.pad(out, ((0, 0), (0, dim - use_dim)))
    return out.astype(np.float64)


def _scaled_action_matrix(
    treatments: list[str],
    descriptor_frame: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    descriptor_mode: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    fingerprint_columns = sorted(column for column in descriptor_frame.columns if str(column).startswith("pubchem_fp2d_"))
    columns = list(PUBCHEM_PROPERTIES) + fingerprint_columns + ["pubchem_found", "is_control_descriptor"]
    lookup = descriptor_frame.set_index("treatment")
    raw = []
    missing = 0
    for treatment in treatments:
        row = lookup.loc[treatment] if treatment in lookup.index else None
        if row is None:
            missing += 1
            raw.append(np.zeros(len(columns), dtype=np.float64))
        else:
            raw.append(np.asarray([_floatish(row.get(column, 0.0)) for column in columns], dtype=np.float64))
            missing += int(not _boolish(row.get("pubchem_found", False)) and not _boolish(row.get("is_control_descriptor", False)))
    matrix = np.vstack(raw).astype(np.float64)
    mean = matrix[train_mask].mean(axis=0, keepdims=True)
    std = matrix[train_mask].std(axis=0, keepdims=True)
    scaled = (matrix - mean) / np.maximum(std, 1.0e-6)
    descriptor_name = (
        "PubChem numeric molecular properties, PubChem 2D structure fingerprint bits, found/control flags; not exact treatment one-hot"
        if descriptor_mode == "pubchem_fingerprint"
        else "PubChem numeric molecular properties plus found/control flags; not exact treatment one-hot"
    )
    return scaled, columns, {
        "action_descriptor": descriptor_name,
        "descriptor_mode": descriptor_mode,
        "action_dim": int(scaled.shape[1]),
        "missing_noncontrol_descriptor_rows": int(missing),
        "descriptor_columns": columns,
    }


def _baseline_rows(train_table: dict[str, Any], eval_table: dict[str, Any], *, seed: int, split: str) -> list[dict[str, Any]]:
    bundle = LatentOperatorBundle(
        name=f"external_{split}",
        arrays={
            "source_z_bio_teacher": eval_table["source_pca_target"].astype(np.float64),
            "target_z_bio_teacher": eval_table["pca_target"].astype(np.float64),
            "action_descriptor": eval_table["action"].astype(np.float64),
        },
        metadata=eval_table["metadata"],
    )
    zeros = np.zeros_like(bundle.delta)
    rows = [_metric_row("source_as_target", zeros, bundle, seed=seed, split=split)]
    train_bundle = LatentOperatorBundle(
        name="external_train",
        arrays={
            "source_z_bio_teacher": train_table["source_pca_target"].astype(np.float64),
            "target_z_bio_teacher": train_table["pca_target"].astype(np.float64),
            "action_descriptor": train_table["action"].astype(np.float64),
        },
        metadata=train_table["metadata"],
    )
    floor_fit = fit_ridge_numpy(bundle_features(train_bundle), train_bundle.delta, alpha=1.0e-2)
    floor_delta = predict_ridge_numpy(floor_fit, bundle_features(bundle))
    rows.append(_metric_row("protected_full_ridge_floor", floor_delta, bundle, seed=seed, split=split))
    return rows


def _raw_uncalibrated_row(model, eval_table: dict[str, Any], *, seed: int, split: str, device: str) -> dict[str, Any]:
    encoded = _encode_program_jepa_table(model, eval_table, device=device)
    bundle = LatentOperatorBundle(
        name=f"raw_{split}",
        arrays={
            "source_z_bio_teacher": eval_table["source_pca_target"].astype(np.float64),
            "target_z_bio_teacher": eval_table["pca_target"].astype(np.float64),
            "action_descriptor": eval_table["action"].astype(np.float64),
        },
        metadata=eval_table["metadata"],
    )
    return _metric_row("F082_no_delta_calibration", encoded["transition"] - bundle.source, bundle, seed=seed, split=split)


def _fit_split_safe_jepa_calibration_gate(
    model,
    train_table: dict[str, Any],
    *,
    seed: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = _encode_program_jepa_table(model, train_table, device=device)
    raw_delta = encoded["transition"] - train_table["source_pca_target"].astype(np.float64)
    true_delta = train_table["pca_target"].astype(np.float64) - train_table["source_pca_target"].astype(np.float64)
    metadata = train_table["metadata"].reset_index(drop=True).copy()
    folds = make_action_grouped_folds(metadata, n_folds=4, action_key="treatment", seed=int(seed) + 9400)
    alpha_grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        if fold.get("status") != "OK":
            continue
        fit_idx = np.asarray(fold["fit_indices"], dtype=int)
        cal_idx = np.asarray(fold["calibration_indices"], dtype=int)
        if fit_idx.size < 2 or cal_idx.size < 1:
            continue
        fold_fit = fit_ridge_numpy(raw_delta[fit_idx], true_delta[fit_idx], alpha=1.0e-2)
        fold_raw = raw_delta[cal_idx]
        fold_cal = predict_ridge_numpy(fold_fit, fold_raw)
        bundle = _table_bundle(train_table, cal_idx, name=f"f094_train_fold_{fold.get('fold_id', 0)}")
        raw_metrics = bundle_transition_metrics(bundle, fold_raw)
        for alpha in alpha_grid:
            blend_delta = (1.0 - float(alpha)) * fold_raw + float(alpha) * fold_cal
            metrics = bundle_transition_metrics(bundle, blend_delta)
            fold_rows.append(
                {
                    "fold_id": int(fold.get("fold_id", 0)),
                    "blend_alpha": float(alpha),
                    "transition_improvement": metrics["transition_source_cosine_improvement"],
                    "delta_cosine": metrics["delta_cosine"],
                    "recall_at_1": metrics.get("transition_to_target_recall@1", 0.0),
                    "delta_rank": metrics["delta_prediction_effective_rank"],
                    "magnitude_ratio": metrics["delta_magnitude_ratio"],
                    "transition_gap_vs_raw": metrics["transition_source_cosine_improvement"]
                    - raw_metrics["transition_source_cosine_improvement"],
                    "delta_cosine_gap_vs_raw": metrics["delta_cosine"] - raw_metrics["delta_cosine"],
                    "recall_gap_vs_raw": metrics.get("transition_to_target_recall@1", 0.0)
                    - raw_metrics.get("transition_to_target_recall@1", 0.0),
                }
            )
    selection = select_jepa_calibration_blend_gate(fold_rows, alpha_grid=alpha_grid)
    full_fit = fit_ridge_numpy(raw_delta, true_delta, alpha=1.0e-2)
    full_cal = predict_ridge_numpy(full_fit, raw_delta)
    full_blend = (1.0 - selection.blend_alpha) * raw_delta + selection.blend_alpha * full_cal
    train_bundle = _table_bundle(train_table, np.arange(raw_delta.shape[0]), name="f094_train_gate")
    raw_train_metrics = bundle_transition_metrics(train_bundle, raw_delta)
    blend_train_metrics = bundle_transition_metrics(train_bundle, full_blend)
    fold_frame = pd.DataFrame(fold_rows)
    if not fold_frame.empty:
        fold_frame.to_csv(output_dir / "f094_gate_fold_metrics.tsv", sep="\t", index=False)
    selection_payload = {
        "blend_alpha": float(selection.blend_alpha),
        "gate_status": selection.status,
        "cv_lcb_transition_gap_vs_raw": float(selection.cv_lcb_transition_gap_vs_raw),
        "cv_lcb_delta_cosine_gap_vs_raw": float(selection.cv_lcb_delta_cosine_gap_vs_raw),
        "cv_lcb_recall_gap_vs_raw": float(selection.cv_lcb_recall_gap_vs_raw),
        "mean_transition_gap_vs_raw": float(selection.mean_transition_gap_vs_raw),
        "mean_delta_cosine_gap_vs_raw": float(selection.mean_delta_cosine_gap_vs_raw),
        "mean_recall_gap_vs_raw": float(selection.mean_recall_gap_vs_raw),
        "train_raw_transition_improvement": raw_train_metrics["transition_source_cosine_improvement"],
        "train_raw_delta_cosine": raw_train_metrics["delta_cosine"],
        "train_raw_recall_at_1": raw_train_metrics.get("transition_to_target_recall@1", 0.0),
        "train_raw_delta_rank": raw_train_metrics["delta_prediction_effective_rank"],
        "train_calibrated_transition_improvement": blend_train_metrics["transition_source_cosine_improvement"],
        "train_calibrated_delta_cosine": blend_train_metrics["delta_cosine"],
        "train_calibrated_recall_at_1": blend_train_metrics.get("transition_to_target_recall@1", 0.0),
        "train_calibrated_delta_rank": blend_train_metrics["delta_prediction_effective_rank"],
        "train_calibrated_magnitude_ratio": blend_train_metrics["delta_magnitude_ratio"],
    }
    return {"fit": full_fit, "selection": selection_payload, "train_metrics": selection_payload}


def _evaluate_split_safe_jepa_calibration_gate(
    model,
    train_table: dict[str, Any],
    eval_table: dict[str, Any],
    *,
    fit: Any,
    blend_alpha: float,
    train_metrics: dict[str, float],
    selection: dict[str, float],
    device: str,
) -> dict[str, float]:
    raw = _evaluate_source_delta_rank_program_jepa(model, train_table, eval_table, device=device)
    eval_encoded = _encode_program_jepa_table(model, eval_table, device=device)
    raw_delta = eval_encoded["transition"] - eval_table["source_pca_target"].astype(np.float64)
    calibrated_delta = predict_ridge_numpy(fit, raw_delta)
    blend_delta = (1.0 - float(blend_alpha)) * raw_delta + float(blend_alpha) * calibrated_delta
    image_bundle = _table_bundle(eval_table, np.arange(raw_delta.shape[0]), name="f094_image_teacher_eval")
    blend_metrics = bundle_transition_metrics(image_bundle, blend_delta)
    wrong_action = np.roll(eval_table["action"], shift=1, axis=0)
    wrong_transition = _predict_program_jepa_transition(model, eval_table["control_expression"], wrong_action, device=device)
    wrong_raw_delta = wrong_transition - eval_table["source_pca_target"].astype(np.float64)
    wrong_calibrated_delta = predict_ridge_numpy(fit, wrong_raw_delta)
    wrong_blend_delta = (1.0 - float(blend_alpha)) * wrong_raw_delta + float(blend_alpha) * wrong_calibrated_delta
    wrong_metrics = bundle_transition_metrics(image_bundle, wrong_blend_delta)
    return {
        **raw,
        **train_metrics,
        "selected_blend_alpha": float(blend_alpha),
        "gate_status_code": 1.0 if selection.get("gate_status") == "JEPA_GATE_SELECTED_BLEND" else 0.0,
        "gate_cv_lcb_transition_gap_vs_raw": float(selection.get("cv_lcb_transition_gap_vs_raw", 0.0)),
        "gate_cv_lcb_delta_cosine_gap_vs_raw": float(selection.get("cv_lcb_delta_cosine_gap_vs_raw", 0.0)),
        "gate_cv_lcb_recall_gap_vs_raw": float(selection.get("cv_lcb_recall_gap_vs_raw", 0.0)),
        "calibrated_transition_improvement": blend_metrics["transition_source_cosine_improvement"],
        "calibrated_delta_cosine": blend_metrics["delta_cosine"],
        "calibrated_recall_at_1": blend_metrics.get("transition_to_target_recall@1", 0.0),
        "calibrated_delta_rank": blend_metrics["delta_prediction_effective_rank"],
        "calibrated_magnitude_ratio": blend_metrics["delta_magnitude_ratio"],
        "calibrated_action_negative_gap": blend_metrics["transition_source_cosine_improvement"]
        - wrong_metrics["transition_source_cosine_improvement"],
        "calibrated_delta_cosine_gain": blend_metrics["delta_cosine"] - raw["image_teacher_delta_cosine"],
        "calibrated_transition_gain": blend_metrics["transition_source_cosine_improvement"]
        - raw["image_teacher_transition_improvement"],
        "calibrated_recall_gain": blend_metrics.get("transition_to_target_recall@1", 0.0) - raw["image_teacher_recall_at_1"],
    }


def _table_bundle(table: dict[str, Any], indices: np.ndarray, *, name: str) -> LatentOperatorBundle:
    indices = np.asarray(indices, dtype=int)
    return LatentOperatorBundle(
        name=name,
        arrays={
            "source_z_bio_teacher": table["source_pca_target"].astype(np.float64)[indices],
            "target_z_bio_teacher": table["pca_target"].astype(np.float64)[indices],
            "action_descriptor": table["action"].astype(np.float64)[indices],
        },
        metadata=table["metadata"].iloc[indices].reset_index(drop=True),
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
    f082 = per_seed.rename(
        columns={
            "calibrated_transition_improvement": "transition_improvement",
            "calibrated_delta_cosine": "delta_cosine",
            "calibrated_recall_at_1": "recall_at_1",
            "calibrated_delta_rank": "delta_rank",
            "calibrated_magnitude_ratio": "magnitude_ratio",
        }
    )
    f082 = f082[
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
    all_rows = pd.concat([f082, baselines], ignore_index=True, sort=False)
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


def _external_decision(summary: pd.DataFrame, preflight: dict[str, Any], *, candidate_method: str) -> str:
    if not preflight["obs_contract"]["all_required_obs_columns_present"]:
        return "FAIL_OBS_CONTRACT_NON_PROMOTING"
    if not preflight["split_pairing"]["all_expected_splits_present"]:
        return "FAIL_SPLIT_PAIRING_NON_PROMOTING"
    f082 = summary[summary["method"].eq(candidate_method)]
    eval_splits = {"validation", "test", "alternate_test"}
    if set(f082["split"]) != eval_splits:
        return "FAIL_MISSING_EVAL_SPLITS_NON_PROMOTING"
    no_leak = bool(f082["max_identity_violation"].fillna(0.0).le(0.0).all() and f082["max_leakage_flag"].fillna(0.0).le(0.0).all())
    beats_floor = bool(
        f082["floor_gap_transition_improvement"].gt(0.0).all()
        and f082["floor_gap_delta_cosine"].ge(0.0).all()
        and f082["floor_gap_recall_at_1"].ge(0.0).all()
    )
    positive = bool(f082["mean_transition_improvement"].gt(0.0).all() and f082["mean_delta_cosine"].gt(0.0).all())
    if no_leak and beats_floor and positive:
        return "PASS_EXTERNAL_TIER3_NON_PROMOTING"
    return "FAIL_EXTERNAL_NO_PROMOTION"


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    preflight = payload["preflight"]
    summary = pd.DataFrame(payload.get("summary_metrics", []))
    baselines = pd.DataFrame(payload.get("baseline_metrics", []))
    candidate_method = payload.get("candidate_method", "F082_delta_calibrated")
    gate_mode = payload.get("gate_mode", "calibrated")
    summary_tsv = summary.to_csv(sep="\t", index=False) if not summary.empty else "No model metrics.\n"
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
    baseline_tsv = baseline_summary.to_csv(sep="\t", index=False) if not baseline_summary.empty else "No baseline metrics.\n"
    gate_selection = pd.DataFrame(payload.get("gate_selection", []))
    gate_tsv = gate_selection.to_csv(sep="\t", index=False) if not gate_selection.empty else "No gate selection rows.\n"
    split_tsv = pd.DataFrame(preflight["split_pairing"]["rows"]).to_csv(sep="\t", index=False)
    obs_tsv = pd.DataFrame(preflight["obs_contract"]["rows"]).to_csv(sep="\t", index=False)
    descriptor = preflight["descriptor_summary"]
    text = f"""# F082 scGeneScope External Validation Report

## Decision
`{payload['decision']}`

No model is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record unless a later real Tier 3 pass explicitly supersedes it.

## Scope
- candidate: `F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`
- candidate method: `{candidate_method}`
- model path: `ProgramBootstrapJEPA` with train-only delta calibration/gating mode `{gate_mode}`
- external validator: scGeneScope feature H5ADs, RNA `scvi_n200`, imaging `imagenet/vit-l`
- action descriptor: {descriptor.get('action_descriptor', '')}
- action dim: `{descriptor.get('action_dim', 0)}`
- missing non-control descriptor rows: `{descriptor.get('missing_noncontrol_descriptor_rows', 0)}`

## Backed Obs Contract
```tsv
{obs_tsv}
```

## Split And Pairing
Replicate mapping is fixed as `3=train`, `5=validation`, `4=test`, and `1/2=alternate_test`.

```tsv
{split_tsv}
```

## Candidate And Floor Comparison
```tsv
{summary_tsv}
```

## Baselines
```tsv
{baseline_tsv}
```

## Gate Selection
```tsv
{gate_tsv}
```

## Leakage And Identity Checks
- `condition_key`, `biological_key`, exact treatment one-hot, held-out target means, and pooled train+test statistics were not used as model inputs.
- Treatment labels were used for grouping, pairing, and retrieval relevance only.
- Image PCA and delta calibration were fit on train split only.
- PLS/full-ridge outputs are audit floors/baselines only; they are not the JEPA representation path.
- Raw data and cache files remain under ignored `data/raw/` and `outputs/` paths.
"""
    path.write_text(text, encoding="utf-8")


def _descriptor_summary(descriptor_frame: pd.DataFrame, table_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(table_payload.get("action_summary", {}))
    summary["pubchem_found_treatments"] = int(descriptor_frame["pubchem_found"].astype(bool).sum()) if "pubchem_found" in descriptor_frame else 0
    summary["total_treatments"] = int(descriptor_frame.shape[0])
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__") and value.__class__.__name__ == "ProgramBootstrapJEPAConfig":
        return _jsonable(value.__dict__)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.DataFrame,)):
        return value.to_dict("records")
    return value


def _normalize_integerish(value: object) -> str:
    text = normalize_value(value)
    match = re.search(r"\d+", text)
    return str(int(match.group(0))) if match else text


def _floatish(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        if value.strip().lower() in {"", "nan", "none", "false"}:
            return 0.0
        if value.strip().lower() == "true":
            return 1.0
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def _boolish(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_control(value: object) -> bool:
    token = normalize_value(value).strip().lower()
    return token in CONTROL_TOKENS or "dmso" in token


if __name__ == "__main__":
    raise SystemExit(main())
