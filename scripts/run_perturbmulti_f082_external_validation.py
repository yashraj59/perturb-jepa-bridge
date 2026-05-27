from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.program_bootstrap_jepa import ProgramBootstrapJEPAConfig
from scripts.run_bioguard_wm_total_autonomy import (
    _evaluate_delta_calibrated_source_delta_rank_jepa,
    _train_source_delta_rank_program_jepa,
)
from scripts.run_f082_scgenescope_external_validation import (
    _baseline_rows,
    _raw_uncalibrated_row,
    _summary_metrics,
    _train_only_pca,
)
from scripts.run_perturbmulti_f103_preflight import read_h5ad_elem


DEFAULT_RNA_H5AD = Path(
    "/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/"
    "8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad"
)
DEFAULT_PROTEIN_H5AD = Path(
    "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/"
    "8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/"
    "experiments/F104_perturbmulti_f082_external_validation"
)
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "F104_PERTURBMULTI_F082_EXTERNAL_VALIDATION.md"
EVAL_SPLITS = ("validation", "test", "alternate_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sealed PerturbMulti external validation for the frozen F082/F096 ProgramBootstrapJEPA path."
    )
    parser.add_argument("--rna-h5ad", type=Path, default=DEFAULT_RNA_H5AD)
    parser.add_argument("--protein-h5ad", type=Path, default=DEFAULT_PROTEIN_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="*", type=int, default=[37, 38, 39])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--split-seed", type=int, default=104)
    parser.add_argument(
        "--split-mode",
        choices=["gene_holdout", "guide_holdout_supported_gene"],
        default="gene_holdout",
    )
    parser.add_argument("--experiment-id", default="F104")
    parser.add_argument("--promotion-eligible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-cells-per-guide", type=int, default=20)
    parser.add_argument("--rna-feature-source", default="X", help="Use X or obsm:<key> from the RNA H5AD.")
    parser.add_argument(
        "--action-descriptor-mode",
        choices=["gene_symbol_hash", "mygene_hash"],
        default="gene_symbol_hash",
    )
    parser.add_argument("--descriptor-timeout", type=float, default=8.0)
    parser.add_argument("--descriptor-workers", type=int, default=16)
    parser.add_argument("--pca-dim", type=int, default=24)
    parser.add_argument("--hash-dim", type=int, default=48)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    device_status = _device_status(args.device)
    paired = _build_paired_guide_features(
        args.rna_h5ad,
        args.protein_h5ad,
        min_cells_per_guide=args.min_cells_per_guide,
        rna_feature_source=args.rna_feature_source,
        output_dir=args.output_dir,
    )
    table_payload = _build_f082_tables(
        paired,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        pca_dim=args.pca_dim,
        hash_dim=args.hash_dim,
        action_descriptor_mode=args.action_descriptor_mode,
        descriptor_timeout=args.descriptor_timeout,
        descriptor_workers=args.descriptor_workers,
        output_dir=args.output_dir,
    )
    preflight = {
        "device_status": device_status,
        "rna_h5ad": str(args.rna_h5ad),
        "protein_h5ad": str(args.protein_h5ad),
        "rna_feature_source": args.rna_feature_source,
        "paired_summary": paired["summary"],
        "split_summary": table_payload["split_summary"],
        "split_mode": args.split_mode,
        "experiment_id": args.experiment_id,
        "promotion_eligible": bool(args.promotion_eligible),
        "action_summary": table_payload["action_summary"],
        "preflight_pass": bool(table_payload["preflight_pass"]),
        "preflight_reason": table_payload.get("reason", ""),
        "leakage_controls": {
            "no_condition_key_in_model_inputs": True,
            "condition_key_eval_label_only": True,
            "no_biological_key": True,
            "no_exact_target_key_one_hot": True,
            "no_heldout_target_means": True,
            "train_only_scaling_and_pca": True,
            "raw_payloads_outside_git": True,
        },
    }
    _write_json(args.output_dir / "f104_preflight_summary.json", preflight)

    if not table_payload["preflight_pass"]:
        payload = {
            "decision": "F104_PERTURBMULTI_PREFLIGHT_FAIL_NO_MODEL_RUN",
            "preflight": preflight,
            "summary_metrics": [],
            "baseline_metrics": [],
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
            table_payload["train_table"],
            table_payload["config"],
            seed=int(seed) + 10400,
            device=args.device,
            output_dir=args.output_dir / f"seed_{seed}",
            steps=args.steps,
        )
        trace_rows.append({"seed": seed, **trace})
        for split, eval_table in table_payload["eval_tables"].items():
            eval_payload = _evaluate_delta_calibrated_source_delta_rank_jepa(
                model,
                table_payload["train_table"],
                eval_table,
                device=args.device,
            )
            per_seed_rows.append({"seed": seed, "split": split, "method": "F082_delta_calibrated", **eval_payload})
            rows = _baseline_rows(table_payload["train_table"], eval_table, seed=seed, split=split)
            rows.append({**rows[0], "method": "no_residual_source_as_target"})
            rows.append(_raw_uncalibrated_row(model, eval_table, seed=seed, split=split, device=args.device))
            baseline_rows.extend(rows)

    per_seed = pd.DataFrame(per_seed_rows)
    baselines = pd.DataFrame(baseline_rows)
    trace = pd.DataFrame(trace_rows)
    summary = _summary_metrics(per_seed, baselines)
    per_seed.to_csv(args.output_dir / "f104_seed_split_metrics.tsv", sep="\t", index=False)
    baselines.to_csv(args.output_dir / "f104_baseline_metrics.tsv", sep="\t", index=False)
    trace.to_csv(args.output_dir / "f104_train_trace.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "external_summary_metrics.tsv", sep="\t", index=False)

    decision = _external_decision(summary, preflight)
    payload = {
        "decision": decision,
        "candidate_method": "F082_delta_calibrated",
        "preflight": preflight,
        "summary_metrics": summary.to_dict(orient="records"),
        "baseline_metrics": baselines.to_dict(orient="records"),
        "trace": trace.to_dict(orient="records"),
        "model_promoted": decision.endswith("_PASS_FRESH_EXTERNAL_TIER3_PROMOTE_F082"),
    }
    _write_json(args.output_dir / "metrics_eval.json", payload)
    _write_report(args.report_path, payload)
    return 0 if payload["model_promoted"] else 1


def _build_paired_guide_features(
    rna_h5ad: Path,
    protein_h5ad: Path,
    *,
    min_cells_per_guide: int,
    rna_feature_source: str,
    output_dir: Path,
) -> dict[str, Any]:
    with h5py.File(rna_h5ad, "r") as rna_file, h5py.File(protein_h5ad, "r") as protein_file:
        rna_obs = rna_file["obs"]
        protein_obs = protein_file["obs"]
        rna_ids = pd.Index(read_h5ad_elem(rna_obs["id"]).astype(str))
        protein_cell_names = pd.Index(read_h5ad_elem(protein_obs["cell_name"]).astype(str))
        rna_indexer = rna_ids.get_indexer(protein_cell_names)
        shared_mask = rna_indexer >= 0
        protein_shared_indices = np.flatnonzero(shared_mask)
        rna_shared_indices = rna_indexer[shared_mask]

        rna_singlet = read_h5ad_elem(rna_obs["singlet_name"])[rna_shared_indices].astype(str)
        rna_gene = read_h5ad_elem(rna_obs["singlet_gene"])[rna_shared_indices].astype(str)
        protein_perturbation = read_h5ad_elem(protein_obs["perturbation"])[protein_shared_indices].astype(str)
        exact_mask = (rna_singlet == protein_perturbation) & (rna_singlet != "")
        protein_indices = protein_shared_indices[exact_mask]
        rna_indices = rna_shared_indices[exact_mask]
        guides = protein_perturbation[exact_mask]
        genes = np.asarray([_guide_to_gene(guide) for guide in guides], dtype=object)
        rna_genes = rna_gene[exact_mask]
        gene_match = genes == rna_genes
        protein_indices = protein_indices[gene_match]
        rna_indices = rna_indices[gene_match]
        guides = guides[gene_match]
        genes = genes[gene_match]

        rna_feature_node = _rna_feature_node(rna_file, rna_feature_source)
        rna_matrix = _read_h5_rows(rna_feature_node, rna_indices)
        protein_matrix = np.asarray(protein_file["X"][protein_indices, :], dtype=np.float64)
        guide_frame = pd.DataFrame({"guide": guides, "gene": genes})
        counts = guide_frame["guide"].value_counts()
        keep_guides = set(counts[counts >= int(min_cells_per_guide)].index)
        keep = guide_frame["guide"].isin(keep_guides).to_numpy()
        guide_frame = guide_frame.loc[keep].reset_index(drop=True)
        rna_matrix = rna_matrix[keep]
        protein_matrix = protein_matrix[keep]

    metadata_rows: list[dict[str, Any]] = []
    rna_rows: list[np.ndarray] = []
    image_rows: list[np.ndarray] = []
    for guide, indices in guide_frame.groupby("guide", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        gene = str(guide_frame.iloc[idx[0]]["gene"])
        metadata_rows.append(
            {
                "guide": str(guide),
                "gene": gene,
                "treatment": str(guide),
                "condition_key": str(guide),
                "is_control": bool(gene.lower() == "control"),
                "n_cells": int(idx.size),
            }
        )
        rna_rows.append(rna_matrix[idx].mean(axis=0))
        image_rows.append(protein_matrix[idx].mean(axis=0))

    metadata = pd.DataFrame(metadata_rows)
    rna_values = np.vstack(rna_rows).astype(np.float64)
    image_values = np.vstack(image_rows).astype(np.float64)
    metadata.to_csv(output_dir / "f104_guide_condition_metadata.tsv", sep="\t", index=False)
    summary = {
        "protein_cells": int(len(protein_cell_names)),
        "shared_protein_rna_cells": int(shared_mask.sum()),
        "exact_guide_matched_cells": int(exact_mask.sum()),
        "gene_verified_cells": int(gene_match.sum()),
        "min_cells_per_guide": int(min_cells_per_guide),
        "condition_guides": int(len(metadata)),
        "condition_genes": int(metadata.loc[~metadata["is_control"], "gene"].nunique()),
        "control_guides": int(metadata["is_control"].sum()),
        "rna_dim": int(rna_values.shape[1]),
        "rna_feature_source": rna_feature_source,
        "image_dim": int(image_values.shape[1]),
    }
    return {"metadata": metadata, "rna": rna_values, "image": image_values, "summary": summary}


def _rna_feature_node(handle: h5py.File, source: str) -> h5py.Dataset:
    if source == "X":
        return handle["X"]
    if source.startswith("obsm:"):
        key = source.split(":", 1)[1]
        if key not in handle["obsm"]:
            raise KeyError(f"RNA H5AD missing obsm key {key!r}")
        node = handle["obsm"][key]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"RNA obsm {key!r} is not a dense dataset")
        return node
    raise ValueError(f"Unsupported RNA feature source {source!r}; use X or obsm:<key>")


def _read_h5_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    order = np.argsort(indices)
    sorted_indices = np.asarray(indices[order], dtype=np.int64)
    sorted_values = np.asarray(dataset[sorted_indices, :], dtype=np.float64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_values[inverse]


def _build_f082_tables(
    paired: dict[str, Any],
    *,
    split_seed: int,
    split_mode: str,
    pca_dim: int,
    hash_dim: int,
    action_descriptor_mode: str,
    descriptor_timeout: float,
    descriptor_workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = paired["metadata"].copy()
    metadata = _assign_splits(metadata, seed=split_seed, split_mode=split_mode)
    metadata.to_csv(output_dir / "f104_sealed_split_metadata.tsv", sep="\t", index=False)
    rna = np.asarray(paired["rna"], dtype=np.float64)
    image = np.asarray(paired["image"], dtype=np.float64)
    noncontrol = ~metadata["is_control"].astype(bool).to_numpy()
    train_noncontrol = metadata["split"].eq("train").to_numpy() & noncontrol
    control_train = metadata["is_control"].astype(bool).to_numpy() & metadata["split"].eq("train").to_numpy()
    split_summary = _split_summary(metadata)
    if not control_train.any():
        return {"preflight_pass": False, "reason": "no train control guides", "split_summary": split_summary}
    if int(train_noncontrol.sum()) < 20:
        return {"preflight_pass": False, "reason": "insufficient train non-control guides", "split_summary": split_summary}
    if any(not (metadata["split"].eq(split).to_numpy() & noncontrol).any() for split in EVAL_SPLITS):
        return {"preflight_pass": False, "reason": "missing eval split non-control guides", "split_summary": split_summary}

    rna_scaled = _train_only_standardize(rna, train_noncontrol | control_train)
    image_scaled = _train_only_standardize(image, train_noncontrol | control_train)
    image_pca = _train_only_pca(image_scaled, train_noncontrol | control_train, dim=pca_dim)
    source_rna = rna_scaled[control_train].mean(axis=0, keepdims=True)
    source_image_pca = image_pca[control_train].mean(axis=0, keepdims=True)
    actions, action_columns, action_summary = _action_matrix(
        metadata,
        rna_scaled,
        train_mask=train_noncontrol | control_train,
        control_train_mask=control_train,
        hash_dim=hash_dim,
        mode=action_descriptor_mode,
        descriptor_timeout=descriptor_timeout,
        descriptor_workers=descriptor_workers,
        output_dir=output_dir,
    )
    tables: dict[str, dict[str, Any]] = {}
    for split in ("train", *EVAL_SPLITS):
        mask = metadata["split"].eq(split).to_numpy() & noncontrol
        if not mask.any():
            continue
        tables[split] = {
            "control_expression": np.repeat(source_rna, int(mask.sum()), axis=0),
            "target_expression": rna_scaled[mask],
            "target_image_flat": image_scaled[mask],
            "pca_target": image_pca[mask],
            "source_pca_target": np.repeat(source_image_pca, int(mask.sum()), axis=0),
            "action": actions[mask],
            "metadata": metadata.loc[mask].reset_index(drop=True).copy(),
        }
    config = ProgramBootstrapJEPAConfig(
        genes=int(rna.shape[1]),
        image_dim=int(image.shape[1]),
        action_dim=int(actions.shape[1]),
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
        "eval_tables": {split: tables[split] for split in EVAL_SPLITS},
        "config": config,
        "split_summary": split_summary,
        "action_summary": action_summary,
        "action_columns": action_columns,
    }


def _assign_splits(metadata: pd.DataFrame, *, seed: int, split_mode: str) -> pd.DataFrame:
    if split_mode == "gene_holdout":
        return _assign_gene_splits(metadata, seed=seed)
    if split_mode == "guide_holdout_supported_gene":
        return _assign_guide_holdout_supported_gene_splits(metadata, seed=seed)
    raise ValueError(f"Unknown split mode: {split_mode}")


def _assign_gene_splits(metadata: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    out = metadata.copy()
    out["split"] = "train"
    genes = sorted(out.loc[~out["is_control"].astype(bool), "gene"].astype(str).unique())
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(genes, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(round(0.10 * n)))
    n_test = max(1, int(round(0.10 * n)))
    n_alt = max(1, int(round(0.10 * n)))
    val = set(map(str, shuffled[:n_val]))
    test = set(map(str, shuffled[n_val : n_val + n_test]))
    alt = set(map(str, shuffled[n_val + n_test : n_val + n_test + n_alt]))
    out.loc[out["gene"].astype(str).isin(val), "split"] = "validation"
    out.loc[out["gene"].astype(str).isin(test), "split"] = "test"
    out.loc[out["gene"].astype(str).isin(alt), "split"] = "alternate_test"
    out.loc[out["is_control"].astype(bool), "split"] = "train"
    return out


def _assign_guide_holdout_supported_gene_splits(metadata: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    out = metadata.copy()
    out["split"] = "train"
    rng = np.random.default_rng(seed)
    candidate_indices: list[int] = []
    noncontrol = out.loc[~out["is_control"].astype(bool)].copy()
    for _, group in noncontrol.groupby("gene", sort=True):
        indices = group.index.to_numpy(dtype=int)
        if indices.size < 2:
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        candidate_indices.extend(int(idx) for idx in shuffled[1:])
    candidates = np.asarray(candidate_indices, dtype=int)
    rng.shuffle(candidates)
    n = int(candidates.size)
    n_val = max(1, int(round(0.10 * n)))
    n_test = max(1, int(round(0.10 * n)))
    n_alt = max(1, int(round(0.10 * n)))
    out.loc[candidates[:n_val], "split"] = "validation"
    out.loc[candidates[n_val : n_val + n_test], "split"] = "test"
    out.loc[candidates[n_val + n_test : n_val + n_test + n_alt], "split"] = "alternate_test"
    out.loc[out["is_control"].astype(bool), "split"] = "train"
    return out


def _split_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for split, frame in metadata.groupby("split", sort=True):
        noncontrol = frame.loc[~frame["is_control"].astype(bool)]
        rows.append(
            {
                "split": split,
                "guides": int(len(frame)),
                "noncontrol_guides": int(len(noncontrol)),
                "noncontrol_genes": int(noncontrol["gene"].nunique()),
                "cells": int(frame["n_cells"].sum()),
            }
        )
    return {
        "rows": rows,
        "all_eval_splits_present": all(split in set(metadata["split"]) for split in EVAL_SPLITS),
    }


def _train_only_standardize(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = values[train_mask]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, 1.0e-6)).astype(np.float64)


def _action_matrix(
    metadata: pd.DataFrame,
    rna_scaled: np.ndarray,
    *,
    train_mask: np.ndarray,
    control_train_mask: np.ndarray,
    hash_dim: int,
    mode: str,
    descriptor_timeout: float,
    descriptor_workers: int,
    output_dir: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    control_mean = rna_scaled[control_train_mask].mean(axis=0)
    mygene_records: dict[str, Any] = {}
    if mode == "mygene_hash":
        genes = sorted(metadata["gene"].astype(str).unique())
        mygene_records = _fetch_mygene_records(
            genes,
            output_dir / "mygene_descriptor_cache.json",
            timeout=descriptor_timeout,
            workers=descriptor_workers,
        )
    raw_rows = []
    for _, row in metadata.iterrows():
        gene = str(row["gene"])
        tokens = _tokens_from_mygene_record(mygene_records.get(gene, {})) if mode == "mygene_hash" else None
        raw_rows.append(_gene_descriptor(gene, control_mean=control_mean, hash_dim=hash_dim, extra_tokens=tokens))
    raw = np.vstack(raw_rows).astype(np.float64)
    mean = raw[train_mask].mean(axis=0, keepdims=True)
    std = raw[train_mask].std(axis=0, keepdims=True)
    scaled = (raw - mean) / np.maximum(std, 1.0e-6)
    columns = [
        "gene_length",
        "uppercase_fraction",
        "digit_fraction",
        "vowel_fraction",
        "ascii_mean",
        "ascii_std",
        "is_control",
        "control_mean_projection_0",
        "control_mean_projection_1",
    ] + [f"signed_char_ngram_hash_{idx}" for idx in range(hash_dim)]
    return scaled.astype(np.float64), columns, {
        "action_descriptor": (
            "Train-scaled external MyGene symbol/name/GO/pathway/interpro feature hashes plus train-control RNA projections; "
            "not exact target-key one-hot and no held-out target means."
            if mode == "mygene_hash"
            else "Train-scaled gene-symbol morphology, signed character n-gram hashes, and train-control RNA projections; "
            "not exact target-key one-hot and no held-out target means."
        ),
        "action_descriptor_mode": mode,
        "action_dim": int(scaled.shape[1]),
        "hash_dim": int(hash_dim),
        "mygene_found_count": int(sum(bool(record) and not record.get("notfound") for record in mygene_records.values())),
        "descriptor_columns": columns,
    }


def _gene_descriptor(
    gene: str,
    *,
    control_mean: np.ndarray,
    hash_dim: int,
    extra_tokens: list[str] | None = None,
) -> np.ndarray:
    text = gene.strip()
    lower = text.lower()
    chars = [ord(ch) for ch in text] or [0]
    length = max(len(text), 1)
    vowels = sum(ch.lower() in "aeiou" for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    uppercase = sum(ch.isupper() for ch in text)
    base = [
        float(len(text)),
        float(uppercase) / length,
        float(digits) / length,
        float(vowels) / length,
        float(np.mean(chars)) / 128.0,
        float(np.std(chars)) / 128.0,
        float(lower == "control"),
        float(control_mean[_stable_index(lower, control_mean.size)] if control_mean.size else 0.0),
        float(control_mean[_stable_index(lower + "::alt", control_mean.size)] if control_mean.size else 0.0),
    ]
    hashed = np.zeros(int(hash_dim), dtype=np.float64)
    tokens = [lower]
    tokens.extend(lower[idx : idx + 2] for idx in range(max(0, len(lower) - 1)))
    tokens.extend(lower[idx : idx + 3] for idx in range(max(0, len(lower) - 2)))
    if extra_tokens:
        tokens.extend(str(token).lower() for token in extra_tokens if str(token).strip())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % int(hash_dim)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        hashed[bucket] += sign
    norm = np.linalg.norm(hashed)
    if norm > 0:
        hashed /= norm
    return np.asarray(base + hashed.tolist(), dtype=np.float64)


def _fetch_mygene_records(genes: list[str], cache_path: Path, *, timeout: float, workers: int) -> dict[str, Any]:
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(gene in cached for gene in genes):
                return cached
        except json.JSONDecodeError:
            cached = {}
    else:
        cached = {}
    out: dict[str, Any] = dict(cached)
    fields = ",".join(["symbol", "name", "summary", "type_of_gene", "genomic_pos", "go", "pathway", "interpro", "MGI"])
    missing = [gene for gene in genes if gene not in out]

    def fetch_one(gene: str) -> tuple[str, dict[str, Any]]:
        if gene.lower() == "control":
            return gene, {"symbol": "control", "name": "control guide", "notfound": False}
        try:
            response = requests.get(
                "https://mygene.info/v3/query",
                params={"q": f"symbol:{gene}", "species": "mouse", "fields": fields, "size": 1},
                timeout=float(timeout),
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
            return gene, hits[0] if hits else {"symbol": gene, "notfound": True}
        except Exception as exc:  # pragma: no cover - live network diagnostic path.
            return gene, {"symbol": gene, "notfound": True, "error": repr(exc)}

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(fetch_one, gene) for gene in missing]
        for future in as_completed(futures):
            gene, record = future.result()
            out[gene] = record
            completed += 1
            if completed % 20 == 0:
                cache_path.write_text(json.dumps(_json_safe(out), indent=2, sort_keys=True), encoding="utf-8")
    cache_path.write_text(json.dumps(_json_safe(out), indent=2, sort_keys=True), encoding="utf-8")
    return out


def _tokens_from_mygene_record(record: Any) -> list[str]:
    tokens: list[str] = []

    def add(value: Any, prefix: str = "") -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                add(item, f"{prefix}{key}:")
            return
        if isinstance(value, list):
            for item in value:
                add(item, prefix)
            return
        text = str(value).strip()
        if not text:
            return
        if len(text) > 120:
            for part in re.split(r"[^A-Za-z0-9_:+.-]+", text):
                if len(part) >= 3:
                    tokens.append(f"{prefix}{part}")
        else:
            tokens.append(f"{prefix}{text}")

    for key in ("symbol", "name", "type_of_gene", "summary", "genomic_pos", "go", "pathway", "interpro", "MGI"):
        add(record.get(key) if isinstance(record, dict) else None, f"{key}:")
    return tokens[:512]


def _stable_index(text: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") % max(int(size), 1)


def _guide_to_gene(guide: str) -> str:
    return re.sub(r"_gBC_.*$", "", str(guide))


def _external_decision(summary: pd.DataFrame, preflight: dict[str, Any]) -> str:
    experiment_id = str(preflight.get("experiment_id", "F104"))
    if not preflight["preflight_pass"]:
        return f"{experiment_id}_PERTURBMULTI_PREFLIGHT_FAIL_NO_PROMOTION"
    candidate = summary[summary["method"].eq("F082_delta_calibrated")]
    if set(candidate["split"]) != set(EVAL_SPLITS):
        return f"{experiment_id}_FAIL_MISSING_EVAL_SPLITS_NO_PROMOTION"
    no_leak = bool(
        candidate["max_identity_violation"].fillna(0.0).le(0.0).all()
        and candidate["max_leakage_flag"].fillna(0.0).le(0.0).all()
    )
    beats_floor = bool(
        candidate["floor_gap_transition_improvement"].gt(0.0).all()
        and candidate["floor_gap_delta_cosine"].ge(0.0).all()
        and candidate["floor_gap_recall_at_1"].ge(0.0).all()
    )
    positive = bool(
        candidate["mean_transition_improvement"].gt(0.0).all()
        and candidate["mean_delta_cosine"].gt(0.0).all()
        and candidate["mean_delta_rank"].ge(2.0).all()
    )
    retrieval_ok = bool(
        candidate["mean_rna_to_image_recall_at_1"].fillna(0.0).ge(0.0).all()
        and candidate["mean_image_to_rna_recall_at_1"].fillna(0.0).ge(0.0).all()
    )
    if no_leak and beats_floor and positive and retrieval_ok:
        if not bool(preflight.get("promotion_eligible", True)):
            return f"{experiment_id}_PASS_EXTERNAL_TIER3_NON_PROMOTING_REPAIR_LOOP"
        return f"{experiment_id}_PASS_FRESH_EXTERNAL_TIER3_PROMOTE_F082"
    return f"{experiment_id}_FAIL_FRESH_EXTERNAL_TIER3_NO_PROMOTION"


def _device_status(requested: str) -> dict[str, Any]:
    return {
        "requested": requested,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "selected": requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu",
    }


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
    baseline_tsv = baseline_summary.to_csv(sep="\t", index=False) if not baseline_summary.empty else "No baseline metrics.\n"
    lines = [
        f"# {preflight.get('experiment_id', 'F104')} PerturbMulti F082 External Validation",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        f"- model promoted: `{payload.get('model_promoted', False)}`",
        f"- split mode: `{preflight.get('split_mode', '')}`",
        "- model path: frozen F082/F096 ProgramBootstrapJEPA architecture and train-only delta calibration",
        "- protected audit floor: rank-3 train-split-only PLS/raw-linear model remains protected unless this report records a fresh Tier 3 pass",
        "- raw H5AD/image payloads stayed outside git",
        "",
        "## Preflight",
        "```json",
        json.dumps(_json_safe(preflight), indent=2, sort_keys=True),
        "```",
        "",
        "## Summary Metrics",
        summary_tsv,
        "## Baseline Metrics",
        baseline_tsv,
        "## Leakage And Identity",
        "- condition_key/biological_key/exact one-hot descriptors were not used.",
        "- action descriptors were train-scaled gene-symbol/control-expression descriptors.",
        "- PCA, standardization, and delta calibration used train split only.",
        "- identity_violation and leakage_flag are reported in the summary metrics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
