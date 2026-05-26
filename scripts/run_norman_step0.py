from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import (  # noqa: E402
    DEFAULT_GEARS_NORMAN_URL,
    DEFAULT_NORMAN_H5AD,
    add_gears_simulation_split,
    assert_no_combo_order_leakage,
    is_combo,
    is_single,
    load_norman2019_condition_data,
    perturbation_genes,
)


RUN_DIR = Path("outputs/autoresearch_norman_v1")
STEP0_DIRNAME = "step0_baselines"
PUBLISHED_BASELINES: tuple[dict[str, Any], ...] = (
    {
        "baseline": "gears_published",
        "display_name": "GEARS published",
        "source": "Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6",
        "metric_scope": "Norman 2019, five GEARS simulation splits, published aggregate",
        "mse": 0.216,
        "mse_std": 0.053,
        "pearson_de": 0.556,
        "pearson_de_std": 0.030,
        "top20_de_overlap": math.nan,
        "top20_de_overlap_note": "not reported in GEARS Supplementary Table 6",
    },
    {
        "baseline": "cpa_original_published",
        "display_name": "CPA Original published",
        "source": "Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6",
        "metric_scope": "Norman 2019, five GEARS simulation splits, published aggregate",
        "mse": 0.354,
        "mse_std": 0.049,
        "pearson_de": 0.440,
        "pearson_de_std": 0.036,
        "top20_de_overlap": math.nan,
        "top20_de_overlap_note": "not reported in GEARS Supplementary Table 6",
    },
    {
        "baseline": "cpa_kg_published",
        "display_name": "CPA + KG published",
        "source": "Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6",
        "metric_scope": "Norman 2019, five GEARS simulation splits, published aggregate",
        "mse": 0.333,
        "mse_std": 0.046,
        "pearson_de": 0.504,
        "pearson_de_std": 0.029,
        "top20_de_overlap": math.nan,
        "top20_de_overlap_note": "not reported in GEARS Supplementary Table 6",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Norman 2019 Step 0 condition-level baselines.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_NORMAN_H5AD)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--pls-rank", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = args.output_dir
    step0_dir = output_dir / STEP0_DIRNAME
    step0_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_norman2019_condition_data(args.h5ad)
    _write_gene_symbol_map(dataset, step0_dir / "gene_symbol_resolution.tsv")

    per_condition_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for seed in args.split_seeds:
        split_dataset = add_gears_simulation_split(dataset, seed=seed)
        if split_dataset.split is None:
            raise RuntimeError("GEARS split was not attached to NormanDataset")
        split = split_dataset.split
        assert_no_combo_order_leakage(split)
        split_rows.append(_split_summary_row(split))

        predictors = [
            ("global_train_mean", predict_global_train_mean(split_dataset)),
            ("single_perturbation_additive", predict_single_perturbation_additive(split_dataset)),
            ("family_n_condition_mean_table", predict_family_n_condition_mean_table(split_dataset)),
            ("closed_form_pls_perturbation_readout", predict_pls_perturbation_readout(split_dataset, rank=args.pls_rank)),
        ]
        for baseline, prediction_payload in predictors:
            rows, aggregates = evaluate_predictions(split_dataset, baseline, seed, prediction_payload)
            per_condition_rows.extend(rows)
            aggregate_rows.extend(aggregates)

    aggregate_df = pd.DataFrame(aggregate_rows)
    per_condition_df = pd.DataFrame(per_condition_rows)
    split_df = pd.DataFrame(split_rows)
    published_df = pd.DataFrame(PUBLISHED_BASELINES)

    per_condition_path = step0_dir / "per_condition_metrics.tsv"
    aggregate_path = step0_dir / "baseline_metrics.tsv"
    split_path = step0_dir / "split_summary.tsv"
    published_path = step0_dir / "published_baselines.tsv"
    per_condition_df.to_csv(per_condition_path, sep="\t", index=False)
    aggregate_df.to_csv(aggregate_path, sep="\t", index=False)
    split_df.to_csv(split_path, sep="\t", index=False)
    published_df.to_csv(published_path, sep="\t", index=False)

    commit = _git_commit()
    result_rows.extend(_result_rows_for_recomputed(commit, aggregate_df))
    result_rows.extend(_result_rows_for_published(commit))
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(output_dir / "results.tsv", sep="\t", index=False)

    metadata = {
        "h5ad_path": str(args.h5ad),
        "h5ad_size_bytes": args.h5ad.stat().st_size if args.h5ad.exists() else None,
        "source_url": DEFAULT_GEARS_NORMAN_URL,
        "split_seeds": args.split_seeds,
        "pls_rank": args.pls_rank,
        "conditions": len(dataset.conditions),
        "genes": len(dataset.gene_ids),
        "elapsed_seconds": time.perf_counter() - started,
        "commit": commit,
    }
    (step0_dir / "run_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    _write_summary(output_dir, metadata, split_df, aggregate_df)
    _write_registry(output_dir, metadata, split_df, aggregate_df)
    _write_journal(output_dir, metadata, aggregate_df)
    _write_architecture_log(output_dir)
    _write_family_allocation(output_dir)
    _write_papers_consulted(output_dir)
    _write_external_resources(output_dir, metadata)
    _write_identity_log(output_dir)
    _write_step0_review_packet(output_dir, metadata, aggregate_df)

    print(f"Wrote Norman Step 0 baselines to {step0_dir}")


def predict_global_train_mean(dataset: Any) -> dict[str, Any]:
    split = _require_split(dataset)
    train = [condition for condition in split.train_conditions if condition != dataset.ctrl_condition]
    if not train:
        mean = dataset.ctrl_mean
    else:
        mean = np.stack([dataset.mean_for(condition) for condition in train]).mean(axis=0)
    predictions = {condition: mean.copy() for condition in split.test_conditions}
    return {
        "predictions": predictions,
        "fit_metadata": {
            "train_conditions_used": len(train),
            "fallback": "none" if train else "control_mean",
            "description": "Mean expression vector across final train perturbation conditions.",
        },
    }


def predict_single_perturbation_additive(dataset: Any) -> dict[str, Any]:
    split = _require_split(dataset)
    single_delta_by_gene: dict[str, np.ndarray] = {}
    for condition in split.train_conditions:
        if is_single(condition):
            genes = perturbation_genes(condition)
            single_delta_by_gene[genes[0]] = dataset.delta_for(condition)

    fallback_mean = _train_condition_mean(dataset)
    predictions: dict[str, np.ndarray] = {}
    fallback_counts = {"exact_additive": 0, "global_train_mean": 0}
    for condition in split.test_conditions:
        genes = perturbation_genes(condition)
        if genes and all(gene in single_delta_by_gene for gene in genes):
            delta = np.stack([single_delta_by_gene[gene] for gene in genes]).sum(axis=0)
            predictions[condition] = dataset.ctrl_mean + delta
            fallback_counts["exact_additive"] += 1
        else:
            predictions[condition] = fallback_mean.copy()
            fallback_counts["global_train_mean"] += 1
    return {
        "predictions": predictions,
        "fit_metadata": {
            "train_single_genes": len(single_delta_by_gene),
            **fallback_counts,
            "description": "delta(A+B) := delta(A) + delta(B) using train singles only; missing singles fall back to train mean.",
        },
    }


def predict_family_n_condition_mean_table(dataset: Any) -> dict[str, Any]:
    split = _require_split(dataset)
    table = {condition: dataset.mean_for(condition).copy() for condition in split.train_conditions}
    global_mean = _train_condition_mean(dataset)
    predictions: dict[str, np.ndarray] = {}
    fallback_counts = {"exact_condition": 0, "global_train_mean": 0}
    for condition in split.test_conditions:
        if condition in table:
            predictions[condition] = table[condition].copy()
            fallback_counts["exact_condition"] += 1
        else:
            predictions[condition] = global_mean.copy()
            fallback_counts["global_train_mean"] += 1
    return {
        "predictions": predictions,
        "fit_metadata": {
            "train_condition_keys": len(table),
            **fallback_counts,
            "description": "Train-only condition mean lookup matching the Family N pattern; unseen keys fall back to train mean.",
        },
    }


def predict_pls_perturbation_readout(dataset: Any, *, rank: int) -> dict[str, Any]:
    split = _require_split(dataset)
    feature_genes = sorted({gene for condition in dataset.conditions for gene in perturbation_genes(condition)})
    train_conditions = [condition for condition in split.train_conditions if condition != dataset.ctrl_condition]
    fallback_mean = _train_condition_mean(dataset)
    if len(train_conditions) < 3 or not feature_genes:
        return {
            "predictions": {condition: fallback_mean.copy() for condition in split.test_conditions},
            "fit_metadata": {"fallback": "insufficient_train_conditions", "description": "PLS was not fit."},
        }

    x_train = _design_matrix(train_conditions, feature_genes)
    y_train = np.stack([dataset.delta_for(condition) for condition in train_conditions])
    n_components = max(1, min(int(rank), x_train.shape[0] - 1, x_train.shape[1], y_train.shape[1]))
    pls = PLSRegression(n_components=n_components, scale=True, max_iter=1000)
    pls.fit(x_train, y_train)
    x_test = _design_matrix(split.test_conditions, feature_genes)
    delta_predictions = pls.predict(x_test)
    predictions = {
        condition: dataset.ctrl_mean + np.asarray(delta_predictions[index], dtype=np.float32)
        for index, condition in enumerate(split.test_conditions)
    }
    return {
        "predictions": predictions,
        "fit_metadata": {
            "rank_requested": int(rank),
            "rank_used": int(n_components),
            "train_conditions_used": len(train_conditions),
            "feature_genes": len(feature_genes),
            "description": "Closed-form PLS regression from perturbation multi-hot design to pseudobulk delta.",
        },
    }


def evaluate_predictions(dataset: Any, baseline: str, split_seed: int, prediction_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = _require_split(dataset)
    predictions = prediction_payload["predictions"]
    fit_metadata = prediction_payload.get("fit_metadata", {})
    subsets = {
        "all_test": tuple(split.test_conditions),
        "exact_train_combo": tuple(split.exact_train_combo_conditions),
        "unseen_single": tuple(split.unseen_single_conditions),
    }
    per_condition_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for subset, conditions in subsets.items():
        pred_deltas = []
        true_deltas = []
        subset_rows = []
        for condition in conditions:
            if condition not in predictions:
                continue
            pred_expression = np.asarray(predictions[condition], dtype=np.float32)
            true_expression = dataset.mean_for(condition)
            pred_delta = pred_expression - dataset.ctrl_mean
            true_delta = true_expression - dataset.ctrl_mean
            de20 = dataset.de20_indices(condition)
            row = {
                "split_seed": split_seed,
                "baseline": baseline,
                "subset": subset,
                "condition": condition,
                "condition_label": dataset.perturbation_label_for(condition),
                "perturbation_ids": "+".join(dataset.perturbation_ids_for(condition)),
                "pearson_delta_all": _safe_pearson(pred_delta, true_delta),
                "pearson_delta_de20": _safe_pearson(pred_delta[de20], true_delta[de20]),
                "top20_de_overlap": _topk_overlap(pred_delta, de20, k=20),
                "mse_delta_all": float(np.mean((pred_delta - true_delta) ** 2)),
                "mse_delta_de20": float(np.mean((pred_delta[de20] - true_delta[de20]) ** 2)),
                "mse_expression_all": float(np.mean((pred_expression - true_expression) ** 2)),
                "mse_expression_de20": float(np.mean((pred_expression[de20] - true_expression[de20]) ** 2)),
                "direction_accuracy_de20": _direction_accuracy(pred_delta[de20], true_delta[de20]),
                "pred_delta_l2": float(np.linalg.norm(pred_delta)),
                "true_delta_l2": float(np.linalg.norm(true_delta)),
                "de20_count": int(len(de20)),
            }
            subset_rows.append(row)
            pred_deltas.append(pred_delta)
            true_deltas.append(true_delta)
        per_condition_rows.extend(subset_rows)
        aggregate_rows.append(_aggregate_rows(split_seed, baseline, subset, subset_rows, pred_deltas, true_deltas, fit_metadata))
    return per_condition_rows, aggregate_rows


def _aggregate_rows(
    split_seed: int,
    baseline: str,
    subset: str,
    rows: list[dict[str, Any]],
    pred_deltas: list[np.ndarray],
    true_deltas: list[np.ndarray],
    fit_metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "pearson_delta_all",
        "pearson_delta_de20",
        "top20_de_overlap",
        "mse_delta_all",
        "mse_delta_de20",
        "mse_expression_all",
        "mse_expression_de20",
        "direction_accuracy_de20",
    ]
    aggregate: dict[str, Any] = {
        "split_seed": split_seed,
        "baseline": baseline,
        "subset": subset,
        "condition_count": len(rows),
        "fit_metadata_json": json.dumps(_jsonable(fit_metadata), sort_keys=True),
    }
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows if not _is_nan(row[metric])], dtype=float)
        aggregate[f"{metric}_mean"] = float(values.mean()) if values.size else math.nan
        aggregate[f"{metric}_std"] = float(values.std(ddof=0)) if values.size else math.nan
    if pred_deltas and true_deltas:
        pred_var = float(np.mean(np.var(np.stack(pred_deltas), axis=0)))
        true_var = float(np.mean(np.var(np.stack(true_deltas), axis=0)))
        ratio = pred_var / true_var if true_var > 1e-12 else math.nan
        aggregate["prediction_delta_variance"] = pred_var
        aggregate["true_delta_variance"] = true_var
        aggregate["prediction_delta_variance_ratio"] = ratio
        aggregate["mean_collapse_flag"] = bool(ratio < 0.1) if not _is_nan(ratio) else False
    else:
        aggregate["prediction_delta_variance"] = math.nan
        aggregate["true_delta_variance"] = math.nan
        aggregate["prediction_delta_variance_ratio"] = math.nan
        aggregate["mean_collapse_flag"] = False
    return aggregate


def _design_matrix(conditions: Any, feature_genes: list[str]) -> np.ndarray:
    index = {gene: idx for idx, gene in enumerate(feature_genes)}
    rows = []
    for condition in conditions:
        row = np.zeros(len(feature_genes), dtype=np.float32)
        for gene in perturbation_genes(condition):
            if gene in index:
                row[index[gene]] = 1.0
        rows.append(row)
    return np.stack(rows)


def _train_condition_mean(dataset: Any) -> np.ndarray:
    split = _require_split(dataset)
    train = [condition for condition in split.train_conditions if condition != dataset.ctrl_condition]
    if not train:
        return dataset.ctrl_mean.copy()
    return np.stack([dataset.mean_for(condition) for condition in train]).mean(axis=0)


def _split_summary_row(split: Any) -> dict[str, Any]:
    return {
        "seed": split.seed,
        "train_gene_set_size": split.train_gene_set_size,
        "combo_seen2_train_frac": split.combo_seen2_train_frac,
        "train_conditions": len(split.train_conditions),
        "val_conditions": len(split.val_conditions),
        "test_conditions": len(split.test_conditions),
        "exact_train_combo": len(split.exact_train_combo_conditions),
        "unseen_single": len(split.unseen_single_conditions),
        "test_combo_seen0": len(split.test_subgroups.get("combo_seen0", ())),
        "test_combo_seen1": len(split.test_subgroups.get("combo_seen1", ())),
        "test_combo_seen2": len(split.test_subgroups.get("combo_seen2", ())),
        "test_unseen_single": len(split.test_subgroups.get("unseen_single", ())),
    }


def _result_rows_for_recomputed(commit: str, aggregate_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    baselines = [
        "global_train_mean",
        "single_perturbation_additive",
        "family_n_condition_mean_table",
        "closed_form_pls_perturbation_readout",
    ]
    for experiment_num, baseline in enumerate(baselines):
        exact = _aggregate_lookup(aggregate_df, baseline, "exact_train_combo")
        unseen = _aggregate_lookup(aggregate_df, baseline, "unseen_single")
        rows.append(
            {
                "commit": commit,
                "experiment_num": experiment_num,
                "family": "Step0",
                "tier_reached": "StageA",
                "status": "BASELINE_COMPLETE",
                "primary_metric": _format_metrics(
                    exact,
                    ["pearson_delta_all_mean", "pearson_delta_de20_mean"],
                    prefix="exact_train_combo",
                ),
                "secondary_metric": _format_metrics(
                    exact,
                    ["top20_de_overlap_mean", "mse_delta_all_mean", "mse_expression_de20_mean"],
                    prefix="exact_train_combo",
                ),
                "protected_metric_summary": _format_metrics(
                    unseen,
                    ["direction_accuracy_de20_mean", "pearson_delta_all_mean", "prediction_delta_variance_ratio"],
                    prefix="unseen_single",
                ),
                "architectural_change": "none",
                "description": f"Norman Step 0 recomputed baseline: {baseline}",
            }
        )
    return rows


def _result_rows_for_published(commit: str) -> list[dict[str, Any]]:
    rows = []
    for offset, published in enumerate(PUBLISHED_BASELINES, start=4):
        rows.append(
            {
                "commit": commit,
                "experiment_num": offset,
                "family": "Step0",
                "tier_reached": "Published",
                "status": "BASELINE_COMPLETE",
                "primary_metric": (
                    f"published_overall pearson_de={published['pearson_de']:.3f}"
                    f"+/-{published['pearson_de_std']:.3f}"
                ),
                "secondary_metric": (
                    f"published_overall mse={published['mse']:.3f}+/-{published['mse_std']:.3f}; "
                    "top20_de_overlap=not_reported"
                ),
                "protected_metric_summary": "not subset-specific in published table",
                "architectural_change": "none",
                "description": f"Immutable published comparator: {published['display_name']}",
            }
        )
    return rows


def _write_summary(output_dir: Path, metadata: dict[str, Any], split_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    lines = [
        "# Norman 2019 Step 0 Baseline Summary",
        "",
        f"- Commit: `{metadata['commit']}`",
        f"- Source h5ad: `{metadata['h5ad_path']}`",
        f"- Conditions: `{metadata['conditions']}`",
        f"- Genes: `{metadata['genes']}`",
        f"- Split seeds: `{metadata['split_seeds']}`",
        f"- Runtime seconds: `{metadata['elapsed_seconds']:.2f}`",
        "",
        "## Split Summary",
        "",
        _df_to_markdown(split_df),
        "",
        "## Recomputed Baselines",
        "",
        _aggregate_markdown(aggregate_df),
        "",
        "## Published Baselines",
        "",
        "| baseline | MSE | Pearson DE | top-20 DE overlap | caveat |",
        "|---|---:|---:|---:|---|",
    ]
    for baseline in PUBLISHED_BASELINES:
        lines.append(
            f"| {baseline['display_name']} | {baseline['mse']:.3f} +/- {baseline['mse_std']:.3f} | "
            f"{baseline['pearson_de']:.3f} +/- {baseline['pearson_de_std']:.3f} | not reported | "
            f"{baseline['top20_de_overlap_note']} |"
        )
    lines.extend(
        [
            "",
            "## Review Gate",
            "",
            "Step 0 is complete. Family A and Family B were not started; the loop is paused for review.",
        ]
    )
    (output_dir / STEP0_DIRNAME / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_registry(output_dir: Path, metadata: dict[str, Any], split_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    lines = [
        "# Baseline Registry",
        "",
        "## Model Of Record",
        "",
        "- Active published baseline: GEARS published Norman 2019 numbers.",
        "- Active carried reference: Family N train-only condition-mean table recomputed on the Norman split.",
        "- Rebasing rule: no Step 0 or Tier 1/Tier 2 result can supersede GEARS; only a Tier 3 pass can rebase.",
        "",
        "## Dataset And Split",
        "",
        f"- Dataset file: `{metadata['h5ad_path']}`",
        f"- Source URL: {metadata['source_url']}",
        f"- Conditions: `{metadata['conditions']}`",
        f"- Genes: `{metadata['genes']}`",
        "- Canonical split implementation: `perturb_jepa/data/norman2019.py::gears_simulation_split`.",
        "- Split fidelity unit test: `tests/test_norman2019_split.py`.",
        "- Known metadata caveat: the processed GEARS h5ad has `cell_type == A549`; the Norman/GEARS text describes the screen as K562. This run preserves the file metadata and records the ambiguity.",
        "",
        "## Split Counts",
        "",
        _df_to_markdown(split_df),
        "",
        "## Recomputed Baselines",
        "",
        _aggregate_markdown(aggregate_df),
        "",
        "## Published Baselines",
        "",
        "| baseline | source | MSE | Pearson DE | top-20 DE overlap | caveat |",
        "|---|---|---:|---:|---:|---|",
    ]
    for baseline in PUBLISHED_BASELINES:
        lines.append(
            f"| {baseline['display_name']} | {baseline['source']} | "
            f"{baseline['mse']:.3f} +/- {baseline['mse_std']:.3f} | "
            f"{baseline['pearson_de']:.3f} +/- {baseline['pearson_de_std']:.3f} | "
            f"not reported | {baseline['top20_de_overlap_note']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- GEARS Supplementary Table 6 reports MSE and Pearson DE, not subset-specific exact-train-combo/unseen-single metrics.",
            "- GEARS Supplementary Table 6 does not report top-20 DE overlap, so Tier 3 top-20 comparison remains unresolved until raw GEARS predictions or a paper-faithful rerun are available.",
            "- Closed-form PLS is RNA-only here: perturbation multi-hot features are mapped to pseudobulk delta by PLS regression because no image modality exists in the Norman h5ad.",
        ]
    )
    (output_dir / "BASELINE_REGISTRY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_journal(output_dir: Path, metadata: dict[str, Any], aggregate_df: pd.DataFrame) -> None:
    exact_best = aggregate_df[aggregate_df["subset"].eq("exact_train_combo")].sort_values(
        "pearson_delta_all_mean", ascending=False
    )
    best_name = str(exact_best.iloc[0]["baseline"]) if not exact_best.empty else "unavailable"
    best_value = float(exact_best.iloc[0]["pearson_delta_all_mean"]) if not exact_best.empty else math.nan
    lines = [
        "# Norman v1 Research Journal",
        "",
        "## Experiment 0: Step 0 Baselines",
        "",
        "**Hypothesis**: Before any architecture search, the Norman split must be anchored by simple recomputed baselines and immutable published comparators.",
        "",
        "**Family**: Step0 / Stage A.",
        "",
        "**Implementation**: Added `perturb_jepa/data/norman2019.py`, `scripts/run_norman_step0.py`, and `tests/test_norman2019_split.py`. No locked evaluator, gene set, DE definition, or published number was modified.",
        "",
        "**Initialization / identity preservation**: GEARS published numbers and Family N remain the active model-of-record references. The bridge architecture is unchanged.",
        "",
        f"**Tier result**: BASELINE_COMPLETE. Best recomputed exact-train-combo all-gene Pearson delta baseline: `{best_name}` = `{best_value:.4f}`.",
        "",
        "**Diagnostics**: per-condition metrics, split counts, prediction-variance ratio, direction-aware DE agreement, and published-number caveats are written under `outputs/autoresearch_norman_v1/step0_baselines/`.",
        "",
        "**Decision**: Step 0 review gate reached. Family A and Family B were not started.",
        "",
        "**Learning**: Published GEARS numbers are aggregate Pearson DE/MSE values. They are not directly subset-specific and do not include top-20 DE overlap in Supplementary Table 6.",
        "",
        f"Runtime seconds: `{metadata['elapsed_seconds']:.2f}`.",
    ]
    (output_dir / "research_journal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_architecture_log(output_dir: Path) -> None:
    lines = [
        "# Architectural Changes Log",
        "",
        "## Step 0",
        "",
        "- Architecture experiments run: `0`.",
        "- Bridge, perturbation encoder, evaluator code, Norman split definition after creation, gene set/DE definitions, and published baseline values were not modified during search.",
        "- New non-architecture infrastructure: Norman condition loader, split fidelity test, and Step 0 baseline script.",
        "- Family A and Family B are locked until Step 0 review.",
    ]
    (output_dir / "architectural_changes_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_family_allocation(output_dir: Path) -> None:
    lines = [
        "# Family Allocation",
        "",
        "| stage/family | experiments used | status | notes |",
        "|---|---:|---|---|",
        "| Step 0 / Stage A | 6 baseline registrations | complete | Four recomputed baselines plus GEARS and CPA published comparators. |",
        "| Family A | 0 | blocked by review gate | Not started. |",
        "| Family B | 0 | blocked by review gate | Not started. |",
        "",
        "Architecture experiment cap remains `20`; used `0`.",
    ]
    (output_dir / "family_allocation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_papers_consulted(output_dir: Path) -> None:
    lines = [
        "# Papers Consulted",
        "",
        "| paper/resource | use in this run | outcome |",
        "|---|---|---|",
        "| Roohani, Huang, Leskovec, `Predicting transcriptional outcomes of novel multigene perturbations with GEARS`, Nature Biotechnology 2024 | GEARS and CPA published Norman comparator values from Supplementary Table 6; GEARS-compatible split context. | Values registered as immutable published baselines. |",
        "| Norman et al., `Exploring genetic interaction manifolds constructed from rich single-cell phenotypes`, Science 2019 | Biological provenance for single/double perturbation screen. | Dataset source recorded; no ortholog mapping applied. |",
        "| Lotfollahi et al., `Predicting cellular responses to complex perturbations in high-throughput screens`, Molecular Systems Biology 2023 | CPA comparator identity. | CPA values taken from GEARS Supplementary Table 6 because the protocol requests published CPA numbers on the GEARS Norman split. |",
    ]
    (output_dir / "papers_consulted.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_external_resources(output_dir: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# External Resources",
        "",
        "## Norman / GEARS Processed Data",
        "",
        f"- Download URL: {metadata['source_url']}",
        f"- Local h5ad: `{metadata['h5ad_path']}`",
        f"- Local h5ad bytes: `{metadata['h5ad_size_bytes']}`",
        "- Version marker: Harvard Dataverse datafile ID `6154020`, GEARS processed `norman/perturb_processed.h5ad`.",
        "- License: not embedded in the h5ad; do not redistribute the raw file from this repo without checking Dataverse terms.",
        "- Gene-symbol resolution: `outputs/autoresearch_norman_v1/step0_baselines/gene_symbol_resolution.tsv` maps h5ad var index IDs to `var['gene_name']`.",
        "- Ortholog mapping: none applied.",
        "- Metadata caveat: h5ad `cell_type` is `A549`; protocol and papers refer to Norman 2019 K562. This run preserves source metadata and does not relabel cells.",
    ]
    (output_dir / "external_resources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_identity_log(output_dir: Path) -> None:
    lines = [
        "# Identity Violations Considered",
        "",
        "- Changing the Norman/GEARS split to make subset counts more convenient: rejected; canonical split is kept.",
        "- Filling missing published top-20 DE overlap by assumption: rejected; logged as not reported.",
        "- Relabeling h5ad `cell_type` from A549 to K562: rejected; source metadata is preserved and caveated.",
        "- Starting Family A/B before Step 0 review: rejected by protocol.",
    ]
    (output_dir / "identity_violations_considered.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_step0_review_packet(output_dir: Path, metadata: dict[str, Any], aggregate_df: pd.DataFrame) -> None:
    exact = aggregate_df[aggregate_df["subset"].eq("exact_train_combo")].copy()
    exact = exact.sort_values("pearson_delta_all_mean", ascending=False)
    best = exact.iloc[0].to_dict() if not exact.empty else {}
    lines = [
        "# Norman v1 Step 0 Review Packet",
        "",
        "Status: paused at the mandatory Step 0 review gate. This is not a Tier 3 promotion or architecture-search closure.",
        "",
        "## What Ran",
        "",
        "- Built Norman 2019 condition-level loader and GEARS-compatible simulation split.",
        "- Added split leakage test for perturbation-order aliases.",
        "- Ran Step 0 recomputed baselines on CPU: global train mean, single additive, Family N condition-mean table, and closed-form PLS readout.",
        "- Registered GEARS and CPA published values without modification.",
        "- Did not start Family A or Family B.",
        "",
        "## Best Recomputed Exact-Train-Combo Baseline",
        "",
        f"- Baseline: `{best.get('baseline', 'unavailable')}`",
        f"- Pearson delta all genes: `{best.get('pearson_delta_all_mean', math.nan):.4f}`",
        f"- Pearson delta DE20: `{best.get('pearson_delta_de20_mean', math.nan):.4f}`",
        f"- Top-20 DE overlap: `{best.get('top20_de_overlap_mean', math.nan):.4f}`",
        f"- MSE delta all genes: `{best.get('mse_delta_all_mean', math.nan):.4f}`",
        "",
        "## Active Model Of Record",
        "",
        "- Published GEARS remains active.",
        "- Family N recomputed condition-mean table remains the carried train-only reference.",
        "- No Tier 3 pass occurred.",
        "",
        "## Required Next Action",
        "",
        "Review Step 0 artifacts before authorizing Family A or Family B.",
        "",
        f"Runtime seconds: `{metadata['elapsed_seconds']:.2f}`.",
    ]
    (output_dir / "STEP0_REVIEW_PACKET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gene_symbol_map(dataset: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene_id": dataset.gene_ids, "gene_name": dataset.gene_names}).to_csv(path, sep="\t", index=False)


def _aggregate_markdown(aggregate_df: pd.DataFrame) -> str:
    columns = [
        "baseline",
        "subset",
        "condition_count",
        "pearson_delta_all_mean",
        "pearson_delta_de20_mean",
        "top20_de_overlap_mean",
        "mse_delta_all_mean",
        "mse_expression_de20_mean",
        "direction_accuracy_de20_mean",
        "prediction_delta_variance_ratio",
        "mean_collapse_flag",
    ]
    existing = [column for column in columns if column in aggregate_df.columns]
    view = aggregate_df[existing].copy()
    for column in view.columns:
        if column.endswith("_mean") or column.endswith("_ratio"):
            view[column] = view[column].map(lambda value: f"{float(value):.4f}" if not _is_nan(value) else "nan")
    return _df_to_markdown(view)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rows = [[str(column) for column in df.columns]]
    for _, row in df.iterrows():
        rows.append([_markdown_cell(row[column]) for column in df.columns])
    widths = [max(len(values[index]) for values in rows) for index in range(len(rows[0]))]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, separator, *body])


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if _is_nan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _aggregate_lookup(aggregate_df: pd.DataFrame, baseline: str, subset: str) -> dict[str, Any]:
    rows = aggregate_df[aggregate_df["baseline"].eq(baseline) & aggregate_df["subset"].eq(subset)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _format_metrics(row: dict[str, Any], columns: list[str], *, prefix: str) -> str:
    if not row:
        return f"{prefix}=unavailable"
    parts = []
    for column in columns:
        value = row.get(column, math.nan)
        formatted = "nan" if _is_nan(value) else f"{float(value):.4f}"
        parts.append(f"{column}={formatted}")
    return f"{prefix} " + "; ".join(parts)


def _safe_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    true = np.asarray(true, dtype=float).ravel()
    if pred.size < 2 or true.size < 2:
        return math.nan
    if float(np.std(pred)) < 1e-12 or float(np.std(true)) < 1e-12:
        return math.nan
    return float(np.corrcoef(pred, true)[0, 1])


def _topk_overlap(pred_delta: np.ndarray, true_de_indices: np.ndarray, *, k: int) -> float:
    pred_top = set(np.argsort(np.abs(pred_delta))[::-1][:k].tolist())
    true_top = set(np.asarray(true_de_indices, dtype=int)[:k].tolist())
    return float(len(pred_top & true_top) / max(1, min(k, len(true_top))))


def _direction_accuracy(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    true_sign = np.sign(np.asarray(true_delta, dtype=float))
    pred_sign = np.sign(np.asarray(pred_delta, dtype=float))
    mask = true_sign != 0
    if not np.any(mask):
        return math.nan
    return float(np.mean(pred_sign[mask] == true_sign[mask]))


def _require_split(dataset: Any) -> Any:
    if dataset.split is None:
        raise RuntimeError("NormanDataset requires an attached split")
    return dataset.split


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
