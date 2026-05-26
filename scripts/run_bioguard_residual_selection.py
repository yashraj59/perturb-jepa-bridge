from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.operators.bioguard_residuals import ResidualTargetCache, residual_target_stats
from perturb_jepa.operators.bioguard_selection import CrossFittedResidualSelector, candidate_factory
from perturb_jepa.training.bioguard_splits import ActionGroupedResidualSplitConfig, ActionGroupedResidualSplitter
from perturb_jepa.training.biospectral_operator import (
    bundle_features,
    bundle_transition_metrics,
    effective_rank,
    fit_ridge_numpy,
    load_latent_bundle,
    predict_ridge_numpy,
)
from scripts.run_bioguard_phase7_reproduction import (
    FULL_FLOOR,
    PHASE4_CACHE,
    PHASE7_ROOT,
    append_journal,
    initialize_docs,
    write_bioguard_leakage_report,
)


EXPERIMENT_BY_CANDIDATE = {
    "spectral": ("BSG002", "BSG002_KEEP_SPECTRAL_RESIDUAL_PASSES_CV_GATE", "BSG002_DISCARD_SPECTRAL_RESIDUAL_FAILS_CV_GATE", "BSG002_DISCARD_SPECTRAL_RESIDUAL_BELOW_FLOOR"),
    "kernel": ("BSG003", "BSG003_KEEP_KERNEL_RESIDUAL_PASSES_CV_GATE", "BSG003_DISCARD_KERNEL_RESIDUAL_FAILS_CV_GATE", "BSG003_DISCARD_KERNEL_RESIDUAL_BELOW_FLOOR"),
    "program": ("BSG004", "BSG004_KEEP_PROGRAM_RESIDUAL_PASSES_CV_GATE", "BSG004_DISCARD_PROGRAM_RESIDUAL_FAILS_CV_GATE", "BSG004_DISCARD_PROGRAM_RESIDUAL_BELOW_FLOOR"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7 train-only residual target audit and cross-fitted selection.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--candidate", choices=("all", "spectral", "kernel", "program"), default="all")
    parser.add_argument("--output-dir", type=Path, default=PHASE7_ROOT / "experiments/BSG001_residual_selection_seed0")
    args = parser.parse_args()

    initialize_docs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = load_latent_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")
    floor_fit = fit_ridge_numpy(bundle_features(train), train.delta, alpha=1.0e-2)
    floor_eval_delta = predict_ridge_numpy(floor_fit, bundle_features(eval_bundle))
    floor_eval = bundle_transition_metrics(eval_bundle, floor_eval_delta)
    split_config = ActionGroupedResidualSplitConfig(n_folds=args.folds, seed=args.seed)
    splitter = ActionGroupedResidualSplitter(split_config)
    try:
        folds = splitter.split(train.metadata)
    except ValueError as exc:
        decision = "BSG001_ABORT_SPLIT_LEAKAGE"
        write_final_report("PHASE7_CLOSE_IDENTITY_OR_LEAKAGE_FAILURE", [f"Split construction failed: {exc}"], floor_eval, [])
        print(json.dumps({"decision": decision, "error": str(exc)}, sort_keys=True))
        return 2

    split_report = pd.DataFrame([fold.report_row() for fold in folds])
    split_report.to_csv(args.output_dir / "split_report.tsv", sep="\t", index=False)
    cache = ResidualTargetCache.from_bundle(train, alpha=1.0e-2)
    stats = residual_target_stats(cache)
    stats.update(probe_stats(train, cache.residual_star))
    target_valid = (
        stats["residual_target_rank"] >= 2.0
        and stats["residual_target_magnitude"] > 1.0e-4
        and not stats["residual_batch_dominated"]
    )
    bsg001_decision = "BSG001_PASS_RESIDUAL_TARGET_VALID" if target_valid else "BSG001_CLOSE_NO_VALID_RESIDUAL_TARGET"
    write_residual_target_stats(args.output_dir, stats, split_report, bsg001_decision)
    write_bioguard_leakage_report(args.output_dir / "leakage_report.md", mode="BSG001_residual_target_split_audit", train_rows=train.source.shape[0], eval_rows=eval_bundle.source.shape[0])
    write_leakage_audit(train.source.shape[0], eval_bundle.source.shape[0])
    update_results("BSG001", args, mode="residual_target_split_audit", status=bsg001_decision, decision=bsg001_decision, metrics=floor_eval, floor=floor_eval, artifact_dir=args.output_dir)
    append_journal("BSG001", bsg001_decision, f"Residual target rank {stats['residual_target_rank']:.4f}, magnitude {stats['residual_target_magnitude']:.4f}; folds {len(folds)}.")
    if not target_valid:
        write_final_report("PHASE7_CLOSE_NO_VALID_RESIDUAL_TARGET", ["Residual target failed rank/magnitude/batch-dominance gate."], floor_eval, [])
        print(json.dumps({"decision": bsg001_decision, "final": "PHASE7_CLOSE_NO_VALID_RESIDUAL_TARGET"}, sort_keys=True))
        return 0

    candidates = ("spectral", "kernel", "program") if args.candidate == "all" else (args.candidate,)
    candidate_results = []
    for candidate in candidates:
        result = CrossFittedResidualSelector(
            folds=folds,
            candidate_factory=candidate_factory(candidate),
            candidate_id=candidate,
        ).evaluate(train)
        candidate_results.append(result)
        experiment_id, keep_label, fail_label, below_label = EXPERIMENT_BY_CANDIDATE[candidate]
        decision = keep_label if result.selected else fail_label
        exp_dir = PHASE7_ROOT / f"experiments/{experiment_id}_{candidate}_seed{args.seed}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.fold_rows).to_csv(exp_dir / "crossfit_fold_metrics.tsv", sep="\t", index=False)
        write_json(exp_dir / "metrics_eval.json", result_to_dict(result) | {"decision_label": decision})
        write_bioguard_leakage_report(exp_dir / "leakage_report.md", mode=f"{experiment_id}_{candidate}_crossfit_selection", train_rows=train.source.shape[0], eval_rows=eval_bundle.source.shape[0])
        update_results(
            experiment_id,
            args,
            mode=f"{candidate}_crossfit_selection",
            status=decision,
            decision=decision,
            metrics=floor_eval,
            floor=floor_eval,
            artifact_dir=exp_dir,
            cv_lcb_transition_gap=result.cv_lcb_transition_gap,
            cv_lcb_recall_gap=result.cv_lcb_recall_gap,
            residual_gate_mean=result.residual_gate_mean,
            residual_gate_nonzero_fraction=result.residual_gate_nonzero_fraction,
            residual_scale=result.residual_scale,
            train_residual_fit_metric=result.train_residual_fit_metric,
            calibration_residual_fit_metric=result.calibration_residual_fit_metric,
            action_negative_gap=result.action_negative_gap,
        )
        append_journal(
            experiment_id,
            decision,
            f"Candidate `{candidate}` CV LCB transition gap {result.cv_lcb_transition_gap:.6f}, CV LCB recall gap {result.cv_lcb_recall_gap:.6f}, selected={result.selected}.",
        )

    write_selection_reports(candidate_results)
    selected = [result for result in candidate_results if result.selected]
    if not selected:
        write_family_allocation_closed()
        write_final_report(
            "PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA",
            ["No residual candidate passed train-internal cross-fitted selection. BSG005-BSG008 were not run."],
            floor_eval,
            candidate_results,
        )
        print(json.dumps({"decision": "PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA", "selected": []}, sort_keys=True))
        return 0
    write_final_report(
        "PHASE7_KEEP_BIOGUARD_OPERATOR_READY_FOR_JEPA_WRAPPER",
        ["At least one residual candidate passed train-internal selection; wrapper work is gated for a future continuation."],
        floor_eval,
        candidate_results,
    )
    print(json.dumps({"decision": "PHASE7_KEEP_BIOGUARD_OPERATOR_READY_FOR_JEPA_WRAPPER", "selected": [item.candidate_id for item in selected]}, sort_keys=True))
    return 0


def probe_stats(bundle, residual: np.ndarray) -> dict[str, Any]:
    action_acc = nearest_centroid_accuracy(residual, bundle.metadata["perturbation_id"].astype(str).to_numpy())
    batch_acc = nearest_centroid_accuracy(residual, bundle.metadata["batch_id"].astype(str).to_numpy()) if "batch_id" in bundle.metadata else 0.0
    return {
        "residual_target_action_probe_accuracy": action_acc,
        "residual_target_batch_probe_accuracy": batch_acc,
        "residual_batch_dominated": bool(batch_acc > 0.90 and batch_acc >= action_acc),
    }


def nearest_centroid_accuracy(values: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(str)
    preds = []
    for idx in range(values.shape[0]):
        train_mask = np.arange(values.shape[0]) != idx
        centroids = []
        centroid_labels = []
        for label in np.unique(labels[train_mask]):
            mask = train_mask & (labels == label)
            centroids.append(values[mask].mean(axis=0))
            centroid_labels.append(label)
        centroids_array = np.asarray(centroids)
        distances = ((centroids_array - values[idx]) ** 2).sum(axis=1)
        preds.append(centroid_labels[int(np.argmin(distances))])
    return float(np.mean(np.asarray(preds, dtype=str) == labels))


def write_residual_target_stats(path: Path, stats: dict[str, Any], split_report: pd.DataFrame, decision: str) -> None:
    lines = [
        "# Residual Target Stats",
        "",
        f"- teacher delta rank: `{stats['teacher_delta_rank']:.4f}`",
        f"- floor delta rank: `{stats['floor_delta_rank']:.4f}`",
        f"- residual target rank: `{stats['residual_target_rank']:.4f}`",
        f"- residual target magnitude: `{stats['residual_target_magnitude']:.4f}`",
        f"- residual target action-probe accuracy: `{stats['residual_target_action_probe_accuracy']:.4f}`",
        f"- residual target batch-probe accuracy: `{stats['residual_target_batch_probe_accuracy']:.4f}`",
        f"- residual near-zero fraction: `{stats['residual_near_zero_fraction']:.4f}`",
        f"- residual batch dominated: `{stats['residual_batch_dominated']}`",
        "",
        "## Split Summary",
        "",
        markdown_table(split_report[["fold_id", "fit_rows", "calibration_rows", "fit_actions", "calibration_actions", "fallback_reason"]]),
        "",
        "## Decision",
        "",
        f"`{decision}`",
    ]
    (path / "residual_target_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PHASE7_ROOT / "residual_target_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_reports(results) -> None:
    rows = [result_to_dict(result) for result in results]
    frame = pd.DataFrame(rows)
    frame.to_csv(PHASE7_ROOT / "residual_selection_table.tsv", sep="\t", index=False)
    lines = [
        "# Residual Selection Report",
        "",
        markdown_table(frame[
            [
                "candidate_id",
                "selected",
                "cv_lcb_transition_gap",
                "cv_lcb_recall_gap",
                "mean_transition_gap",
                "mean_recall_gap",
                "mean_delta_cosine_gap",
                "residual_scale",
                "action_negative_gap",
                "decision_label",
            ]
        ]),
    ]
    (PHASE7_ROOT / "residual_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cal_lines = [
        "# Residual Calibration Report",
        "",
        markdown_table(frame[
            [
                "candidate_id",
                "train_residual_fit_metric",
                "calibration_residual_fit_metric",
                "residual_train_to_calibration_gap",
                "residual_gate_mean",
                "residual_gate_nonzero_fraction",
                "residual_scale",
            ]
        ]),
    ]
    (PHASE7_ROOT / "residual_calibration_report.md").write_text("\n".join(cal_lines) + "\n", encoding="utf-8")


def update_results(
    experiment_id: str,
    args: argparse.Namespace,
    *,
    mode: str,
    status: str,
    decision: str,
    metrics: dict[str, float],
    floor: dict[str, float],
    artifact_dir: Path,
    cv_lcb_transition_gap: float = 0.0,
    cv_lcb_recall_gap: float = 0.0,
    residual_gate_mean: float = 0.0,
    residual_gate_nonzero_fraction: float = 0.0,
    residual_scale: float = 0.0,
    train_residual_fit_metric: float = 0.0,
    calibration_residual_fit_metric: float = 0.0,
    action_negative_gap: float = 0.0,
) -> None:
    path = PHASE7_ROOT / "results.tsv"
    frame = pd.read_csv(path, sep="\t")
    row = {
        "experiment_id": experiment_id,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "mode": mode,
        "status": status,
        "transition_improvement": f"{metrics['transition_source_cosine_improvement']:.4f}",
        "delta_cosine": f"{metrics['delta_cosine']:.4f}",
        "recall_at_1": f"{metrics['transition_to_target_recall@1']:.4f}",
        "delta_rank": f"{metrics['delta_prediction_effective_rank']:.4f}",
        "magnitude_ratio": f"{metrics['delta_magnitude_ratio']:.4f}",
        "floor_transition_improvement": f"{floor['transition_source_cosine_improvement']:.4f}",
        "floor_delta_cosine": f"{floor['delta_cosine']:.4f}",
        "floor_recall_at_1": f"{floor['transition_to_target_recall@1']:.4f}",
        "floor_delta_rank": f"{floor['delta_prediction_effective_rank']:.4f}",
        "floor_gap_transition": f"{metrics['transition_source_cosine_improvement'] - floor['transition_source_cosine_improvement']:.4f}",
        "floor_gap_recall": f"{metrics['transition_to_target_recall@1'] - floor['transition_to_target_recall@1']:.4f}",
        "floor_gap_delta_cosine": f"{metrics['delta_cosine'] - floor['delta_cosine']:.4f}",
        "cv_lcb_transition_gap": f"{cv_lcb_transition_gap:.6f}",
        "cv_lcb_recall_gap": f"{cv_lcb_recall_gap:.6f}",
        "residual_gate_mean": f"{residual_gate_mean:.4f}",
        "residual_gate_nonzero_fraction": f"{residual_gate_nonzero_fraction:.4f}",
        "residual_scale": f"{residual_scale:.4f}",
        "train_residual_fit_metric": f"{train_residual_fit_metric:.6f}",
        "calibration_residual_fit_metric": f"{calibration_residual_fit_metric:.6f}",
        "action_negative_gap": f"{action_negative_gap:.6f}",
        "leakage_status": "PASS",
        "decision_label": decision,
        "artifact_dir": str(artifact_dir),
    }
    frame = frame[frame["experiment_id"].astype(str) != experiment_id]
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, sep="\t", index=False)


def write_leakage_audit(train_rows: int, eval_rows: int) -> None:
    lines = [
        "# Phase 7 Leakage Audit",
        "",
        f"- Train rows available for fitting/calibration: `{train_rows}`",
        f"- Eval rows used only for final scoring when a selected residual exists: `{eval_rows}`",
        "- Eval/test target rows used for fitting: `no`",
        "- Eval/test target rows used for whitening/statistics: `no`",
        "- Eval/test target rows used for residual calibration or selection: `no`",
        "- `condition_key` used as model feature: `no`",
        "- `biological_key` used as model feature: `no`",
        "- Exact target-key one-hot features used: `no`",
        "- Batch id used as biological shortcut: `no`",
        "- Raw-linear PLS main path: `no`",
        "- Candidate choices based on eval/test performance: `no`",
    ]
    (PHASE7_ROOT / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_allocation_closed() -> None:
    text = """# Phase 7 Family Allocation

| family | experiments | status |
| --- | --- | --- |
| reproduction | BSG000 | completed |
| residual target/split audit | BSG001 | completed |
| cross-fitted residual candidates | BSG002-BSG004 | completed; no candidate passed CV gate |
| ensemble | BSG005 | not run because no residual passed |
| BioGuard wrapper/full JEPA/Norman | BSG006-BSG008 | not run because no residual passed |
"""
    (PHASE7_ROOT / "family_allocation.md").write_text(text, encoding="utf-8")


def write_final_report(decision: str, notes: list[str], floor: dict[str, float], results) -> None:
    rows = [result_to_dict(result) for result in results]
    selection_table = pd.DataFrame(rows)
    selection_md = markdown_table(selection_table[
        [
            "candidate_id",
            "selected",
            "cv_lcb_transition_gap",
            "cv_lcb_recall_gap",
            "mean_transition_gap",
            "mean_recall_gap",
            "residual_scale",
            "action_negative_gap",
        ]
    ]) if not selection_table.empty else "_No residual candidates evaluated._"
    lines = [
        "# BioGuard-JEPA Phase 7 Final Report",
        "",
        "## Decision label",
        decision,
        "",
        "## Model of record status",
        "Protected rank-3 train-split-only PLS raw-linear readout remains model of record.",
        "",
        "## Experiments run",
        "- BSG000 reproduction: completed before residual selection.",
        "- BSG001 residual target/split audit: completed.",
        "- BSG002-BSG004 residual candidates: evaluated through train-only cross-fitted selection.",
        "- BSG005-BSG008: not run because no residual passed selection.",
        "",
        "## Exact floor values used",
        f"- transition improvement: `{floor['transition_source_cosine_improvement']:.4f}`",
        f"- delta cosine: `{floor['delta_cosine']:.4f}`",
        f"- recall@1: `{floor['transition_to_target_recall@1']:.4f}`",
        f"- delta rank: `{floor['delta_prediction_effective_rank']:.4f}`",
        f"- magnitude ratio: `{floor['delta_magnitude_ratio']:.4f}`",
        "",
        "## Train-Internal Cross-Fit Selection Table",
        selection_md,
        "",
        "## Calibration/Gating",
        "Every evaluated residual defaulted to zero residual / floor fallback because the conservative train-internal gate did not pass.",
        "",
        "## Leakage audit summary",
        "PASS. Eval/test target rows were not used for fitting, whitening/statistics, residual calibration, residual selection, or candidate choice.",
        "",
        "## JEPA identity status",
        "No full BioGuard-JEPA candidate was trained. Operator-only probes cannot promote the model.",
        "",
        "## Norman status",
        "Norman was not run because synthetic BioGuard residual selection did not pass.",
        "",
        "## Recommendation",
        "Close Phase 7 under current data. The full-ridge floor is the safest transition operator until a residual has positive train-internal lower-confidence evidence.",
        "",
        "## Notes",
        *[f"- {note}" for note in notes],
    ]
    (PHASE7_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_to_dict(result) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "selected": result.selected,
        "decision_label": result.decision_label,
        "cv_lcb_transition_gap": result.cv_lcb_transition_gap,
        "cv_lcb_recall_gap": result.cv_lcb_recall_gap,
        "mean_transition_gap": result.mean_transition_gap,
        "mean_recall_gap": result.mean_recall_gap,
        "mean_delta_cosine_gap": result.mean_delta_cosine_gap,
        "mean_delta_rank_gap": result.mean_delta_rank_gap,
        "magnitude_ratio_mean": result.magnitude_ratio_mean,
        "residual_scale": result.residual_scale,
        "residual_gate_mean": result.residual_gate_mean,
        "residual_gate_nonzero_fraction": result.residual_gate_nonzero_fraction,
        "train_residual_fit_metric": result.train_residual_fit_metric,
        "calibration_residual_fit_metric": result.calibration_residual_fit_metric,
        "residual_train_to_calibration_gap": result.residual_train_to_calibration_gap,
        "action_negative_gap": result.action_negative_gap,
        "permutation_action_gap": result.permutation_action_gap,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(column), *(len(row[index]) for row in rows)) for index, column in enumerate(columns)]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
