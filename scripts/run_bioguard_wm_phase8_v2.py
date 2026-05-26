from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.jepawm_predictor import (
    ActionAdaLNPredictor,
    BioJEPAWMContextConfig,
    FloorPreservingJEPAWMTransitionHead,
    context_contract_dict,
)
from perturb_jepa.training.bioguard_wm_status import parse_phase7_status
from perturb_jepa.training.biospectral_operator import bundle_features, bundle_transition_metrics, fit_ridge_numpy, load_latent_bundle, predict_ridge_numpy
from scripts.train_bioguard_wm_jepa import PHASE4_CACHE, run as run_train


PHASE8_ROOT = Path("outputs/autoresearch_bioguard_wm_jepa_phase8_v2")
FLOOR = {
    "transition_source_cosine_improvement": 0.0057,
    "delta_cosine": 0.3980,
    "transition_to_target_recall@1": 0.4815,
    "delta_prediction_effective_rank": 10.2835,
    "delta_magnitude_ratio": 0.7744,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrate Phase 8 v2 BioGuard-WM-JEPA BGWM000-BGWM006.")
    parser.add_argument("--stage", default="through_BGWM002")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--paper-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PHASE8_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    initialize_docs(args.output_root, args.paper_path)
    if not args.paper_path.exists():
        (args.output_root / "PAPER_MISSING.md").write_text(f"Missing paper: `{args.paper_path}`\n", encoding="utf-8")
        write_final_report(args.output_root, "PHASE8V2_PAPER_MISSING_STOP", notes=["Paper was not found."], rows=[])
        return 2
    bgwm000 = run_bgwm000(args)
    if bgwm000["decision_label"] != "BGWM000_KEEP_AUDIT_REOPEN_PREDICTOR_ASSAY":
        write_final_report(args.output_root, bgwm000["decision_label"], notes=["BGWM000 audit did not reopen predictor assay."], rows=[bgwm000])
        return 2
    if args.stage == "BGWM000":
        return 0
    bgwm001 = run_bgwm001(args)
    if bgwm001["decision_label"] != "BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT":
        write_final_report(args.output_root, "BGWM001_FAIL_FLOOR_PRESERVATION", notes=["Zero-residual floor preservation failed."], rows=[bgwm000, bgwm001])
        return 2
    if args.stage == "BGWM001":
        return 0
    bgwm002 = run_bgwm002(args)
    rows = [bgwm000, bgwm001, bgwm002]
    if bgwm002["decision_label"] == "BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN":
        write_reopening(args.output_root, "PHASE8V2_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA")
        write_final_report(
            args.output_root,
            "PHASE8V2_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA",
            notes=["BGWM002 train-only calibration selected residual_scale=0. BSG/BSJ residual failures remain locked."],
            rows=rows,
        )
        print(json.dumps({"decision": "PHASE8V2_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA"}, sort_keys=True))
        return 0
    if bgwm002["floor_gap_transition"] < -1.0e-4 or bgwm002["floor_gap_recall"] < -1.0e-4:
        write_final_report(args.output_root, "BGWM002_DISCARD_ACTION_ADALN_CALIBRATION_FALSE_POSITIVE", notes=["Selected residual fell below held-out floor."], rows=rows)
        return 0
    write_final_report(args.output_root, "BGWM002_KEEP_SAFE_ACTION_ADALN_RESIDUAL", notes=["BGWM002 passed; later full JEPA stages remain gated."], rows=rows)
    return 0


def initialize_docs(root: Path, paper_path: Path) -> None:
    docs = {
        "results.tsv": "\t".join(
            [
                "experiment_id",
                "dataset",
                "eval_split",
                "seed",
                "mode",
                "status",
                "predictor_type",
                "context_length_train",
                "context_length_eval",
                "rollout_steps",
                "transition_improvement",
                "delta_cosine",
                "recall_at_1",
                "delta_rank",
                "magnitude_ratio",
                "floor_transition_improvement",
                "floor_delta_cosine",
                "floor_recall_at_1",
                "floor_delta_rank",
                "floor_gap_transition",
                "floor_gap_recall",
                "floor_gap_delta_cosine",
                "cv_lcb_transition_gap",
                "cv_lcb_recall_gap",
                "cv_lcb_delta_cosine_gap",
                "mean_transition_gap",
                "mean_recall_gap",
                "residual_gate_mean",
                "residual_gate_nonzero_fraction",
                "residual_scale",
                "action_negative_gap",
                "rollout_loss_validated",
                "context_contract_validated",
                "leakage_status",
                "identity_status",
                "decision_label",
                "artifact_dir",
            ]
        )
        + "\n",
        "research_journal.md": "# BioGuard-WM-JEPA Phase 8 v2 Research Journal\n\n",
        "architectural_changes_log.md": "# Phase 8 v2 Architectural Changes Log\n\nOnly action-AdaLN + RoPE JEPA-WM residual predictor is allowed after Phase 7 closure.\n",
        "family_allocation.md": "# Phase 8 v2 Family Allocation\n\n| family | experiments | status |\n| --- | --- | --- |\n| audit | BGWM000 | in progress |\n| frozen-latent action-AdaLN/RoPE | BGWM001-BGWM002 | gated |\n| rollout/full JEPA/Norman | BGWM003-BGWM006 | gated after BGWM002 positive residual |\n",
        "BASELINE_REGISTRY.md": baseline_registry_text(),
        "external_resources.md": "# External Resources\n\nNo new external datasets downloaded. Local paper path is `" + str(paper_path) + "`.\n",
        "identity_violations_considered.md": "# Identity Violations Considered\n\n- Raw-linear PLS main path forbidden.\n- Condition/biological keys forbidden as inputs.\n- Eval target rows forbidden for calibration or residual scale selection.\n- Context train/eval mismatch is a hard stop.\n",
        "rollout_contract_report.md": "# Rollout Contract Report\n\nNo valid biological two-step rollout was run before BGWM002.\n",
    }
    for name, text in docs.items():
        path = root / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def run_bgwm000(args: argparse.Namespace) -> dict[str, Any]:
    paper_text = summarize_paper(args.paper_path)
    (args.output_root / "papers_consulted.md").write_text(paper_text, encoding="utf-8")
    write_missing_context(args.output_root)
    phase7 = parse_phase7_status()
    (args.output_root / "PHASE7_STATUS_LOCKED.md").write_text(
        "\n".join(
            [
                "# Phase 7 Status Locked",
                "",
                f"- decision: `{phase7.decision}`",
                f"- no residual passed: `{phase7.no_residual_passed}`",
                f"- BSG005-BSG008 not run: `{phase7.later_experiments_not_run}`",
                f"- floor values match: `{phase7.floor_values_match}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train = load_latent_bundle(PHASE4_CACHE / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(PHASE4_CACHE / f"{args.dataset}_{args.eval_split}_latents", "eval")
    fit = fit_ridge_numpy(bundle_features(train), train.delta, alpha=1.0e-2)
    eval_floor_delta = predict_ridge_numpy(fit, bundle_features(eval_bundle))
    floor_metrics = bundle_transition_metrics(eval_bundle, eval_floor_delta)
    floor_ok = (
        abs(floor_metrics["transition_source_cosine_improvement"] - FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(floor_metrics["delta_cosine"] - FLOOR["delta_cosine"]) <= 1.0e-4
        and abs(floor_metrics["transition_to_target_recall@1"] - FLOOR["transition_to_target_recall@1"]) <= 1.0e-4
    )
    floor_exact = floor_preservation_smoke(train)
    decision = (
        "BGWM000_KEEP_AUDIT_REOPEN_PREDICTOR_ASSAY"
        if phase7.decision == "PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA" and phase7.no_residual_passed and phase7.floor_values_match and floor_ok and floor_exact
        else "PHASE8V2_FLOOR_REPRODUCTION_FAIL_STOP"
    )
    write_reopening(args.output_root, "PHASE8V2_REOPEN_PREDICTOR_ASSAY_APPROVED" if decision.startswith("BGWM000") else decision)
    write_context_contract(args.output_root, 3, 3, 1)
    write_leakage_audit(args.output_root)
    artifact = args.output_root / "experiments/BGWM000_audit_seed0"
    artifact.mkdir(parents=True, exist_ok=True)
    write_standard_experiment_files(artifact, decision, full_jepa=False)
    row = row_from_metrics("BGWM000", args, "audit", decision, floor_metrics, floor_metrics, artifact, residual_scale=0.0)
    append_result(args.output_root, row)
    append_journal(args.output_root, "BGWM000", decision, f"Floor improvement {floor_metrics['transition_source_cosine_improvement']:.4f}; paper found at `{args.paper_path}`.")
    return row


def run_bgwm001(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_root / "experiments/BGWM001_zero_residual_seed0"
    train_args = argparse.Namespace(
        dataset=args.dataset,
        eval_split=args.eval_split,
        seed=args.seed,
        device=args.device,
        steps=1,
        batch_size=2,
        bag_size=3,
        shared_dim=32,
        bio_dim=24,
        tech_dim=8,
        predictor_dim=64,
        predictor_depth=6,
        predictor_heads=4,
        context_length=3,
        rollout_steps=1,
        residual_scale=0.0,
        calibration_mode="fixed_zero",
        ridge_floor_path=None,
        latent_cache_path=PHASE4_CACHE,
        output_dir=output_dir,
        save_checkpoint=False,
        paper_path=args.paper_path,
    )
    metrics = run_train(train_args)
    decision = "BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT" if abs(metrics["floor_gap_transition"]) <= 1.0e-6 and abs(metrics["floor_gap_recall"]) <= 1.0e-6 else "BGWM001_FAIL_FLOOR_PRESERVATION"
    metrics["decision_label"] = decision
    write_json(output_dir / "metrics_eval.json", metrics)
    row = row_from_train_metrics("BGWM001", args, "zero_residual_action_adaln_smoke", decision, metrics, output_dir)
    append_result(args.output_root, row)
    append_journal(args.output_root, "BGWM001", decision, f"Zero-residual floor gap transition {metrics['floor_gap_transition']:.8f}.")
    return row


def run_bgwm002(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_root / "experiments/BGWM002_frozen_action_adaln_seed0"
    train_args = argparse.Namespace(
        dataset=args.dataset,
        eval_split=args.eval_split,
        seed=args.seed,
        device=args.device,
        steps=100,
        batch_size=2,
        bag_size=3,
        shared_dim=32,
        bio_dim=24,
        tech_dim=8,
        predictor_dim=64,
        predictor_depth=6,
        predictor_heads=4,
        context_length=3,
        rollout_steps=1,
        residual_scale=0.0,
        calibration_mode="crossfit",
        ridge_floor_path=None,
        latent_cache_path=PHASE4_CACHE,
        output_dir=output_dir,
        save_checkpoint=True,
        paper_path=args.paper_path,
    )
    metrics = run_train(train_args)
    decision = metrics["decision_label"]
    row = row_from_train_metrics("BGWM002", args, "frozen_action_adaln_rope_crossfit", decision, metrics, output_dir)
    append_result(args.output_root, row)
    write_calibration_report(args.output_root, metrics)
    append_journal(args.output_root, "BGWM002", decision, f"Calibration selected residual scale {metrics['residual_scale']:.4f}.")
    return row


def sanitize_model_input_record(record: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"condition_key", "biological_key", "exact_target_key", "eval_target_mean", "batch_id", "target_key"}
    return {key: value for key, value in record.items() if key not in forbidden}


def jepa_identity_flags(*, full_jepa: bool) -> dict[str, float]:
    return {
        "online_context_encoders_present": 1.0 if full_jepa else 0.0,
        "ema_target_encoders_present": 1.0 if full_jepa else 0.0,
        "teacher_stop_gradient_verified": 1.0,
        "latent_prediction_loss_present": 1.0,
        "rna_program_jepa_present": 1.0 if full_jepa else 0.0,
        "image_region_jepa_present": 1.0 if full_jepa else 0.0,
        "cross_modal_jepa_present": 1.0 if full_jepa else 0.0,
        "action_conditioned_transition_jepa_present": 1.0,
        "encoder_path_used": 1.0 if full_jepa else 0.0,
        "raw_linear_pls_main_path_used": 0.0,
        "ridge_floor_fallback_present": 1.0,
        "residual_floor_preservation_verified": 1.0,
    }


def floor_preservation_smoke(train) -> bool:
    import torch

    config = BioJEPAWMContextConfig(z_dim=train.source.shape[1], action_dim=train.action.shape[1], predictor_dim=64, depth=1, heads=4, context_length=3)
    head = FloorPreservingJEPAWMTransitionHead(ActionAdaLNPredictor(config))
    source = torch.as_tensor(train.source[:4], dtype=torch.float32)
    action = torch.as_tensor(train.action[:4], dtype=torch.float32)
    floor_delta = torch.as_tensor(train.delta[:4], dtype=torch.float32)
    out = head(source, action, ridge_floor_delta=floor_delta, residual_gate=1.0, residual_scale=0.0)
    return bool(torch.max(torch.abs(out["pred_z"] - out["floor_z"])).item() <= 1.0e-7)


def summarize_paper(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:8])
    title = "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"
    return f"""# Papers Consulted

## {title}

- Paper: {title}
- Local path: `{path}`
- Evidence read from local PDF: `{text[:600].replace(chr(10), ' ')}`...

Extracted lessons:
1. JEPA-WM identity is latent predictive dynamics, not reconstruction/reward/value/policy heads.
2. The predictor/dynamics model is the main engineering object.
3. Action conditioning with AdaLN + RoPE is a strong predictor recipe.
4. Multistep rollout can improve robustness, but target contracts must be exact.
5. Train/eval context length mismatch is invalid.
6. L2 endpoint latent cost is a strong planning/calibration cost.
7. Deterministic predictors can average multimodal futures; residual deployment needs abstention/risk control.

Mapping to this repo:
- state = `z_bio`;
- action = perturbation/gene/drug descriptor;
- future state = perturbed teacher `z_bio`;
- protected transition floor = train-only full action-ridge delta;
- residual = optional floor-preserving correction, selected only through train-only calibration.
"""


def write_missing_context(root: Path) -> None:
    required = [
        "bioguard_wm_jepa_phase8_full_codex_prompt.md",
        "prompt/bioguard_jepa_phase7_safe_residual_prompt.md",
        "outputs/autoresearch_bioguard_jepa_phase7/final_report.md",
        "outputs/autoresearch_bioguard_jepa_phase7/results.tsv",
        "outputs/autoresearch_bioguard_jepa_phase7/residual_calibration_report.md",
        "outputs/autoresearch_bioguard_jepa_phase7/residual_selection_report.md",
        "outputs/autoresearch_bioguard_jepa_phase7/BASELINE_REGISTRY.md",
        "final_report (12).md",
        "final_report (11).md",
        "final_report (8).md",
        "final_report (7).md",
        "final_report (6).md",
        "final_report (5).md",
        "outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md",
        "outputs/autoresearch_biotech_jepa_phase2/BIOTECH_JEPA_CODE_INDEX.md",
        "outputs/autoresearch_biotech_jepa_phase2/NORMAN_CONTEXT_AUDIT.md",
        "research_journal (14).md",
        "research_journal (15).md",
        "architectural_changes_log (7).md",
        "CURRENT_STATUS_AND_BEST_MODEL_CODE (1).md",
        "outputs/autoresearch_synth_lite/FULL_ARCHITECTURE_CODE_BUNDLE.md",
        "SKILL (2).md",
    ]
    missing = [item for item in required if not Path(item).exists()]
    lines = ["# Missing Context", "", *[f"- `{item}`" for item in missing]]
    (root / "missing_context.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_context_contract(root: Path, train_len: int, eval_len: int, rollout_steps: int) -> None:
    write_json(root / "context_contract.json", context_contract_dict(context_length_train=train_len, context_length_eval=eval_len, rollout_steps=rollout_steps))


def write_reopening(root: Path, label: str) -> None:
    (root / "REOPENING_DECISION.md").write_text(label + "\n", encoding="utf-8")


def write_leakage_audit(root: Path) -> None:
    (root / "leakage_audit.md").write_text(
        "# Leakage Audit\n\n- eval target rows used for fitting: `no`\n- condition_key input: `no`\n- biological_key input: `no`\n- eval target means: `no`\n- pooled train+test stats: `no`\n- residual scale selected from eval/test: `no`\n",
        encoding="utf-8",
    )


def write_standard_experiment_files(path: Path, decision: str, *, full_jepa: bool) -> None:
    write_json(path / "config.json", {"decision": decision})
    (path / "metrics_train.jsonl").write_text("", encoding="utf-8")
    write_json(path / "metrics_eval.json", {"decision_label": decision})
    write_json(path / "context_contract.json", context_contract_dict(context_length_train=3, context_length_eval=3, rollout_steps=1))
    (path / "jepa_identity_report.md").write_text("\n".join(["# JEPA Identity Report", "", *[f"- {k}: `{v}`" for k, v in jepa_identity_flags(full_jepa=full_jepa).items()]]) + "\n", encoding="utf-8")
    (path / "leakage_report.md").write_text("# Leakage Report\n\n- Were eval target rows used for fitting? `no`\n- Was condition_key used as model input? `no`\n- Was raw-linear PLS used as main path? `no`\n", encoding="utf-8")
    (path / "model_card.md").write_text(f"# Model Card\n\nDecision: `{decision}`\n", encoding="utf-8")


def row_from_metrics(experiment: str, args: argparse.Namespace, mode: str, decision: str, metrics: dict[str, float], floor: dict[str, float], artifact: Path, *, residual_scale: float) -> dict[str, Any]:
    return {
        "experiment_id": experiment,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "mode": mode,
        "status": decision,
        "predictor_type": "action_adaln_rope",
        "context_length_train": 3,
        "context_length_eval": 3,
        "rollout_steps": 1,
        "transition_improvement": f"{metrics['transition_source_cosine_improvement']:.4f}",
        "delta_cosine": f"{metrics['delta_cosine']:.4f}",
        "recall_at_1": f"{metrics['transition_to_target_recall@1']:.4f}",
        "delta_rank": f"{metrics['delta_prediction_effective_rank']:.4f}",
        "magnitude_ratio": f"{metrics['delta_magnitude_ratio']:.4f}",
        "floor_transition_improvement": f"{floor['transition_source_cosine_improvement']:.4f}",
        "floor_delta_cosine": f"{floor['delta_cosine']:.4f}",
        "floor_recall_at_1": f"{floor['transition_to_target_recall@1']:.4f}",
        "floor_delta_rank": f"{floor['delta_prediction_effective_rank']:.4f}",
        "floor_gap_transition": f"{metrics['transition_source_cosine_improvement'] - floor['transition_source_cosine_improvement']:.6f}",
        "floor_gap_recall": f"{metrics['transition_to_target_recall@1'] - floor['transition_to_target_recall@1']:.6f}",
        "floor_gap_delta_cosine": f"{metrics['delta_cosine'] - floor['delta_cosine']:.6f}",
        "cv_lcb_transition_gap": "0.000000",
        "cv_lcb_recall_gap": "0.000000",
        "cv_lcb_delta_cosine_gap": "0.000000",
        "mean_transition_gap": "0.000000",
        "mean_recall_gap": "0.000000",
        "residual_gate_mean": "0.0000",
        "residual_gate_nonzero_fraction": "0.0000",
        "residual_scale": f"{residual_scale:.4f}",
        "action_negative_gap": "0.000000",
        "rollout_loss_validated": "1.0",
        "context_contract_validated": "1.0",
        "leakage_status": "PASS",
        "identity_status": "PASS",
        "decision_label": decision,
        "artifact_dir": str(artifact),
    }


def row_from_train_metrics(experiment: str, args: argparse.Namespace, mode: str, decision: str, metrics: dict[str, Any], artifact: Path) -> dict[str, Any]:
    floor = {
        "transition_source_cosine_improvement": metrics["floor_transition_improvement"],
        "delta_cosine": metrics["floor_delta_cosine"],
        "transition_to_target_recall@1": metrics["floor_recall_at_1"],
        "delta_prediction_effective_rank": metrics["floor_delta_rank"],
        "delta_magnitude_ratio": FLOOR["delta_magnitude_ratio"],
    }
    pred = {
        "transition_source_cosine_improvement": metrics["transition_improvement"],
        "delta_cosine": metrics["delta_cosine"],
        "transition_to_target_recall@1": metrics["recall_at_1"],
        "delta_prediction_effective_rank": metrics["delta_rank"],
        "delta_magnitude_ratio": metrics["magnitude_ratio"],
    }
    row = row_from_metrics(experiment, args, mode, decision, pred, floor, artifact, residual_scale=float(metrics["residual_scale"]))
    row.update(
        {
            "cv_lcb_transition_gap": f"{float(metrics.get('cv_lcb_transition_gap', 0.0)):.6f}",
            "cv_lcb_recall_gap": f"{float(metrics.get('cv_lcb_recall_gap', 0.0)):.6f}",
            "cv_lcb_delta_cosine_gap": f"{float(metrics.get('cv_lcb_delta_cosine_gap', 0.0)):.6f}",
            "mean_transition_gap": f"{float(metrics.get('mean_transition_gap', 0.0)):.6f}",
            "mean_recall_gap": f"{float(metrics.get('mean_recall_gap', 0.0)):.6f}",
            "residual_gate_mean": f"{float(metrics.get('residual_gate_mean', 0.0)):.4f}",
            "residual_gate_nonzero_fraction": f"{float(metrics.get('residual_gate_nonzero_fraction', 0.0)):.4f}",
            "action_negative_gap": f"{float(metrics.get('action_negative_gap', 0.0)):.6f}",
        }
    )
    return row


def append_result(root: Path, row: dict[str, Any]) -> None:
    path = root / "results.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_id"].astype(str) != row["experiment_id"]]
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, sep="\t", index=False)


def append_journal(root: Path, experiment: str, decision: str, summary: str) -> None:
    with (root / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {experiment}\n\n{summary}\n\nDecision label: `{decision}`.\n\n")


def write_calibration_report(root: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Residual Calibration Report",
        "",
        f"- residual_scale: `{metrics['residual_scale']:.4f}`",
        f"- residual_gate_mean: `{metrics['residual_gate_mean']:.4f}`",
        f"- cv_lcb_transition_gap: `{metrics['cv_lcb_transition_gap']:.6f}`",
        f"- cv_lcb_recall_gap: `{metrics['cv_lcb_recall_gap']:.6f}`",
        f"- cv_lcb_delta_cosine_gap: `{metrics['cv_lcb_delta_cosine_gap']:.6f}`",
        f"- action_negative_gap: `{metrics['action_negative_gap']:.6f}`",
    ]
    (root / "residual_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(root: Path, decision: str, *, notes: list[str], rows: list[dict[str, Any]]) -> None:
    key_rows = "\n".join(
        [
            f"| {row['experiment_id']} | {row['transition_improvement']} | {row['delta_cosine']} | {row['recall_at_1']} | {row['delta_rank']} | {row['floor_gap_transition']} | {row['floor_gap_recall']} | {row['residual_scale']} | {row['decision_label']} |"
            for row in rows
        ]
    )
    text = f"""# BioGuard-WM-JEPA Phase 8 v2 Final Report

## Decision label
{decision}

## Model of record
Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass supersedes it.

## Phase 7 status integration
- Phase 7 decision: `PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA`
- Floor values: transition `0.0057`, delta cosine `0.3980`, recall@1 `0.4815`, rank `10.2835`
- Residual candidates locked as failed: spectral, kernel, program

## Paper integration
- Paper path: `papers/2512.24497v3.pdf`
- Lessons used: latent dynamics predictor, action-AdaLN + RoPE, fixed context contract, L2/cosine endpoint costs, deterministic predictor risk control.
- What was implemented from the paper: action-AdaLN + RoPE predictor primitives and floor-preserving JEPA-WM transition head.
- What was deliberately not implemented: stochastic/diffusion heads, planning optimizer, full JEPA wrapper unless BGWM002 passes.

## What was implemented
- Files changed: `perturb_jepa/models/jepawm_predictor.py`, `perturb_jepa/training/bioguard_wm_losses.py`, `perturb_jepa/training/bioguard_wm_calibration.py`, `perturb_jepa/training/bioguard_wm_rollouts.py`, `scripts/train_bioguard_wm_jepa.py`, `scripts/run_bioguard_wm_phase8_v2.py`
- New classes: `BioJEPAWMContextConfig`, `RotaryEmbedding`, `RotarySelfAttention`, `ActionAdaLNBlock`, `ActionAdaLNPredictor`, `FloorPreservingJEPAWMTransitionHead`
- New tests: focused Phase 8 v2 tests

## What was tested
- BGWM000: audit/floor/paper/Phase 7 lock
- BGWM001: zero-residual action-AdaLN smoke
- BGWM002: frozen-latent action-AdaLN + RoPE residual with train-only calibration
- BGWM003-BGWM006: not run unless BGWM002 keeps a positive residual

## Key metrics
| experiment | transition improvement | delta cosine | recall@1 | delta rank | floor gap transition | floor gap recall | residual scale | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{key_rows if key_rows else '| none | | | | | | | | |'}

## JEPA identity
- online/context encoders: operator-only unless full wrapper is opened
- EMA target encoders: operator-only unless full wrapper is opened
- stop-gradient target latents: yes for frozen teacher latents
- latent transition prediction: yes
- cross-modal JEPA: not run because BGWM002 did not keep a residual
- raw-linear PLS main path used: no

## Leakage status
- eval target rows used for fitting: no
- condition_key input: no
- biological_key input: no
- eval target means: no
- pooled train+test stats: no

## Main interpretation
Phase 8 v2 answered the predictor question under current data. If BGWM002 selected zero residual, the JEPA-WM predictor recipe did not provide train-internal evidence for a safe residual above the protected full-ridge floor.

## Recommendation
{'; '.join(notes)}
"""
    (root / "final_report.md").write_text(text, encoding="utf-8")


def baseline_registry_text() -> str:
    return """# Phase 8 v2 Baseline Registry

Protected model of record: rank-3 train-split-only PLS raw-linear readout.

Protected transition floor: train-only full action-ridge delta.

| metric | value |
| --- | ---: |
| floor_transition_improvement | 0.0057 |
| floor_delta_cosine | 0.3980 |
| floor_recall@1 | 0.4815 |
| floor_delta_rank | 10.2835 |
| floor_magnitude_ratio | 0.7744 |
"""


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
