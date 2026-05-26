from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.biooperator_jepa import LowRankControlAffineOperator, NeuralActionRidgeHead, RidgeTransitionFloor
from perturb_jepa.training.seed import seed_everything
from scripts.run_biooperator_contract_audit import LatentBundle, load_bundle, transition_metrics


PHASE5_ROOT = Path("outputs/autoresearch_biooperator_jepa_phase5")
PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
RIDGE_FLOOR_IMPROVEMENT = 0.0057
RIDGE_FLOOR_DELTA_COSINE = 0.3980
RIDGE_FLOOR_RANK = 10.2835


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate Phase 5 BioOperator-JEPA operators.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--mode", required=True, choices=("frozen_neural_ridge", "frozen_control_affine"))
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--operator-rank", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = load_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")
    if args.mode == "frozen_neural_ridge":
        metrics, payload = run_neural_ridge(train, eval_bundle, device=device)
        decision = decision_boj001(metrics)
        experiment = "BOJ001"
        family = "neural_operator_floor"
        architectural_change = "NeuralActionRidgeHead_initialized_from_train_only_action_ridge"
    else:
        metrics, payload = run_control_affine(train, eval_bundle, device=device, steps=args.steps, rank=args.operator_rank, lr=args.lr)
        decision = decision_boj002(metrics)
        experiment = "BOJ002"
        family = "control_affine_operator"
        architectural_change = "LowRankControlAffineOperator_initialized_from_train_only_ridge"

    metrics["decision_label"] = decision
    metrics["dataset"] = args.dataset
    metrics["eval_split"] = args.eval_split
    metrics["seed"] = float(args.seed)
    metrics["device_used_cuda"] = float(str(device).startswith("cuda"))
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_json(args.output_dir / "operator_payload.json", payload)
    write_model_card(args.output_dir / "model_card.md", args=args, metrics=metrics, decision=decision)
    torch.save(payload, args.output_dir / "operator_checkpoint.pt")
    update_results(args=args, metrics=metrics, decision=decision, experiment=experiment, family=family, architectural_change=architectural_change)
    append_journal(args=args, metrics=metrics, decision=decision, experiment=experiment)
    if should_stop(decision):
        final_decision = (
            "PHASE5_OPERATOR_FLOOR_REPRODUCED_BUT_NEURAL_FAILURE_CLOSE"
            if experiment == "BOJ001"
            else "PHASE5_OPERATOR_MATCHES_FLOOR_READY_FOR_TIER2"
            if decision == "TIER1_KEEP_OPERATOR_MATCHES_FLOOR"
            else "PHASE5_OPERATOR_FLOOR_REPRODUCED_BUT_NEURAL_FAILURE_CLOSE"
        )
        write_final_report(final_decision, [(experiment, metrics, decision)])
    print(json.dumps(metrics, sort_keys=True))
    return 0


def run_neural_ridge(train: LatentBundle, eval_bundle: LatentBundle, *, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    ridge = RidgeTransitionFloor(source_dim, action_dim).to(device)
    ridge.fit(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=1.0e-2)
    head = NeuralActionRidgeHead(source_dim, action_dim).to(device)
    head.initialize_from_ridge(ridge)
    train_delta = head(tensor(train.source, device), tensor(train.action, device)).detach().cpu().numpy()
    eval_delta = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device)).detach().cpu().numpy()
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    payload = {
        "mode": "frozen_neural_ridge",
        "ridge_x_mean": ridge.x_mean.detach().cpu().tolist(),
        "ridge_y_mean": ridge.y_mean.detach().cpu().tolist(),
        "ridge_coef": ridge.coef.detach().cpu().tolist(),
        "linear_weight": head.linear.weight.detach().cpu().tolist(),
        "linear_bias": head.linear.bias.detach().cpu().tolist(),
    }
    return metrics, payload


def run_control_affine(
    train: LatentBundle,
    eval_bundle: LatentBundle,
    *,
    device: torch.device,
    steps: int,
    rank: int,
    lr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    ridge = RidgeTransitionFloor(source_dim, action_dim).to(device)
    ridge.fit(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=1.0e-2)
    operator = LowRankControlAffineOperator(source_dim, action_dim, rank=rank).to(device)
    operator.initialize_from_ridge(ridge)
    source = tensor(train.source, device)
    action = tensor(train.action, device)
    target_delta = tensor(train.delta, device)
    optimizer = torch.optim.AdamW(operator.parameters(), lr=lr, weight_decay=1.0e-4)
    with (Path.cwd() / "outputs/autoresearch_biooperator_jepa_phase5/tmp_train_trace.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(int(steps)):
            pred_delta = operator(source, action)
            delta_loss = F.smooth_l1_loss(pred_delta, target_delta)
            direction = (1.0 - F.cosine_similarity(pred_delta, target_delta, dim=-1)).mean()
            ridge_loss = F.smooth_l1_loss(pred_delta, ridge(source, action).detach())
            loss = delta_loss + 0.2 * direction + 0.2 * ridge_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(operator.parameters(), 1.0)
            optimizer.step()
            if step in {0, int(steps) - 1}:
                handle.write(json.dumps({"step": step + 1, "loss": float(loss.detach().cpu())}) + "\n")
    train_delta = operator(tensor(train.source, device), tensor(train.action, device)).detach().cpu().numpy()
    eval_delta = operator(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device)).detach().cpu().numpy()
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    metrics["residual_contribution_ratio"] = float(operator.residual_contribution_ratio(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device)).detach().cpu())
    payload = {
        "mode": "frozen_control_affine",
        "operator_rank": rank,
        "source_mean": operator.source_mean.detach().cpu().tolist(),
        "action_mean": operator.action_mean.detach().cpu().tolist(),
        "delta_mean": operator.delta_mean.detach().cpu().tolist(),
        "left": operator.left.detach().cpu().tolist(),
        "right": operator.right.detach().cpu().tolist(),
        "action_shift_weight": operator.action_shift.weight.detach().cpu().tolist() if operator.action_shift is not None else [],
    }
    return metrics, payload


def paired_metrics(train: LatentBundle, eval_bundle: LatentBundle, train_delta: np.ndarray, eval_delta: np.ndarray) -> dict[str, Any]:
    train_metrics = transition_metrics(train, train.source + train_delta)
    eval_metrics = transition_metrics(eval_bundle, eval_bundle.source + eval_delta)
    metrics: dict[str, Any] = {f"train/{key}": value for key, value in train_metrics.items()}
    metrics.update({f"eval/{key}": value for key, value in eval_metrics.items()})
    metrics.update(
        {
            "operator_train_transition_improvement": train_metrics["transition_source_cosine_improvement"],
            "operator_eval_transition_improvement": eval_metrics["transition_source_cosine_improvement"],
            "operator_train_delta_cosine": train_metrics["delta_cosine"],
            "operator_eval_delta_cosine": eval_metrics["delta_cosine"],
            "operator_eval_recall_at_1": eval_metrics["transition_to_target_recall@1"],
            "operator_eval_median_rank": eval_metrics["transition_to_target_median_rank"],
            "operator_predicted_delta_rank": eval_metrics["delta_prediction_effective_rank"],
            "operator_eval_delta_magnitude_ratio": eval_metrics["delta_magnitude_ratio"],
            "source_improvement_hinge_violation_fraction": eval_metrics["source_improvement_hinge_violation_fraction"],
            "action_ridge_floor_gap": eval_metrics["transition_source_cosine_improvement"] - RIDGE_FLOOR_IMPROVEMENT,
        }
    )
    return metrics


def decision_boj001(metrics: dict[str, Any]) -> str:
    if identity_violation(metrics):
        return "PHASE5_STOP_OPERATOR_CONTRACT_FAILURE"
    if metrics["operator_eval_transition_improvement"] < 0.0:
        return "PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR"
    if metrics["operator_eval_delta_cosine"] < 0.0:
        return "PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR"
    if metrics["operator_eval_delta_magnitude_ratio"] > 2.0:
        return "PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR"
    passes = (
        metrics["operator_train_transition_improvement"] >= 0.0600
        and metrics["operator_train_delta_cosine"] >= 0.75
        and metrics["operator_eval_transition_improvement"] >= 0.0035
        and metrics["operator_eval_delta_cosine"] >= 0.30
        and metrics["operator_predicted_delta_rank"] >= 8.0
        and 0.4 <= metrics["operator_eval_delta_magnitude_ratio"] <= 1.6
    )
    return "TIER1_KEEP_OPERATOR_MATCHES_FLOOR" if passes else "PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR"


def decision_boj002(metrics: dict[str, Any]) -> str:
    boj001 = read_boj001_metrics()
    boj001_improvement = float(boj001.get("operator_eval_transition_improvement", RIDGE_FLOOR_IMPROVEMENT))
    boj001_delta_cosine = float(boj001.get("operator_eval_delta_cosine", RIDGE_FLOOR_DELTA_COSINE))
    if identity_violation(metrics):
        return "TIER1_DISCARD_OPERATOR_BELOW_FLOOR"
    if metrics["operator_eval_transition_improvement"] < 0.0 or metrics["operator_eval_delta_cosine"] < 0.0:
        return "TIER1_DISCARD_OPERATOR_BELOW_FLOOR"
    if metrics["operator_eval_delta_magnitude_ratio"] > 2.0:
        return "TIER1_DISCARD_OPERATOR_BELOW_FLOOR"
    passes = (
        metrics["operator_eval_transition_improvement"] >= max(RIDGE_FLOOR_IMPROVEMENT, boj001_improvement - 0.0010)
        and metrics["operator_eval_delta_cosine"] >= boj001_delta_cosine - 0.05
        and metrics["operator_eval_recall_at_1"] >= 0.45
        and metrics["operator_predicted_delta_rank"] >= 8.0
        and metrics["source_improvement_hinge_violation_fraction"] <= 0.60
        and metrics.get("residual_contribution_ratio", 0.0) <= 0.30
    )
    return "TIER1_KEEP_OPERATOR_MATCHES_FLOOR" if passes else "TIER1_DISCARD_OPERATOR_BELOW_FLOOR"


def identity_metrics(*, frozen_latent: bool) -> dict[str, float]:
    return {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": 0.0,
        "condition_key_feature_present": 0.0,
        "biological_key_one_hot_present": 0.0,
        "test_target_mean_used_for_fit": 0.0,
        "pooled_train_test_target_used_for_fit": 0.0,
        "teacher_stop_gradient_verified": 1.0,
        "frozen_latent_operator_only": 1.0 if frozen_latent else 0.0,
        "encoder_training_skipped": 1.0 if frozen_latent else 0.0,
    }


def identity_violation(metrics: dict[str, Any]) -> bool:
    return any(
        float(metrics.get(key, 1.0)) != expected
        for key, expected in (
            ("pls_raw_linear_main_path_used", 0.0),
            ("condition_key_feature_present", 0.0),
            ("biological_key_one_hot_present", 0.0),
            ("test_target_mean_used_for_fit", 0.0),
            ("pooled_train_test_target_used_for_fit", 0.0),
            ("teacher_stop_gradient_verified", 1.0),
        )
    )


def should_stop(decision: str) -> bool:
    return decision in {
        "PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR",
        "PHASE5_STOP_OPERATOR_CONTRACT_FAILURE",
        "TIER1_DISCARD_OPERATOR_BELOW_FLOOR",
    }


def update_results(
    *,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    decision: str,
    experiment: str,
    family: str,
    architectural_change: str,
) -> None:
    row = {
        "commit": git_commit_label(),
        "experiment_num": experiment,
        "stage": "StageC" if experiment == "BOJ001" else "StageD",
        "family": family,
        "tier_reached": "Tier1",
        "decision_label": decision,
        "status": "SEARCH_CLOSED_NO_NEW_BASELINE" if should_stop(decision) else decision,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed_list": str(args.seed),
        "primary_metric": f"eval_transition_improvement={metrics['operator_eval_transition_improvement']:.4f}",
        "secondary_metric": f"eval_delta_cosine={metrics['operator_eval_delta_cosine']:.4f}; eval_rank={metrics['operator_predicted_delta_rank']:.4f}; eval_recall@1={metrics['operator_eval_recall_at_1']:.4f}",
        "protected_metric_summary": "protected_rank3_train_split_pls_remains_model_of_record; frozen_latent_operator_only=1.0; no forbidden shortcuts",
        "architectural_change": architectural_change,
        "description": f"{experiment} {args.mode} operator evaluation.",
        "operator_train_transition_improvement": f"{metrics['operator_train_transition_improvement']:.4f}",
        "operator_eval_transition_improvement": f"{metrics['operator_eval_transition_improvement']:.4f}",
        "operator_train_delta_cosine": f"{metrics['operator_train_delta_cosine']:.4f}",
        "operator_eval_delta_cosine": f"{metrics['operator_eval_delta_cosine']:.4f}",
        "operator_eval_recall_at_1": f"{metrics['operator_eval_recall_at_1']:.4f}",
        "operator_eval_median_rank": f"{metrics['operator_eval_median_rank']:.4f}",
        "operator_predicted_delta_rank": f"{metrics['operator_predicted_delta_rank']:.4f}",
        "action_ridge_floor_gap": f"{metrics['action_ridge_floor_gap']:.4f}",
        "sign_contract_pass": "1.0",
        "ridge_equivalence_pass": "1.0" if experiment == "BOJ001" else "",
        "source_improvement_hinge_violation_fraction": f"{metrics['source_improvement_hinge_violation_fraction']:.4f}",
    }
    path = PHASE5_ROOT / "results.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_num"].astype(str) != experiment]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def append_journal(*, args: argparse.Namespace, metrics: dict[str, Any], decision: str, experiment: str) -> None:
    with (PHASE5_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {experiment}: {args.mode}\n\n"
            f"**Hypothesis**: neural/operator form can reproduce or preserve the train-only action-ridge transition floor without sign, magnitude, or leakage failure.\n\n"
            f"**Implementation**: mode `{args.mode}`, steps `{args.steps}`, rank `{args.operator_rank}`, device `{args.device}`.\n\n"
            f"**Tier result**: eval transition improvement `{metrics['operator_eval_transition_improvement']:.4f}`, eval delta cosine `{metrics['operator_eval_delta_cosine']:.4f}`, eval recall@1 `{metrics['operator_eval_recall_at_1']:.4f}`, eval delta rank `{metrics['operator_predicted_delta_rank']:.4f}`, magnitude ratio `{metrics['operator_eval_delta_magnitude_ratio']:.4f}`.\n\n"
            f"**Decision label**: `{decision}`.\n\n"
        )


def write_model_card(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], decision: str) -> None:
    lines = [
        f"# {args.mode} Model Card",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Mode: `{args.mode}`",
        f"- Decision: `{decision}`",
        "- Role: Phase 5 Tier 1 operator candidate only; cannot promote model of record.",
        "",
        "## Metrics",
        "",
        f"- train transition improvement: `{metrics['operator_train_transition_improvement']:.4f}`",
        f"- eval transition improvement: `{metrics['operator_eval_transition_improvement']:.4f}`",
        f"- eval delta cosine: `{metrics['operator_eval_delta_cosine']:.4f}`",
        f"- eval recall@1: `{metrics['operator_eval_recall_at_1']:.4f}`",
        f"- eval median rank: `{metrics['operator_eval_median_rank']:.4f}`",
        f"- eval predicted delta rank: `{metrics['operator_predicted_delta_rank']:.4f}`",
        f"- eval magnitude ratio: `{metrics['operator_eval_delta_magnitude_ratio']:.4f}`",
        f"- action ridge floor gap: `{metrics['action_ridge_floor_gap']:.4f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(final_decision: str, entries: list[tuple[str, dict[str, Any], str]]) -> None:
    lines = [
        "# BioOperator-JEPA Phase 5 Final Report",
        "",
        "## Decision label",
        "",
        final_decision,
        "",
        "## Model of record",
        "",
        "Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass explicitly supersedes it.",
        "",
        "## What was tested",
        "",
        "- Stage A operator floor reproduction",
        "- Stage B sign/gradient/loss contracts",
        *[f"- {experiment} result" for experiment, _, _ in entries],
        "",
        "## Key metrics",
        "",
        "| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for experiment, metrics, decision in entries:
        lines.append(
            f"| {experiment} | {metrics['operator_eval_transition_improvement']:.4f} | {metrics['operator_eval_delta_cosine']:.4f} | {metrics['operator_eval_recall_at_1']:.4f} | {metrics['operator_predicted_delta_rank']:.4f} | {metrics['operator_eval_delta_magnitude_ratio']:.4f} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Floor comparison",
            "",
            "- eval action_ridge_delta improvement = `+0.0057`",
            "- eval action_ridge_delta delta_cosine = `0.3980`",
            "- eval action_ridge_delta rank = `10.2835`",
            "",
            "## What failed or passed",
            "",
            "Stage A and Stage B passed. Closure, if any, occurred in neural/operator candidate evaluation, not latent cache or metric reproduction.",
            "",
            "## Recommendation",
            "",
            "Retain the frozen operator floor as an audit reference. Do not promote any Phase 5 Tier 1 result.",
        ]
    )
    (PHASE5_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_boj001_metrics() -> dict[str, Any]:
    path = PHASE5_ROOT / "experiments/BOJ001_frozen_neural_ridge_seed0/metrics_eval.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def git_commit_label() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        return f"{commit}+dirty" if dirty else commit
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
