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

from perturb_jepa.models.biospectral_jepa import (
    FloorPreservingTransitionHead,
    NeuralReducedRankRidgeHead,
    RankLadderTransitionHead,
)
from perturb_jepa.training.biospectral_operator import (
    bundle_features,
    identity_metrics,
    identity_violation,
    load_latent_bundle,
    paired_transition_metrics,
    predict_reduced_rank_ridge_numpy,
    fit_reduced_rank_ridge_numpy,
    write_leakage_report,
)
from perturb_jepa.training.seed import seed_everything


PHASE6_ROOT = Path("outputs/autoresearch_biospectral_jepa_phase6")
PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
FULL_FLOOR = {
    "transition_source_cosine_improvement": 0.0057,
    "delta_cosine": 0.3980,
    "transition_to_target_recall@1": 0.4815,
    "delta_prediction_effective_rank": 10.2835,
    "delta_magnitude_ratio": 0.7744,
}
LOW_RANK_FLOOR = {
    "transition_source_cosine_improvement": 0.0046,
    "delta_cosine": 0.3877,
    "transition_to_target_recall@1": 0.4074,
    "delta_prediction_effective_rank": 6.7681,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 6 BioSpectral-JEPA operator experiments.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "frozen_neural_low_rank_equivalence",
            "frozen_full_ridge_floor_wrapper_zero_residual",
            "frozen_rank_ladder_router",
            "frozen_floor_plus_spectral_residual",
        ),
    )
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--residual-rank", type=int, default=12)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    args = parser.parse_args()

    seed_everything(args.seed)
    PHASE6_ROOT.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    train = load_latent_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")

    if args.mode == "frozen_neural_low_rank_equivalence":
        experiment = "BSJ001"
        metrics, payload = run_neural_low_rank(train, eval_bundle, device=device, rank=args.low_rank, alpha=args.alpha)
        decision = decision_bsj001(metrics)
    elif args.mode == "frozen_full_ridge_floor_wrapper_zero_residual":
        experiment = "BSJ002"
        metrics, payload = run_floor_wrapper(train, eval_bundle, device=device, alpha=args.alpha)
        decision = decision_bsj002(metrics)
    elif args.mode == "frozen_rank_ladder_router":
        experiment = "BSJ003"
        metrics, payload = run_rank_ladder(train, eval_bundle, device=device, alpha=args.alpha)
        decision = decision_bsj003(metrics)
    else:
        experiment = "BSJ004"
        metrics, payload = run_spectral_residual(
            train,
            eval_bundle,
            device=device,
            alpha=args.alpha,
            rank=args.residual_rank,
            steps=args.steps,
            lr=args.lr,
        )
        decision = decision_bsj004(metrics)

    metrics["decision_label"] = decision
    metrics["dataset"] = args.dataset
    metrics["eval_split"] = args.eval_split
    metrics["seed"] = float(args.seed)
    metrics["device_used_cuda"] = float(str(device).startswith("cuda"))
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_json(args.output_dir / "operator_payload.json", payload)
    torch.save(payload, args.output_dir / "operator_checkpoint.pt")
    write_model_card(args.output_dir / "model_card.md", args=args, experiment=experiment, metrics=metrics, decision=decision)
    write_leakage_report(
        args.output_dir / "leakage_report.md",
        train_rows=train.source.shape[0],
        eval_rows=eval_bundle.source.shape[0],
        action_feature_names=[f"action_descriptor_{idx}" for idx in range(train.action.shape[1])],
        mode=args.mode,
    )
    update_results(args=args, experiment=experiment, metrics=metrics, decision=decision)
    append_journal(args=args, experiment=experiment, metrics=metrics, decision=decision)
    update_architecture_log(experiment, args.mode, decision)
    print(json.dumps(metrics, sort_keys=True))
    return 0


def run_neural_low_rank(train, eval_bundle, *, device: torch.device, rank: int, alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    head = NeuralReducedRankRidgeHead(source_dim, action_dim, rank).to(device)
    head.fit(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=alpha)
    with torch.no_grad():
        train_delta = head(tensor(train.source, device), tensor(train.action, device)).cpu().numpy()
        eval_delta = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device)).cpu().numpy()
    analytical = fit_reduced_rank_ridge_numpy(bundle_features(train), train.delta, rank=rank, alpha=alpha)
    analytical_eval_delta = predict_reduced_rank_ridge_numpy(analytical, bundle_features(eval_bundle))
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    metrics["max_abs_eval_delta_diff_vs_analytic"] = float(np.max(np.abs(eval_delta - analytical_eval_delta)))
    payload = {
        "mode": "frozen_neural_low_rank_equivalence",
        "rank": int(rank),
        "linear_weight": head.linear.weight.detach().cpu().tolist(),
        "linear_bias": head.linear.bias.detach().cpu().tolist(),
        "basis": head.basis.detach().cpu().tolist(),
        "delta_mean": head.delta_mean.detach().cpu().tolist(),
    }
    return metrics, payload


def run_floor_wrapper(train, eval_bundle, *, device: torch.device, alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    head = FloorPreservingTransitionHead(source_dim, action_dim, residual_rank=12, hidden_dim=64).to(device)
    head.fit_floor_and_basis(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=alpha)
    with torch.no_grad():
        train_out = head(tensor(train.source, device), tensor(train.action, device))
        eval_out = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device))
    train_delta = train_out["predicted_delta_bio"].cpu().numpy()
    eval_delta = eval_out["predicted_delta_bio"].cpu().numpy()
    floor_eval = eval_out["delta_floor"].cpu().numpy()
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    metrics["residual_to_floor_norm_ratio"] = float(eval_out["residual_to_floor_norm_ratio"].cpu())
    metrics["residual_cap_hit_fraction"] = float(eval_out["residual_cap_hit_fraction"].cpu())
    metrics["max_abs_eval_delta_diff_vs_floor"] = float(np.max(np.abs(eval_delta - floor_eval)))
    payload = floor_payload(head, "frozen_full_ridge_floor_wrapper_zero_residual")
    return metrics, payload


def run_rank_ladder(train, eval_bundle, *, device: torch.device, alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    head = RankLadderTransitionHead(source_dim, action_dim, ranks=(4, 8, 12, 24)).to(device)
    head.fit(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=alpha)
    with torch.no_grad():
        train_out = head(tensor(train.source, device), tensor(train.action, device))
        eval_out = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device))
    train_delta = train_out["delta_ladder"].cpu().numpy()
    eval_delta = eval_out["delta_ladder"].cpu().numpy()
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    metrics["rank_ladder_router_entropy"] = float(eval_out["spectral_entropy"].cpu())
    metrics["rank_ladder_usage_by_action"] = [float(x) for x in eval_out["rank_usage"].cpu().tolist()]
    metrics["rank_ladder_expert_names"] = list(head.expert_names)
    payload = {
        "mode": "frozen_rank_ladder_router",
        "expert_names": list(head.expert_names),
        "rank_usage_by_action": metrics["rank_ladder_usage_by_action"],
        "router_entropy": metrics["rank_ladder_router_entropy"],
        "full_floor": ridge_payload(head.full_floor),
    }
    return metrics, payload


def run_spectral_residual(
    train,
    eval_bundle,
    *,
    device: torch.device,
    alpha: float,
    rank: int,
    steps: int,
    lr: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dim = train.source.shape[1]
    action_dim = train.action.shape[1]
    head = FloorPreservingTransitionHead(source_dim, action_dim, residual_rank=rank, hidden_dim=64, residual_norm_cap=0.25).to(device)
    source = tensor(train.source, device)
    action = tensor(train.action, device)
    target_delta = tensor(train.delta, device)
    head.fit_floor_and_basis(source, action, target_delta, alpha=alpha)
    head.set_residual_gate(1.0)
    optimizer = torch.optim.AdamW(head.residual.parameters(), lr=lr, weight_decay=1.0e-4)
    trace = []
    for step in range(max(0, int(steps))):
        out = head(source, action)
        target_residual = target_delta - out["delta_floor"].detach()
        loss = F.smooth_l1_loss(out["delta_residual"], target_residual)
        loss = loss + 0.1 * F.relu(out["residual_to_floor_norm_ratio"] - 0.25)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.residual.parameters(), 1.0)
        optimizer.step()
        if step in {0, max(0, int(steps)) - 1}:
            trace.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    with torch.no_grad():
        train_out = head(tensor(train.source, device), tensor(train.action, device))
        eval_out = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device))
    train_delta = train_out["predicted_delta_bio"].cpu().numpy()
    eval_delta = eval_out["predicted_delta_bio"].cpu().numpy()
    floor_eval_delta = eval_out["delta_floor"].cpu().numpy()
    residual_eval_delta = eval_out["delta_residual"].cpu().numpy()
    metrics = paired_metrics(train, eval_bundle, train_delta, eval_delta)
    metrics.update(identity_metrics(frozen_latent=True))
    metrics["residual_to_floor_norm_ratio"] = float(eval_out["residual_to_floor_norm_ratio"].cpu())
    metrics["residual_cap_hit_fraction"] = float(eval_out["residual_cap_hit_fraction"].cpu())
    metrics["residual_eval_norm_mean"] = float(np.linalg.norm(residual_eval_delta, axis=1).mean())
    metrics["floor_eval_norm_mean"] = float(np.linalg.norm(floor_eval_delta, axis=1).mean())
    metrics["training_trace"] = trace
    payload = floor_payload(head, "frozen_floor_plus_spectral_residual")
    payload["residual_rank"] = int(rank)
    payload["training_trace"] = trace
    return metrics, payload


def paired_metrics(train, eval_bundle, train_delta: np.ndarray, eval_delta: np.ndarray) -> dict[str, Any]:
    return paired_transition_metrics(
        train,
        eval_bundle,
        train_delta,
        eval_delta,
        floor_eval_improvement=FULL_FLOOR["transition_source_cosine_improvement"],
        floor_eval_delta_cosine=FULL_FLOOR["delta_cosine"],
        floor_eval_recall_at_1=FULL_FLOOR["transition_to_target_recall@1"],
        floor_eval_rank=FULL_FLOOR["delta_prediction_effective_rank"],
    )


def decision_bsj001(metrics: dict[str, Any]) -> str:
    ok = (
        not identity_violation(metrics)
        and abs(metrics["operator_eval_transition_improvement"] - LOW_RANK_FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(metrics["operator_eval_delta_cosine"] - LOW_RANK_FLOOR["delta_cosine"]) <= 1.0e-4
        and abs(metrics["operator_eval_recall_at_1"] - LOW_RANK_FLOOR["transition_to_target_recall@1"]) <= 1.0e-4
        and abs(metrics["operator_predicted_delta_rank"] - LOW_RANK_FLOOR["delta_prediction_effective_rank"]) <= 1.0e-3
        and metrics["max_abs_eval_delta_diff_vs_analytic"] <= 1.0e-5
    )
    return "BSJ001_KEEP_NEURAL_LOWRANK_MATCHES_ANALYTIC" if ok else "BSJ001_DISCARD_NEURAL_LOWRANK_MISMATCH"


def decision_bsj002(metrics: dict[str, Any]) -> str:
    ok = (
        not identity_violation(metrics)
        and abs(metrics["operator_eval_transition_improvement"] - FULL_FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(metrics["operator_eval_delta_cosine"] - FULL_FLOOR["delta_cosine"]) <= 1.0e-4
        and abs(metrics["operator_eval_recall_at_1"] - FULL_FLOOR["transition_to_target_recall@1"]) <= 1.0e-4
        and abs(metrics["operator_predicted_delta_rank"] - FULL_FLOOR["delta_prediction_effective_rank"]) <= 1.0e-3
        and metrics["residual_to_floor_norm_ratio"] <= 1.0e-6
    )
    return "BSJ002_KEEP_FLOOR_WRAPPER_MATCHES_FULL_RIDGE" if ok else "BSJ002_DISCARD_FLOOR_WRAPPER_DRIFT"


def decision_bsj003(metrics: dict[str, Any]) -> str:
    ok = (
        not identity_violation(metrics)
        and metrics["operator_eval_transition_improvement"] >= FULL_FLOOR["transition_source_cosine_improvement"] - 1.0e-4
        and metrics["operator_eval_delta_cosine"] >= FULL_FLOOR["delta_cosine"] - 1.0e-4
        and metrics["operator_eval_recall_at_1"] >= FULL_FLOOR["transition_to_target_recall@1"] - 1.0e-4
        and metrics["operator_predicted_delta_rank"] >= 10.0
        and "rank_ladder_router_entropy" in metrics
    )
    return "BSJ003_KEEP_RANK_LADDER_PRESERVES_FLOOR" if ok else "BSJ003_DISCARD_ROUTER_DEGRADES_FLOOR"


def decision_bsj004(metrics: dict[str, Any]) -> str:
    if identity_violation(metrics):
        return "BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR"
    below_floor = (
        metrics["operator_eval_transition_improvement"] < FULL_FLOOR["transition_source_cosine_improvement"] - 1.0e-4
        or metrics["operator_eval_delta_cosine"] < FULL_FLOOR["delta_cosine"] - 1.0e-4
        or metrics["operator_eval_recall_at_1"] < FULL_FLOOR["transition_to_target_recall@1"] - 1.0e-4
    )
    if below_floor:
        return "BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR"
    keep = (
        metrics["operator_eval_transition_improvement"] >= 0.0075
        and metrics["operator_eval_delta_cosine"] >= FULL_FLOOR["delta_cosine"]
        and metrics["operator_eval_recall_at_1"] >= FULL_FLOOR["transition_to_target_recall@1"]
        and metrics["operator_predicted_delta_rank"] >= 10.0
        and 0.50 <= metrics["operator_eval_delta_magnitude_ratio"] <= 1.50
        and metrics["residual_to_floor_norm_ratio"] <= 0.25
        and metrics["residual_cap_hit_fraction"] <= 0.25
        and metrics["floor_gap_transition_improvement"] >= 0.0015
    )
    if keep:
        return "BSJ004_KEEP_SPECTRAL_RESIDUAL_ABOVE_FLOOR"
    if metrics["operator_eval_transition_improvement"] > 0.0:
        return "BSJ004_NEARMISS_OPERATOR_ABOVE_FLOOR_WEAK_SIGNAL"
    return "BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR"


def update_results(*, args: argparse.Namespace, experiment: str, metrics: dict[str, Any], decision: str) -> None:
    path = PHASE6_ROOT / "results.tsv"
    if not path.exists():
        path.write_text(
            "experiment_id\tstage\tmode\tseed\tdataset\teval_split\ttransition_improvement\tdelta_cosine\trecall_at_1\tdelta_rank\tmagnitude_ratio\tfloor_gap\tidentity_pass\tdecision\tartifact_dir\n",
            encoding="utf-8",
        )
    row = {
        "experiment_id": experiment,
        "stage": stage_for(experiment),
        "mode": args.mode,
        "seed": args.seed,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "transition_improvement": f"{metrics['operator_eval_transition_improvement']:.4f}",
        "delta_cosine": f"{metrics['operator_eval_delta_cosine']:.4f}",
        "recall_at_1": f"{metrics['operator_eval_recall_at_1']:.4f}",
        "delta_rank": f"{metrics['operator_predicted_delta_rank']:.4f}",
        "magnitude_ratio": f"{metrics['operator_eval_delta_magnitude_ratio']:.4f}",
        "floor_gap": f"{metrics['floor_gap_transition_improvement']:.4f}",
        "identity_pass": "0.0" if identity_violation(metrics) else "1.0",
        "decision": decision,
        "artifact_dir": str(args.output_dir),
    }
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_id"].astype(str) != experiment]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def append_journal(*, args: argparse.Namespace, experiment: str, metrics: dict[str, Any], decision: str) -> None:
    with (PHASE6_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {experiment}: {args.mode}\n\n"
            f"**Implementation**: mode `{args.mode}`, seed `{args.seed}`, device `{args.device}`, steps `{args.steps}`.\n\n"
            f"**Result**: eval transition improvement `{metrics['operator_eval_transition_improvement']:.4f}`, delta cosine `{metrics['operator_eval_delta_cosine']:.4f}`, recall@1 `{metrics['operator_eval_recall_at_1']:.4f}`, delta rank `{metrics['operator_predicted_delta_rank']:.4f}`, magnitude ratio `{metrics['operator_eval_delta_magnitude_ratio']:.4f}`, floor gap `{metrics['floor_gap_transition_improvement']:.4f}`.\n\n"
            f"**Decision label**: `{decision}`.\n\n"
        )


def update_architecture_log(experiment: str, mode: str, decision: str) -> None:
    with (PHASE6_ROOT / "architectural_changes_log.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {experiment}\n\n- Mode: `{mode}`\n- Decision: `{decision}`\n")


def write_model_card(path: Path, *, args: argparse.Namespace, experiment: str, metrics: dict[str, Any], decision: str) -> None:
    lines = [
        f"# {experiment} Model Card",
        "",
        f"- Mode: `{args.mode}`",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Decision: `{decision}`",
        "- Role: Phase 6 Tier 1 operator/contract probe only; cannot promote model of record.",
        "- Protected model of record remains rank-3 train-split-only PLS raw-linear readout.",
        "",
        "## Metrics",
        "",
        f"- eval transition improvement: `{metrics['operator_eval_transition_improvement']:.4f}`",
        f"- eval delta cosine: `{metrics['operator_eval_delta_cosine']:.4f}`",
        f"- eval recall@1: `{metrics['operator_eval_recall_at_1']:.4f}`",
        f"- eval median rank: `{metrics['operator_eval_median_rank']:.4f}`",
        f"- eval delta rank: `{metrics['operator_predicted_delta_rank']:.4f}`",
        f"- eval magnitude ratio: `{metrics['operator_eval_delta_magnitude_ratio']:.4f}`",
        f"- floor gap: `{metrics['floor_gap_transition_improvement']:.4f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def floor_payload(head: FloorPreservingTransitionHead, mode: str) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "residual_gate": float(head.residual_gate.detach().cpu()),
        "ridge_floor": ridge_payload(head.ridge_floor),
        "spectral_basis_teacher_delta_eigenvalues": head.spectral_basis.teacher_delta_eigenvalues.detach().cpu().tolist(),
    }
    return payload


def ridge_payload(head) -> dict[str, Any]:
    return {
        "x_mean": head.x_mean.detach().cpu().tolist(),
        "y_mean": head.y_mean.detach().cpu().tolist(),
        "coef": head.coef.detach().cpu().tolist(),
    }


def stage_for(experiment: str) -> str:
    return {
        "BSJ001": "StageC",
        "BSJ002": "StageC",
        "BSJ003": "StageC",
        "BSJ004": "StageD",
    }.get(experiment, "unknown")


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
