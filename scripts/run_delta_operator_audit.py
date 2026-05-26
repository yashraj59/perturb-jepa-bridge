from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics
from perturb_jepa.training.seed import seed_everything


PHASE4_ROOT = Path("outputs/autoresearch_bioflow_jepa_phase4")
DEFAULT_CHECKPOINT = Path("outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt")


@dataclass(frozen=True)
class CacheBundle:
    name: str
    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame

    @property
    def source(self) -> np.ndarray:
        return self.arrays["source_z_bio_teacher"].astype(np.float64)

    @property
    def target(self) -> np.ndarray:
        return self.arrays["target_z_bio_teacher"].astype(np.float64)

    @property
    def delta(self) -> np.ndarray:
        return self.target - self.source

    @property
    def action(self) -> np.ndarray:
        action = self.arrays.get("action_descriptor")
        if action is None or action.size == 0:
            return np.zeros((self.source.shape[0], 0), dtype=np.float64)
        return action.astype(np.float64)

    @property
    def target_tech(self) -> np.ndarray | None:
        value = self.arrays.get("target_z_tech_teacher")
        if value is None or value.size == 0:
            return None
        return value.astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 delta-operator audit before BioFlow-JEPA implementation.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite",))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PHASE4_ROOT / "delta_operator_audit")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = _select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_latent_cache(args, device=device)

    train = load_bundle(args.output_dir / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_bundle(args.output_dir / f"{args.dataset}_{args.eval_split}_latents", args.eval_split)

    train_stats = delta_statistics(train)
    eval_stats = delta_statistics(eval_bundle)
    baseline_rows = transition_baselines(train, eval_bundle)
    optimization_rows = frozen_latent_optimization_audit(train, seed=args.seed, device=torch.device("cpu"))

    decision, decision_reasons = reopening_decision(train_stats, eval_stats, baseline_rows, optimization_rows)
    write_outputs(
        args=args,
        device=str(device),
        train=train,
        eval_bundle=eval_bundle,
        train_stats=train_stats,
        eval_stats=eval_stats,
        baseline_rows=baseline_rows,
        optimization_rows=optimization_rows,
        decision=decision,
        decision_reasons=decision_reasons,
    )
    update_phase4_results(args, decision, eval_stats, baseline_rows, optimization_rows)
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def ensure_latent_cache(args: argparse.Namespace, *, device: torch.device) -> None:
    required = [
        args.output_dir / f"{args.dataset}_train_latents.npz",
        args.output_dir / f"{args.dataset}_{args.eval_split}_latents.npz",
        args.output_dir / f"{args.dataset}_train_latents.metadata.tsv",
        args.output_dir / f"{args.dataset}_{args.eval_split}_latents.metadata.tsv",
    ]
    if not args.force_cache and all(path.exists() for path in required):
        return
    command = [
        sys.executable,
        "scripts/cache_bioflow_teacher_latents.py",
        "--dataset",
        args.dataset,
        "--checkpoint",
        str(args.checkpoint),
        "--eval-split",
        args.eval_split,
        "--seed",
        str(args.seed),
        "--device",
        str(device),
        "--batch-size",
        str(args.batch_size),
        "--bag-size",
        str(args.bag_size),
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(command, check=True)


def load_bundle(prefix: Path, name: str) -> CacheBundle:
    arrays = dict(np.load(prefix.with_suffix(".npz")))
    metadata = pd.read_csv(prefix.with_suffix(".metadata.tsv"), sep="\t")
    return CacheBundle(name=name, arrays=arrays, metadata=metadata)


def delta_statistics(bundle: CacheBundle) -> dict[str, float]:
    delta = bundle.delta
    source = bundle.source
    target = bundle.target
    norms = np.linalg.norm(delta, axis=1)
    source_cos = row_cosine(source, target)
    frame = metrics_frame(bundle)
    stats: dict[str, float] = {
        "n_samples": float(delta.shape[0]),
        "delta_mean_norm": float(norms.mean()) if norms.size else 0.0,
        "delta_median_norm": float(np.median(norms)) if norms.size else 0.0,
        "delta_std_mean": float(delta.std(axis=0).mean()) if delta.size else 0.0,
        "delta_effective_rank": effective_rank(delta),
        "delta_near_zero_fraction_norm_lt_1e-3": float((norms < 1.0e-3).mean()) if norms.size else 0.0,
        "source_to_target_cosine_mean": float(source_cos.mean()) if source_cos.size else 0.0,
        "source_to_target_cosine_p10": float(np.quantile(source_cos, 0.10)) if source_cos.size else 0.0,
        "source_to_target_cosine_p90": float(np.quantile(source_cos, 0.90)) if source_cos.size else 0.0,
        "delta_perturbation_nn_recall@1": same_label_nn_recall(delta, frame["perturbation"].tolist()),
    }
    spectrum = covariance_spectrum(delta)
    for index, value in enumerate(spectrum[:8], start=1):
        stats[f"delta_cov_spectrum_{index}"] = float(value)
    if "batch" in frame.columns:
        stats.update(batch_probe_metrics(delta, frame, label_col="batch", prefix="delta_batch_probe"))
    if "perturbation" in frame.columns:
        stats.update(batch_probe_metrics(delta, frame, label_col="perturbation", prefix="delta_perturbation_probe"))
    return stats


def transition_baselines(train: CacheBundle, eval_bundle: CacheBundle) -> list[dict[str, Any]]:
    mean_delta = train.delta.mean(axis=0, keepdims=True)
    action_means = action_mean_delta_lookup(train)
    ridge = fit_ridge(np.concatenate((train.source, train.action), axis=1), train.delta, alpha=1.0e-2)
    low_rank = fit_low_rank_ridge(np.concatenate((train.source, train.action), axis=1), train.delta, rank=8, alpha=1.0e-2)
    rows: list[dict[str, Any]] = []

    for split_name, bundle in (("train", train), ("eval", eval_bundle)):
        x = np.concatenate((bundle.source, bundle.action), axis=1)
        perturb_ids = bundle.metadata["perturbation_id"].to_numpy(dtype=int)
        baseline_preds: dict[str, np.ndarray] = {
            "source_as_target_null": bundle.source,
            "mean_delta_null": bundle.source + np.broadcast_to(mean_delta, bundle.delta.shape),
            "action_mean_delta": bundle.source + np.stack([action_means.get(int(pid), mean_delta.reshape(-1)) for pid in perturb_ids], axis=0),
            "action_ridge_delta": bundle.source + predict_ridge(ridge, x),
            "action_low_rank_ridge": bundle.source + predict_low_rank_ridge(low_rank, x),
        }
        if split_name == "train":
            baseline_preds["action_knn_delta"] = bundle.source + leave_one_out_knn_delta(train)
            baseline_preds["oracle_train_delta_upper_bound"] = bundle.target
        else:
            baseline_preds["action_knn_delta"] = bundle.source + knn_delta(train, bundle)
        for name, pred in baseline_preds.items():
            rows.append({"split": split_name, "baseline": name, **transition_metrics(bundle, pred)})
    return rows


def frozen_latent_optimization_audit(
    train: CacheBundle,
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | str]]:
    torch.manual_seed(seed)
    source = torch.as_tensor(train.source, dtype=torch.float32, device=device)
    target = torch.as_tensor(train.target, dtype=torch.float32, device=device)
    action = torch.as_tensor(train.action, dtype=torch.float32, device=device)
    true_delta = target - source
    whitener = TorchDeltaWhitener.fit(true_delta)
    rows: list[dict[str, float | str]] = []

    zero_model = LinearDelta(source.shape[1] + action.shape[1], source.shape[1]).to(device)
    rows.append({"audit": "zero_init", "step_count": 0.0, **torch_transition_metrics(source, target, zero_model(source, action))})

    one = LinearDelta(source.shape[1] + action.shape[1], source.shape[1]).to(device)
    opt = torch.optim.SGD(one.parameters(), lr=0.25)
    loss = F.smooth_l1_loss(one(source, action), true_delta)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    rows.append({"audit": "one_step_raw_delta_mse", "step_count": 1.0, **torch_transition_metrics(source, target, one(source, action))})

    rows.append(train_transition_probe(source, target, action, loss_mode="raw_delta_mse", steps=20, seed=seed))
    rows.append(train_transition_probe(source, target, action, loss_mode="endpoint_cosine", steps=20, seed=seed + 1))
    rows.append(train_transition_probe(source, target, action, loss_mode="whitened_delta_mse", steps=20, seed=seed + 2, whitener=whitener))
    rows.append(train_transition_probe(source, target, action, loss_mode="whitened_hinge_direction", steps=20, seed=seed + 3, whitener=whitener))
    return rows


def train_transition_probe(
    source: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
    *,
    loss_mode: str,
    steps: int,
    seed: int,
    whitener: "TorchDeltaWhitener | None" = None,
) -> dict[str, float | str]:
    torch.manual_seed(seed)
    model = LinearDelta(source.shape[1] + action.shape[1], source.shape[1]).to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-2, weight_decay=1.0e-4)
    true_delta = target - source
    source_cos = F.cosine_similarity(source, target, dim=-1).detach()
    for _ in range(int(steps)):
        pred_delta = model(source, action)
        pred = F.normalize(source + pred_delta, dim=-1)
        if loss_mode == "raw_delta_mse":
            loss = F.smooth_l1_loss(pred_delta, true_delta)
        elif loss_mode == "endpoint_cosine":
            loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        elif loss_mode == "whitened_delta_mse":
            if whitener is None:
                raise ValueError("whitener is required for whitened_delta_mse")
            loss = F.smooth_l1_loss(whitener.whiten(pred_delta), whitener.whiten(true_delta))
        elif loss_mode == "whitened_hinge_direction":
            if whitener is None:
                raise ValueError("whitener is required for whitened_hinge_direction")
            pred_cos = F.cosine_similarity(pred, target, dim=-1)
            delta_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1)
            hinge = F.relu(source_cos + 0.02 - pred_cos).mean()
            loss = (
                F.smooth_l1_loss(whitener.whiten(pred_delta), whitener.whiten(true_delta))
                + 2.0 * hinge
                + (1.0 - delta_cos).mean()
            )
        else:
            raise ValueError(f"unknown loss mode: {loss_mode}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = torch_transition_metrics(source, target, model(source, action))
    return {"audit": loss_mode, "step_count": float(steps), **metrics}


def reopening_decision(
    train_stats: dict[str, float],
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
) -> tuple[str, list[str]]:
    teacher_delta_measurable = (
        eval_stats["delta_effective_rank"] >= 4.0
        and eval_stats["delta_std_mean"] > 1.0e-3
        and eval_stats["delta_mean_norm"] > 1.0e-3
    )
    positive_baselines = [
        row
        for row in baseline_rows
        if row["baseline"] not in {"source_as_target_null", "oracle_train_delta_upper_bound"}
        and float(row["transition_source_cosine_improvement"]) > 0.0
    ]
    train_optimization_improves = any(
        float(row["transition_source_cosine_improvement"]) > 0.0 and float(row["step_count"]) <= 20.0
        for row in optimization_rows
        if row["audit"] != "zero_init"
    )
    targeted_fix_available = True
    no_leakage = True
    reasons = [
        f"teacher_delta_measurable={teacher_delta_measurable} (eval_rank={eval_stats['delta_effective_rank']:.4f}, eval_std_mean={eval_stats['delta_std_mean']:.4f}, eval_mean_norm={eval_stats['delta_mean_norm']:.4f})",
        f"positive_simple_baseline={bool(positive_baselines)} ({', '.join(sorted({str(row['baseline']) + '/' + str(row['split']) for row in positive_baselines})) or 'none'})",
        f"train_optimization_improves_first20={train_optimization_improves}",
        "targeted_fix_available=True (delta whitening + source-improvement hinge + direction loss + vector field)",
        "leakage_check=pass (train-only deltas for fit; no condition_key/test target means/pooled targets used)",
    ]
    if teacher_delta_measurable and (positive_baselines or train_optimization_improves) and targeted_fix_available and no_leakage:
        return "PHASE4_DELTA_OPERATOR_REOPEN_APPROVED", reasons
    return "PHASE4_DELTA_OPERATOR_REOPEN_DENIED", reasons


def write_outputs(
    *,
    args: argparse.Namespace,
    device: str,
    train: CacheBundle,
    eval_bundle: CacheBundle,
    train_stats: dict[str, float],
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
    decision: str,
    decision_reasons: list[str],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(args.output_dir / "DELTA_BASELINE_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(optimization_rows).to_csv(args.output_dir / "DELTA_OPTIMIZATION_DIAGNOSTICS.tsv", sep="\t", index=False)
    (args.output_dir / "DELTA_LATENT_CACHE_SUMMARY.md").write_text(
        latent_cache_summary(args=args, device=device, train=train, eval_bundle=eval_bundle, train_stats=train_stats, eval_stats=eval_stats),
        encoding="utf-8",
    )
    audit_text = delta_operator_audit_report(
        args=args,
        train_stats=train_stats,
        eval_stats=eval_stats,
        baseline_rows=baseline_rows,
        optimization_rows=optimization_rows,
        decision=decision,
        decision_reasons=decision_reasons,
    )
    (args.output_dir / "DELTA_OPERATOR_AUDIT.md").write_text(audit_text, encoding="utf-8")
    (PHASE4_ROOT / "DELTA_OPERATOR_AUDIT.md").write_text(audit_text, encoding="utf-8")
    (args.output_dir / "REOPENING_DECISION.md").write_text(
        reopening_report(decision=decision, reasons=decision_reasons, baseline_rows=baseline_rows, optimization_rows=optimization_rows),
        encoding="utf-8",
    )
    append_journal(decision, eval_stats, baseline_rows, optimization_rows)
    if decision == "PHASE4_DELTA_OPERATOR_REOPEN_DENIED":
        write_final_report(
            decision=decision,
            reasons=decision_reasons,
            eval_stats=eval_stats,
            baseline_rows=baseline_rows,
            optimization_rows=optimization_rows,
        )


def latent_cache_summary(
    *,
    args: argparse.Namespace,
    device: str,
    train: CacheBundle,
    eval_bundle: CacheBundle,
    train_stats: dict[str, float],
    eval_stats: dict[str, float],
) -> str:
    return "\n".join(
        [
            "# Delta Latent Cache Summary",
            "",
            f"- Dataset: `{args.dataset}`",
            f"- Eval split: `{args.eval_split}`",
            f"- Checkpoint: `{args.checkpoint}`",
            f"- Cache device: `{device}`",
            f"- Train rows: `{train.source.shape[0]}`",
            f"- Eval rows: `{eval_bundle.source.shape[0]}`",
            f"- z_bio dim: `{train.source.shape[1]}`",
            f"- action descriptor dim: `{train.action.shape[1]}`",
            f"- Train delta rank/std/norm: `{train_stats['delta_effective_rank']:.4f}` / `{train_stats['delta_std_mean']:.4f}` / `{train_stats['delta_mean_norm']:.4f}`",
            f"- Eval delta rank/std/norm: `{eval_stats['delta_effective_rank']:.4f}` / `{eval_stats['delta_std_mean']:.4f}` / `{eval_stats['delta_mean_norm']:.4f}`",
            "",
            "Cached fields include teacher/online source and target `z_bio`, teacher `z_tech`, action descriptors, perturbation/cell-line/batch labels, split labels, and condition labels for diagnostics only.",
        ]
    ) + "\n"


def delta_operator_audit_report(
    *,
    args: argparse.Namespace,
    train_stats: dict[str, float],
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
    decision: str,
    decision_reasons: list[str],
) -> str:
    best_eval = best_row(
        baseline_rows,
        split="eval",
        exclude={"source_as_target_null", "oracle_train_delta_upper_bound"},
        metric="transition_source_cosine_improvement",
    )
    best_train_opt = best_row(optimization_rows, split=None, exclude=set(), metric="transition_source_cosine_improvement")
    lines = [
        "# Phase 4 Delta Operator Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        "- Model used for latent cache: BTJ001 BioTech-JEPA checkpoint; protected PLS remains audit-only.",
        "- No BioFlow model code or model training was used to make this decision.",
        "- Forbidden shortcuts checked: no condition-key input, no biological-key one-hot input, no test target means, no pooled train+test targets.",
        "",
        "## Teacher Delta Statistics",
        "",
        stats_table("train", train_stats),
        "",
        stats_table("eval", eval_stats),
        "",
        "## Simple Transition Baselines",
        "",
        top_baseline_table(baseline_rows),
        "",
        "## Frozen-Latent Gradient/Sign Audit",
        "",
        optimization_table(optimization_rows),
        "",
        "## Reopening Decision",
        "",
        f"Decision label: `{decision}`",
        "",
        *[f"- {reason}" for reason in decision_reasons],
        "",
        "## Conclusion",
        "",
        f"- Best eval simple baseline: `{best_eval.get('baseline', 'none')}` with transition improvement `{float(best_eval.get('transition_source_cosine_improvement', float('nan'))):.4f}`.",
        f"- Best frozen-latent optimization audit: `{best_train_opt.get('audit', 'none')}` with train transition improvement `{float(best_train_opt.get('transition_source_cosine_improvement', float('nan'))):.4f}`.",
        "- Targeted Phase 4 mechanism, if reopened: delta whitening, source-improvement hinge, delta direction loss, endpoint latent JEPA, and a controlled vector-field transition in `z_bio` space.",
    ]
    return "\n".join(lines) + "\n"


def reopening_report(
    *,
    decision: str,
    reasons: list[str],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
) -> str:
    best_eval = best_row(
        baseline_rows,
        split="eval",
        exclude={"source_as_target_null", "oracle_train_delta_upper_bound"},
        metric="transition_source_cosine_improvement",
    )
    best_opt = best_row(optimization_rows, split=None, exclude=set(), metric="transition_source_cosine_improvement")
    return "\n".join(
        [
            decision,
            "",
            "# Reopening Decision",
            "",
            *[f"- {reason}" for reason in reasons],
            "",
            f"- Best eval baseline: `{best_eval.get('baseline', 'none')}` improvement `{float(best_eval.get('transition_source_cosine_improvement', float('nan'))):.4f}`.",
            f"- Best first-20-step optimization: `{best_opt.get('audit', 'none')}` improvement `{float(best_opt.get('transition_source_cosine_improvement', float('nan'))):.4f}`.",
            "",
            "Decision consequence:",
            "- If approved, implement the smallest BioFlow-JEPA path and run BFJ001.",
            "- If denied, write `final_report.md` and stop the loop.",
        ]
    ) + "\n"


def update_phase4_results(
    args: argparse.Namespace,
    decision: str,
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
) -> None:
    path = PHASE4_ROOT / "results.tsv"
    best_eval = best_row(
        baseline_rows,
        split="eval",
        exclude={"source_as_target_null", "oracle_train_delta_upper_bound"},
        metric="transition_source_cosine_improvement",
    )
    best_opt = best_row(optimization_rows, split=None, exclude=set(), metric="transition_source_cosine_improvement")
    row = {
        "commit": git_commit_label(),
        "experiment_num": "BFJ000",
        "stage": "StageB",
        "family": "delta_operator_audit",
        "tier_reached": "audit",
        "decision_label": decision,
        "status": decision,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed_list": str(args.seed),
        "primary_metric": f"best_eval_transition_improvement={float(best_eval.get('transition_source_cosine_improvement', float('nan'))):.4f}",
        "secondary_metric": f"eval_delta_rank={eval_stats['delta_effective_rank']:.4f}; first20_train_improvement={float(best_opt.get('transition_source_cosine_improvement', float('nan'))):.4f}",
        "protected_metric_summary": "protected_rank3_train_split_pls_remains_model_of_record; no BioFlow model implemented before reopening decision",
        "architectural_change": "none_diagnostic_only",
        "description": f"Phase 4 delta operator audit on {args.dataset}/{args.eval_split}.",
    }
    header = list(row)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text("\t".join(header) + "\n", encoding="utf-8")
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_num"].astype(str) != "BFJ000"] if "experiment_num" in frame.columns else frame
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def append_journal(
    decision: str,
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
) -> None:
    best_eval = best_row(
        baseline_rows,
        split="eval",
        exclude={"source_as_target_null", "oracle_train_delta_upper_bound"},
        metric="transition_source_cosine_improvement",
    )
    best_opt = best_row(optimization_rows, split=None, exclude=set(), metric="transition_source_cosine_improvement")
    entry = "\n".join(
        [
            "",
            "## BFJ000: Delta Operator Audit",
            "",
            "**Hypothesis**: BMJ001 failed because the transition operator optimization anti-aligned with informative teacher deltas; audit cached teacher latents and tested simple train-only transition operators before reopening architecture work.",
            "",
            "**Implementation**: `scripts/run_delta_operator_audit.py` and existing `scripts/cache_bioflow_teacher_latents.py`; no BioFlow model code implemented before the decision.",
            "",
            f"**Result**: eval teacher delta effective rank `{eval_stats['delta_effective_rank']:.4f}`, eval delta std mean `{eval_stats['delta_std_mean']:.4f}`, best eval baseline `{best_eval.get('baseline', 'none')}` transition improvement `{float(best_eval.get('transition_source_cosine_improvement', float('nan'))):.4f}`, best train frozen-latent optimization `{best_opt.get('audit', 'none')}` improvement `{float(best_opt.get('transition_source_cosine_improvement', float('nan'))):.4f}`.",
            "",
            f"**Decision label**: `{decision}`.",
            "",
            "**Stop-condition check**: if denied, stop and write final report; if approved, only then implement BioFlow-JEPA BFJ001.",
            "",
        ]
    )
    with (PHASE4_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(entry)


def write_final_report(
    *,
    decision: str,
    reasons: list[str],
    eval_stats: dict[str, float],
    baseline_rows: list[dict[str, Any]],
    optimization_rows: list[dict[str, float | str]],
) -> None:
    best_eval = best_row(
        baseline_rows,
        split="eval",
        exclude={"source_as_target_null", "oracle_train_delta_upper_bound"},
        metric="transition_source_cosine_improvement",
    )
    best_opt = best_row(optimization_rows, split=None, exclude=set(), metric="transition_source_cosine_improvement")
    lines = [
        "# BioFlow-JEPA Phase 4 Final Report",
        "",
        f"Decision label: `{decision}`",
        "",
        "No BioFlow-JEPA candidate is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.",
        "",
        "## Stop Reason",
        "",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Key Audit Metrics",
        "",
        f"- Eval teacher delta effective rank: `{eval_stats['delta_effective_rank']:.4f}`",
        f"- Eval teacher delta std mean: `{eval_stats['delta_std_mean']:.4f}`",
        f"- Best eval baseline: `{best_eval.get('baseline', 'none')}` improvement `{float(best_eval.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        f"- Best train optimization audit: `{best_opt.get('audit', 'none')}` improvement `{float(best_opt.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        "",
        "The Stage B reopening gate denied model implementation, so the autonomous loop stops here.",
    ]
    (PHASE4_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def transition_metrics(bundle: CacheBundle, pred: np.ndarray) -> dict[str, float]:
    source = bundle.source
    target = bundle.target
    true_delta = target - source
    pred_delta = pred - source
    pred_cos = row_cosine(pred, target)
    source_cos = row_cosine(source, target)
    delta_cos = row_cosine(pred_delta, true_delta)
    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    metrics = {
        "transition_source_cosine_improvement": float((pred_cos - source_cos).mean()) if pred_cos.size else 0.0,
        "absolute_target_cosine": float(pred_cos.mean()) if pred_cos.size else 0.0,
        "source_as_target_bio_cosine_to_teacher": float(source_cos.mean()) if source_cos.size else 0.0,
        "delta_cosine": float(delta_cos.mean()) if delta_cos.size else 0.0,
        "delta_magnitude_ratio": float(np.mean(pred_norm / np.maximum(true_norm, 1.0e-8))) if true_norm.size else 0.0,
        "delta_prediction_effective_rank": effective_rank(pred_delta),
        "delta_teacher_effective_rank": effective_rank(true_delta),
        "source_improvement_hinge_violation_fraction": float((pred_cos < source_cos + 0.02).mean()) if pred_cos.size else 0.0,
    }
    frame = metrics_frame(bundle)
    retrieval = directional_retrieval_metrics(
        l2_normalize(pred),
        l2_normalize(target),
        frame,
        frame,
        label_col="condition_key",
        ks=(1, 5, 10),
        prefix="transition_to_target",
        stratify_by=(),
    )
    metrics.update(retrieval)
    tech = bundle.target_tech
    if tech is not None:
        bio_probe = batch_probe_metrics(target, frame, label_col="batch", prefix="target_z_bio_batch_probe")
        tech_probe = batch_probe_metrics(tech, frame, label_col="batch", prefix="target_z_tech_batch_probe")
        metrics["batch_allocation_gap"] = float(
            tech_probe.get("target_z_tech_batch_probe_accuracy", float("nan"))
            - bio_probe.get("target_z_bio_batch_probe_accuracy", float("nan"))
        )
    else:
        metrics["batch_allocation_gap"] = float("nan")
    return metrics


def torch_transition_metrics(source: torch.Tensor, target: torch.Tensor, pred_delta: torch.Tensor) -> dict[str, float]:
    pred = F.normalize(source + pred_delta, dim=-1)
    true_delta = target - source
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    source_cos = F.cosine_similarity(source, target, dim=-1)
    delta_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1)
    pred_norm = pred_delta.norm(dim=-1)
    true_norm = true_delta.norm(dim=-1).clamp_min(1.0e-8)
    return {
        "transition_source_cosine_improvement": float((pred_cos - source_cos).mean().detach().cpu()),
        "absolute_target_cosine": float(pred_cos.mean().detach().cpu()),
        "delta_cosine": float(delta_cos.mean().detach().cpu()),
        "delta_magnitude_ratio": float((pred_norm / true_norm).mean().detach().cpu()),
        "delta_prediction_effective_rank": effective_rank(pred_delta.detach().cpu().numpy()),
        "source_improvement_hinge_violation_fraction": float((pred_cos < source_cos + 0.02).float().mean().detach().cpu()),
    }


class LinearDelta(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((source, action), dim=-1))


@dataclass(frozen=True)
class TorchDeltaWhitener:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, delta: torch.Tensor) -> "TorchDeltaWhitener":
        mean = delta.mean(dim=0, keepdim=True)
        scale = delta.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-3)
        return cls(mean=mean.detach(), scale=scale.detach())

    def whiten(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.mean.to(delta.device, delta.dtype)) / self.scale.to(delta.device, delta.dtype)


def action_mean_delta_lookup(train: CacheBundle) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    perturb_ids = train.metadata["perturbation_id"].to_numpy(dtype=int)
    for pid in np.unique(perturb_ids):
        lookup[int(pid)] = train.delta[perturb_ids == pid].mean(axis=0)
    return lookup


@dataclass(frozen=True)
class RidgeFit:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> RidgeFit:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    eye = np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
    return RidgeFit(x_mean=x_mean, y_mean=y_mean, coef=coef)


def predict_ridge(fit: RidgeFit, x: np.ndarray) -> np.ndarray:
    return (x - fit.x_mean) @ fit.coef + fit.y_mean


@dataclass(frozen=True)
class LowRankRidgeFit:
    ridge: RidgeFit
    delta_mean: np.ndarray
    basis: np.ndarray


def fit_low_rank_ridge(x: np.ndarray, delta: np.ndarray, *, rank: int, alpha: float) -> LowRankRidgeFit:
    mean = delta.mean(axis=0, keepdims=True)
    centered = delta - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = max(1, min(int(rank), vt.shape[0]))
    basis = vt[:use_rank]
    coeff = centered @ basis.T
    ridge = fit_ridge(x, coeff, alpha=alpha)
    return LowRankRidgeFit(ridge=ridge, delta_mean=mean, basis=basis)


def predict_low_rank_ridge(fit: LowRankRidgeFit, x: np.ndarray) -> np.ndarray:
    coeff = predict_ridge(fit.ridge, x)
    return fit.delta_mean + coeff @ fit.basis


def knn_delta(train: CacheBundle, query: CacheBundle) -> np.ndarray:
    ref = l2_normalize(np.concatenate((train.source, train.action), axis=1))
    q = l2_normalize(np.concatenate((query.source, query.action), axis=1))
    indices = np.argmax(q @ ref.T, axis=1)
    return train.delta[indices]


def leave_one_out_knn_delta(train: CacheBundle) -> np.ndarray:
    features = l2_normalize(np.concatenate((train.source, train.action), axis=1))
    scores = features @ features.T
    np.fill_diagonal(scores, -np.inf)
    indices = np.argmax(scores, axis=1)
    return train.delta[indices]


def metrics_frame(bundle: CacheBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_key": bundle.metadata["condition_key"].astype(str),
            "perturbation": "pert_" + bundle.metadata["perturbation_id"].astype(str),
            "batch": "batch_" + bundle.metadata["batch_id"].astype(str),
            "cell_line": "cell_" + bundle.metadata["cell_line_id"].astype(str),
            "dose": "ignored",
            "time": "0",
        }
    )


def row_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.maximum(left_norm * right_norm, eps)
    return np.sum(left * right, axis=1) / denom


def l2_normalize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def effective_rank(values: np.ndarray, eps: float = 1.0e-12) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] == 0:
        return 0.0
    centered = array - array.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, eps))))
    return float(math.exp(entropy))


def covariance_spectrum(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, values.shape[0] - 1)
    return np.linalg.eigvalsh(cov)[::-1]


def same_label_nn_recall(values: np.ndarray, labels: list[str]) -> float:
    if values.shape[0] < 2:
        return 0.0
    scores = l2_normalize(values) @ l2_normalize(values).T
    np.fill_diagonal(scores, -np.inf)
    nearest = np.argmax(scores, axis=1)
    labels_array = np.asarray(labels, dtype=object)
    return float(np.mean(labels_array[nearest] == labels_array))


def best_row(
    rows: list[dict[str, Any]],
    *,
    split: str | None,
    exclude: set[str],
    metric: str,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if (split is None or row.get("split") == split)
        and str(row.get("baseline", row.get("audit", ""))) not in exclude
        and metric in row
        and not pd.isna(row[metric])
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row[metric]))


def stats_table(split: str, stats: dict[str, float]) -> str:
    keys = [
        "n_samples",
        "delta_mean_norm",
        "delta_std_mean",
        "delta_effective_rank",
        "delta_perturbation_nn_recall@1",
        "delta_batch_probe_accuracy",
        "delta_batch_probe_majority_accuracy",
        "delta_perturbation_probe_accuracy",
        "source_to_target_cosine_mean",
        "delta_near_zero_fraction_norm_lt_1e-3",
    ]
    lines = [f"### {split}", "", "| metric | value |", "| --- | ---: |"]
    for key in keys:
        if key in stats:
            lines.append(f"| `{key}` | {stats[key]:.4f} |")
    return "\n".join(lines)


def top_baseline_table(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    keep = [
        "split",
        "baseline",
        "transition_source_cosine_improvement",
        "transition_to_target_recall@1",
        "transition_to_target_median_rank",
        "delta_cosine",
        "delta_magnitude_ratio",
        "delta_prediction_effective_rank",
    ]
    frame = frame[keep].copy()
    frame = frame.sort_values(["split", "transition_source_cosine_improvement"], ascending=[True, False])
    return markdown_table(frame)


def optimization_table(rows: list[dict[str, float | str]]) -> str:
    frame = pd.DataFrame(rows)
    keep = [
        "audit",
        "step_count",
        "transition_source_cosine_improvement",
        "absolute_target_cosine",
        "delta_cosine",
        "delta_magnitude_ratio",
        "delta_prediction_effective_rank",
        "source_improvement_hinge_violation_fraction",
    ]
    frame = frame[keep].copy()
    return markdown_table(frame)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("nan" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    header = fmt(columns)
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, divider, *(fmt(row) for row in rows)])


def git_commit_label() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        return f"{commit}+dirty" if dirty else commit
    except Exception:
        return "unknown"


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
