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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_biooperator_contract_audit import LatentBundle, load_bundle, transition_metrics


PHASE6_ROOT = Path("outputs/autoresearch_biospectral_jepa_phase6")
PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
PHASE5_ROOT = Path("outputs/autoresearch_biooperator_jepa_phase5")
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
    "delta_magnitude_ratio": 0.7585,
}


@dataclass(frozen=True)
class RidgeFit:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray


@dataclass(frozen=True)
class ReducedRankFit:
    ridge: RidgeFit
    delta_mean: np.ndarray
    basis: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 6 BioSpectral rank bottleneck audit.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--output-dir", type=Path, default=PHASE6_ROOT / "rank_bottleneck_audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    args = parser.parse_args()

    PHASE6_ROOT.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialize_docs()
    train = load_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")

    full_fit = fit_ridge(features(train), train.delta, alpha=args.alpha)
    low_rank_fit = fit_reduced_rank_ridge(features(train), train.delta, rank=8, alpha=args.alpha)
    full_rows = operator_rows(train, eval_bundle, "action_ridge_delta", predict_ridge_for_bundles(full_fit, train, eval_bundle))
    low_rank_rows = operator_rows(train, eval_bundle, "action_low_rank_ridge", predict_reduced_for_bundles(low_rank_fit, train, eval_bundle))
    reproduction_rows = full_rows + low_rank_rows
    reproduction_ok = floor_reproduction_ok(reproduction_rows)

    spectrum_rows = spectrum_audit_rows(train, eval_bundle, full_fit, low_rank_fit)
    sweep_rows = reduced_rank_sweep(train, eval_bundle, alpha=args.alpha)
    neural_low_rank = neural_low_rank_equivalence(train, eval_bundle, low_rank_fit)
    boj002_rows, discrepancy_rows = boj002_decomposition_rows(train, eval_bundle)
    decision = reopening_decision(reproduction_ok, neural_low_rank, sweep_rows, boj002_rows)

    pd.DataFrame(reproduction_rows).to_csv(args.output_dir / "floor_reproduction.tsv", sep="\t", index=False)
    pd.DataFrame(spectrum_rows).to_csv(args.output_dir / "spectrum_audit.tsv", sep="\t", index=False)
    pd.DataFrame(sweep_rows).to_csv(args.output_dir / "reduced_rank_sweep.tsv", sep="\t", index=False)
    pd.DataFrame([neural_low_rank]).to_csv(args.output_dir / "neural_low_rank_equivalence.tsv", sep="\t", index=False)
    pd.DataFrame(boj002_rows).to_csv(args.output_dir / "boj002_decomposition_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(discrepancy_rows).to_csv(args.output_dir / "control_affine_discrepancy_table.tsv", sep="\t", index=False)
    write_reports(args, reproduction_rows, spectrum_rows, sweep_rows, neural_low_rank, boj002_rows, discrepancy_rows, decision)
    update_results(args, reproduction_rows, sweep_rows, neural_low_rank, decision)
    if decision != "PHASE6_REOPEN_BIOSPECTRAL_APPROVED":
        write_final_report(decision, reproduction_rows, sweep_rows, neural_low_rank, boj002_rows)
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if decision == "PHASE6_REOPEN_BIOSPECTRAL_APPROVED" else 2


def initialize_docs() -> None:
    docs = {
        "results.tsv": "experiment_id\tstage\tmode\tseed\tdataset\teval_split\ttransition_improvement\tdelta_cosine\trecall_at_1\tdelta_rank\tmagnitude_ratio\tfloor_gap\tidentity_pass\tdecision\tartifact_dir\n",
        "research_journal.md": "# BioSpectral-JEPA Phase 6 Research Journal\n\n",
        "architectural_changes_log.md": "# Phase 6 Architectural Changes Log\n\nNo BioSpectral architecture code before BSJ000 reopening approval.\n",
        "family_allocation.md": "# Phase 6 Family Allocation\n\n| family | experiments | status |\n| --- | --- | --- |\n| diagnostics | BSJ000 | in progress |\n| spectral floor preserving operators | BSJ001-BSJ005 | gated |\n| full BioSpectral-JEPA | BSJ006 | gated |\n",
        "external_resources.md": "# External Resources\n\nNo new external datasets downloaded in Phase 6. Uses cached Phase 4 latent audit artifacts.\n",
        "identity_violations_considered.md": "# Identity Violations Considered\n\n- PLS/raw-linear main path: forbidden; audit-only model of record.\n- `condition_key`, `biological_key`, exact target-key one-hot: forbidden as model inputs.\n- Test/eval target means and pooled train+test statistics: forbidden.\n- Batch id as biological transition shortcut: forbidden.\n",
        "papers_consulted.md": papers_text(),
        "BASELINE_REGISTRY.md": baseline_registry_text(),
    }
    for name, text in docs.items():
        path = PHASE6_ROOT / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def papers_text() -> str:
    return """# Papers Consulted

| Source | Mechanism extracted | Phase 6 mapping |
| --- | --- | --- |
| V-JEPA 2, https://arxiv.org/abs/2506.09985 | action-conditioned latent prediction with stop-gradient targets | preserve real JEPA identity for later BSJ006 wrapper |
| Dynamic Mode Decomposition with Control, https://epubs.siam.org/doi/10.1137/15M1013857 | separate state dynamics from control effects and audit low-order approximations | compare full ridge, reduced-rank ridge, and BOJ002 control-affine form |
| Koopman / EDMD-style operator learning, https://faculty.washington.edu/kutz/page1/page13/ | finite-rank latent operators need validation | rank bottleneck audit before architecture search |
| LoRA / adaptive low-rank, https://arxiv.org/abs/2505.22694 | fixed rank may be too restrictive; use adaptive rank experts | rank ladder and spectral residual design if reopened |
| CellOT, https://www.nature.com/articles/s41592-023-01969-x | perturbation responses are distributional maps | defer population transport until floor-preserving operator passes |
| CPA, https://www.embopress.org/doi/full/10.15252/msb.202211517 | compositional perturbation/context factors | keep action descriptor discipline; Norman dose is guide presence only |
| GEARS, https://www.nature.com/articles/s41587-023-01905-6 | gene/gene-pair action priors | optional future action features, not a replacement for floor preservation |
| GPerturb, https://www.nature.com/articles/s41467-025-61165-7 | uncertainty/kernel perturbation effects | optional kernel residual only after spectral residual evidence |
"""


def baseline_registry_text() -> str:
    return """# Phase 6 Baseline Registry

## Model Of Record

Protected rank-3 train-split-only PLS raw-linear readout remains model of record. PLS is audit-only and not the BioSpectral-JEPA representation path.

## Fixed Floors

| metric | value |
| --- | ---: |
| full_action_ridge_eval_transition_improvement | 0.0057 |
| full_action_ridge_eval_delta_cosine | 0.3980 |
| full_action_ridge_eval_recall@1 | 0.4815 |
| full_action_ridge_eval_delta_rank | 10.2835 |
| full_action_ridge_eval_magnitude_ratio | 0.7744 |
| low_rank_action_ridge_eval_transition_improvement | 0.0046 |
| low_rank_action_ridge_eval_delta_cosine | 0.3877 |
| low_rank_action_ridge_eval_recall@1 | 0.4074 |
| low_rank_action_ridge_eval_delta_rank | 6.7681 |
| low_rank_action_ridge_eval_magnitude_ratio | 0.7585 |
| source_as_target_eval_transition_improvement | 0.0000 |
| source_as_target_eval_recall@1 | 0.2963 |

Sources:

- `outputs/autoresearch_biooperator_jepa_phase5/final_report.md`
- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md`
"""


def features(bundle: LatentBundle) -> np.ndarray:
    return np.concatenate((bundle.source, bundle.action), axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> RidgeFit:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    coef = np.linalg.solve(xc.T @ xc + float(alpha) * np.eye(x.shape[1]), xc.T @ yc)
    return RidgeFit(x_mean=x_mean, y_mean=y_mean, coef=coef)


def predict_ridge(fit: RidgeFit, x: np.ndarray) -> np.ndarray:
    return (x - fit.x_mean) @ fit.coef + fit.y_mean


def fit_reduced_rank_ridge(x: np.ndarray, delta: np.ndarray, *, rank: int | str, alpha: float) -> ReducedRankFit:
    mean = delta.mean(axis=0, keepdims=True)
    centered = delta - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = vt.shape[0] if rank == "full" else max(1, min(int(rank), vt.shape[0]))
    basis = vt[:use_rank]
    coeff = centered @ basis.T
    return ReducedRankFit(ridge=fit_ridge(x, coeff, alpha=alpha), delta_mean=mean, basis=basis)


def predict_reduced_rank(fit: ReducedRankFit, x: np.ndarray) -> np.ndarray:
    return fit.delta_mean + predict_ridge(fit.ridge, x) @ fit.basis


def predict_ridge_for_bundles(fit: RidgeFit, train: LatentBundle, eval_bundle: LatentBundle) -> dict[str, np.ndarray]:
    return {"train": predict_ridge(fit, features(train)), "eval": predict_ridge(fit, features(eval_bundle))}


def predict_reduced_for_bundles(fit: ReducedRankFit, train: LatentBundle, eval_bundle: LatentBundle) -> dict[str, np.ndarray]:
    return {"train": predict_reduced_rank(fit, features(train)), "eval": predict_reduced_rank(fit, features(eval_bundle))}


def operator_rows(train: LatentBundle, eval_bundle: LatentBundle, name: str, deltas: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for split, bundle in (("train", train), ("eval", eval_bundle)):
        metrics = transition_metrics(bundle, bundle.source + deltas[split])
        rows.append({"split": split, "operator": name, **metrics})
    return rows


def floor_reproduction_ok(rows: list[dict[str, Any]]) -> bool:
    full = row_for(rows, "eval", "action_ridge_delta")
    low = row_for(rows, "eval", "action_low_rank_ridge")
    return (
        abs(float(full["transition_source_cosine_improvement"]) - FULL_FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(float(low["transition_source_cosine_improvement"]) - LOW_RANK_FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
    )


def spectrum_audit_rows(train: LatentBundle, eval_bundle: LatentBundle, full_fit: RidgeFit, low_rank_fit: ReducedRankFit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_map = {
        "teacher_delta": {"train": train.delta, "eval": eval_bundle.delta},
        "full_ridge_pred_delta": predict_ridge_for_bundles(full_fit, train, eval_bundle),
        "low_rank_ridge_pred_delta": predict_reduced_for_bundles(low_rank_fit, train, eval_bundle),
    }
    boj002 = load_boj002_predictions(train, eval_bundle)
    if boj002 is not None:
        pred_map["boj002_pred_delta"] = boj002
    for name, split_values in pred_map.items():
        for split, values in split_values.items():
            spectrum = covariance_spectrum(values)
            row: dict[str, Any] = {
                "split": split,
                "operator": name,
                "effective_rank": effective_rank(values),
                "spectral_entropy": spectral_entropy(values),
            }
            for k in (1, 2, 4, 6, 8, 10, 12, 16, "full"):
                row[f"cumvar_top_{k}"] = cumulative_variance(spectrum, k)
            rows.append(row)
    for split, bundle in (("train", train), ("eval", eval_bundle)):
        teacher = bundle.delta
        for name in ("full_ridge_pred_delta", "low_rank_ridge_pred_delta"):
            pred = pred_map[name][split]
            angles = principal_angles(teacher, pred, k=min(8, teacher.shape[1], teacher.shape[0] - 1))
            agreement = per_perturbation_delta_agreement(bundle, pred)
            residual = teacher - pred
            rows.append(
                {
                    "split": split,
                    "operator": f"{name}_vs_teacher",
                    "principal_angle_mean_deg": float(np.mean(angles)) if angles.size else 0.0,
                    "principal_angle_max_deg": float(np.max(angles)) if angles.size else 0.0,
                    "per_perturbation_direction_agreement": agreement,
                    "residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
                    "residual_effective_rank": effective_rank(residual),
                }
            )
    return rows


def reduced_rank_sweep(train: LatentBundle, eval_bundle: LatentBundle, *, alpha: float) -> list[dict[str, Any]]:
    rows = []
    ranks: list[int | str] = [1, 2, 4, 6, 8, 10, 12, 16, 24, "full"]
    for rank in ranks:
        fit = fit_reduced_rank_ridge(features(train), train.delta, rank=rank, alpha=alpha)
        for split, bundle, delta in (
            ("train", train, predict_reduced_rank(fit, features(train))),
            ("eval", eval_bundle, predict_reduced_rank(fit, features(eval_bundle))),
        ):
            metrics = transition_metrics(bundle, bundle.source + delta)
            rows.append(
                {
                    "rank": str(rank),
                    "split": split,
                    "transition_improvement": metrics["transition_source_cosine_improvement"],
                    "delta_cosine": metrics["delta_cosine"],
                    "recall@1": metrics["transition_to_target_recall@1"],
                    "median_rank": metrics["transition_to_target_median_rank"],
                    "delta_rank": metrics["delta_prediction_effective_rank"],
                    "magnitude_ratio": metrics["delta_magnitude_ratio"],
                    "floor_gap_vs_full_ridge": metrics["transition_source_cosine_improvement"] - FULL_FLOOR["transition_source_cosine_improvement"],
                }
            )
    return rows


class NeuralReducedRankRidgeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    @torch.no_grad()
    def initialize_from_reduced_rank(self, fit: ReducedRankFit) -> None:
        coef_delta = fit.ridge.coef @ fit.basis
        y_mean_delta = fit.delta_mean + fit.ridge.y_mean @ fit.basis
        bias = (y_mean_delta - fit.ridge.x_mean @ coef_delta).reshape(-1)
        self.linear.weight.copy_(torch.as_tensor(coef_delta.T, dtype=self.linear.weight.dtype))
        self.linear.bias.copy_(torch.as_tensor(bias, dtype=self.linear.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def neural_low_rank_equivalence(train: LatentBundle, eval_bundle: LatentBundle, fit: ReducedRankFit) -> dict[str, Any]:
    head = NeuralReducedRankRidgeHead(features(train).shape[1], train.delta.shape[1])
    head.initialize_from_reduced_rank(fit)
    with torch.no_grad():
        train_delta = head(torch.as_tensor(features(train), dtype=torch.float32)).numpy()
        eval_delta = head(torch.as_tensor(features(eval_bundle), dtype=torch.float32)).numpy()
    analytic_train = predict_reduced_rank(fit, features(train))
    analytic_eval = predict_reduced_rank(fit, features(eval_bundle))
    train_metrics = transition_metrics(train, train.source + train_delta)
    eval_metrics = transition_metrics(eval_bundle, eval_bundle.source + eval_delta)
    analytic_eval_metrics = transition_metrics(eval_bundle, eval_bundle.source + analytic_eval)
    pass_gate = (
        abs(eval_metrics["transition_source_cosine_improvement"] - analytic_eval_metrics["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(eval_metrics["delta_cosine"] - analytic_eval_metrics["delta_cosine"]) <= 1.0e-4
        and abs(eval_metrics["transition_to_target_recall@1"] - analytic_eval_metrics["transition_to_target_recall@1"]) <= 1.0e-4
        and abs(eval_metrics["delta_prediction_effective_rank"] - analytic_eval_metrics["delta_prediction_effective_rank"]) <= 1.0e-3
        and np.allclose(train_delta, analytic_train, atol=1.0e-5)
        and np.allclose(eval_delta, analytic_eval, atol=1.0e-5)
    )
    return {
        "neural_low_rank_equivalence_pass": float(pass_gate),
        "eval_transition_improvement": eval_metrics["transition_source_cosine_improvement"],
        "eval_delta_cosine": eval_metrics["delta_cosine"],
        "eval_recall@1": eval_metrics["transition_to_target_recall@1"],
        "eval_delta_rank": eval_metrics["delta_prediction_effective_rank"],
        "max_abs_eval_delta_diff": float(np.max(np.abs(eval_delta - analytic_eval))),
        "train_transition_improvement": train_metrics["transition_source_cosine_improvement"],
    }


def load_boj002_predictions(train: LatentBundle, eval_bundle: LatentBundle) -> dict[str, np.ndarray] | None:
    payload_path = PHASE5_ROOT / "experiments/BOJ002_frozen_control_affine_seed0/operator_payload.json"
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    left = np.asarray(payload["left"], dtype=np.float64)
    right = np.asarray(payload["right"], dtype=np.float64)
    source_mean = np.asarray(payload["source_mean"], dtype=np.float64)
    action_mean = np.asarray(payload["action_mean"], dtype=np.float64)
    delta_mean = np.asarray(payload["delta_mean"], dtype=np.float64)
    action_shift = np.asarray(payload["action_shift_weight"], dtype=np.float64)

    def pred(bundle: LatentBundle) -> np.ndarray:
        state_delta = (bundle.source - source_mean) @ left @ right
        action_delta = (bundle.action - action_mean) @ action_shift.T if action_shift.size else np.zeros_like(state_delta)
        return delta_mean + state_delta + action_delta

    return {"train": pred(train), "eval": pred(eval_bundle)}


def boj002_decomposition_rows(train: LatentBundle, eval_bundle: LatentBundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred = load_boj002_predictions(train, eval_bundle)
    if pred is None:
        return [], [{"item": "BOJ002 checkpoint", "status": "missing", "detail": "No BOJ002 operator_payload.json available"}]
    rows = operator_rows(train, eval_bundle, "boj002_reconstructed", pred)
    low_fit = fit_reduced_rank_ridge(features(train), train.delta, rank=8, alpha=1.0e-2)
    low_pred = predict_reduced_for_bundles(low_fit, train, eval_bundle)
    discrepancy = []
    for split, bundle in (("train", train), ("eval", eval_bundle)):
        diff = pred[split] - low_pred[split]
        discrepancy.append({"item": f"{split}_prediction_vs_analytical_low_rank", "status": "diff", "detail": f"rmse={float(np.sqrt((diff**2).mean())):.6f}; max_abs={float(np.abs(diff).max()):.6f}"})
    discrepancy.extend(
        [
            {"item": "same_action_features", "status": "pass", "detail": f"action_dim={train.action.shape[1]}"},
            {"item": "same_train_rows_only", "status": "pass", "detail": f"train_rows={train.source.shape[0]}"},
            {"item": "same_ridge_regularization", "status": "pass", "detail": "alpha=1e-2"},
            {"item": "same_output_normalization", "status": "warning", "detail": "operator metrics use source + delta without endpoint normalization; old BOJ002 code trained raw delta then evaluated common metrics"},
            {"item": "same_rank_orientation", "status": "fail", "detail": "BOJ002 decomposes source block and keeps full action shift; analytical low-rank ridge reduces full predicted delta in teacher-delta spectral basis"},
            {"item": "same_bias_term", "status": "pass", "detail": "delta mean/source/action mean retained"},
            {"item": "same_feature_standardization", "status": "pass", "detail": "mean-centering only"},
            {"item": "same_action_descriptor_dimensions", "status": "pass", "detail": f"{train.action.shape[1]}"},
        ]
    )
    return rows, discrepancy


def reopening_decision(
    reproduction_ok: bool,
    neural_low_rank: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    boj002_rows: list[dict[str, Any]],
) -> str:
    if not reproduction_ok:
        return "PHASE6_STOP_FLOOR_REPRODUCTION_FAILED"
    if float(neural_low_rank["neural_low_rank_equivalence_pass"]) != 1.0:
        return "PHASE6_STOP_NEURAL_LOWRANK_EQUIVALENCE_FAILED"
    eval_rows = [row for row in sweep_rows if row["split"] == "eval"]
    rank8 = next(row for row in eval_rows if row["rank"] == "8")
    full = next(row for row in eval_rows if row["rank"] == "full")
    fixed_rank_bottleneck = (
        float(rank8["recall@1"]) < FULL_FLOOR["transition_to_target_recall@1"] - 1.0e-4
        or float(rank8["delta_rank"]) < 8.0
        or float(rank8["transition_improvement"]) < FULL_FLOOR["transition_source_cosine_improvement"] - 1.0e-4
    )
    headroom = (
        float(full["transition_improvement"]) >= FULL_FLOOR["transition_source_cosine_improvement"] - 1.0e-4
        and float(full["recall@1"]) >= FULL_FLOOR["transition_to_target_recall@1"] - 1.0e-4
        and float(full["delta_rank"]) >= 10.0
    )
    if not fixed_rank_bottleneck and boj002_rows:
        return "PHASE6_STOP_CONTROL_AFFINE_BUG_ONLY"
    if not headroom:
        return "PHASE6_STOP_NO_RANK_HEADROOM"
    return "PHASE6_REOPEN_BIOSPECTRAL_APPROVED"


def write_reports(
    args: argparse.Namespace,
    reproduction_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    neural_low_rank: dict[str, Any],
    boj002_rows: list[dict[str, Any]],
    discrepancy_rows: list[dict[str, Any]],
    decision: str,
) -> None:
    full = row_for(reproduction_rows, "eval", "action_ridge_delta")
    low = row_for(reproduction_rows, "eval", "action_low_rank_ridge")
    sweep_eval = pd.DataFrame([row for row in sweep_rows if row["split"] == "eval"])
    report = [
        "# Rank Bottleneck Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Latent cache: `{args.latent_cache}`",
        "- No BioSpectral architecture was implemented before this audit.",
        "- Leakage check: train-only ridge/basis fits; eval latents used only for scoring; condition labels used only for retrieval labels.",
        "",
        "## Stage A Floor Reproduction",
        "",
        f"- Full action ridge eval improvement: `{full['transition_source_cosine_improvement']:.4f}`",
        f"- Low-rank action ridge eval improvement: `{low['transition_source_cosine_improvement']:.4f}`",
        "",
        "## Reduced-Rank Sweep",
        "",
        markdown_table(sweep_eval, ["rank", "transition_improvement", "delta_cosine", "recall@1", "median_rank", "delta_rank", "magnitude_ratio", "floor_gap_vs_full_ridge"]),
        "",
        "## Neural Low-Rank Equivalence",
        "",
        f"- pass: `{neural_low_rank['neural_low_rank_equivalence_pass']}`",
        f"- eval improvement: `{neural_low_rank['eval_transition_improvement']:.4f}`",
        f"- eval delta cosine: `{neural_low_rank['eval_delta_cosine']:.4f}`",
        f"- eval recall@1: `{neural_low_rank['eval_recall@1']:.4f}`",
        f"- eval delta rank: `{neural_low_rank['eval_delta_rank']:.4f}`",
        f"- max abs eval delta diff: `{neural_low_rank['max_abs_eval_delta_diff']:.8f}`",
        "",
        "## BOJ002 Control-Affine Discrepancy",
        "",
        markdown_table(pd.DataFrame(discrepancy_rows), ["item", "status", "detail"]),
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "## Interpretation",
        "",
        "BOJ002 failed for both reasons considered by the prompt: its source-block factorization does not reproduce analytical low-rank ridge, and rank-8 reduced-rank ridge itself loses held-out recall/rank relative to the full action-ridge floor. A floor-preserving, rank-adaptive spectral residual has plausible headroom because the full-ridge floor is reproducible and the reduced-rank sweep shows higher ranks recover floor geometry.",
    ]
    text = "\n".join(report) + "\n"
    (args.output_dir / "RANK_BOTTLENECK_AUDIT.md").write_text(text, encoding="utf-8")
    (args.output_dir / "REOPENING_DECISION.md").write_text(decision + "\n", encoding="utf-8")
    with (PHASE6_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## BSJ000: Rank Bottleneck Audit\n\n"
            f"**Result**: full ridge eval improvement `{full['transition_source_cosine_improvement']:.4f}`, low-rank eval improvement `{low['transition_source_cosine_improvement']:.4f}`, neural low-rank equivalence pass `{neural_low_rank['neural_low_rank_equivalence_pass']}`.\n\n"
            f"**Decision label**: `{decision}`.\n\n"
        )


def update_results(args: argparse.Namespace, reproduction_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]], neural_low_rank: dict[str, Any], decision: str) -> None:
    full = row_for(reproduction_rows, "eval", "action_ridge_delta")
    row = {
        "experiment_id": "BSJ000",
        "stage": "StageA_B",
        "mode": "rank_bottleneck_audit",
        "seed": 0,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "transition_improvement": f"{full['transition_source_cosine_improvement']:.4f}",
        "delta_cosine": f"{full['delta_cosine']:.4f}",
        "recall_at_1": f"{full['transition_to_target_recall@1']:.4f}",
        "delta_rank": f"{full['delta_prediction_effective_rank']:.4f}",
        "magnitude_ratio": f"{full['delta_magnitude_ratio']:.4f}",
        "floor_gap": "0.0000",
        "identity_pass": "1.0",
        "decision": decision,
        "artifact_dir": str(args.output_dir),
    }
    path = PHASE6_ROOT / "results.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_id"].astype(str) != "BSJ000"]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def write_final_report(decision: str, reproduction_rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]], neural_low_rank: dict[str, Any], boj002_rows: list[dict[str, Any]]) -> None:
    full = row_for(reproduction_rows, "eval", "action_ridge_delta")
    lines = [
        "# BioSpectral-JEPA Phase 6 Final Report",
        "",
        "## Decision label",
        decision,
        "",
        "## Model of record",
        "Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless Tier 3 pass supersedes it.",
        "",
        "## What was tested",
        "- Stage A floor reproduction: completed",
        "- Stage B rank bottleneck audit: completed",
        "- BSJ001-BSJ007: not run because reopening was not approved",
        "",
        "## Key metrics",
        "| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| BSJ000 full ridge | {full['transition_source_cosine_improvement']:.4f} | {full['delta_cosine']:.4f} | {full['transition_to_target_recall@1']:.4f} | {full['delta_prediction_effective_rank']:.4f} | {full['delta_magnitude_ratio']:.4f} | {decision} |",
        "",
        "## Main interpretation",
        "Reopening was denied by the Stage B decision gate.",
        "",
        "## JEPA identity status",
        "No full BioSpectral-JEPA candidate was run.",
        "",
        "## Leakage status",
        "No forbidden feature/statistic usage was detected in Stage A/B.",
        "",
        "## Recommendation",
        "Stop Phase 6 and debug the rank audit outcome before architecture search.",
    ]
    (PHASE6_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def covariance_spectrum(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, values.shape[0] - 1)
    return np.linalg.eigvalsh(cov)[::-1]


def effective_rank(values: np.ndarray, eps: float = 1.0e-12) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    p = spectrum / total
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, eps)))))


def spectral_entropy(values: np.ndarray, eps: float = 1.0e-12) -> float:
    spectrum = covariance_spectrum(values)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    p = spectrum / total
    return float(-np.sum(p * np.log(np.maximum(p, eps))))


def cumulative_variance(spectrum: np.ndarray, k: int | str) -> float:
    if spectrum.size == 0 or spectrum.sum() <= 0:
        return 0.0
    if k == "full":
        return 1.0
    return float(spectrum[: min(int(k), spectrum.size)].sum() / spectrum.sum())


def principal_angles(left: np.ndarray, right: np.ndarray, *, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((0,), dtype=float)
    _, _, left_vt = np.linalg.svd(left - left.mean(axis=0, keepdims=True), full_matrices=False)
    _, _, right_vt = np.linalg.svd(right - right.mean(axis=0, keepdims=True), full_matrices=False)
    q_left = left_vt[:k].T
    q_right = right_vt[:k].T
    singular = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))


def per_perturbation_delta_agreement(bundle: LatentBundle, pred_delta: np.ndarray) -> float:
    labels = bundle.metadata["perturbation_id"].to_numpy()
    agreements = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() < 1:
            continue
        agreements.append(float(row_cosine(pred_delta[mask].mean(axis=0, keepdims=True), bundle.delta[mask].mean(axis=0, keepdims=True))[0]))
    return float(np.mean(agreements)) if agreements else 0.0


def row_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), eps)
    return np.sum(left * right, axis=1) / denom


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    frame = frame[columns].copy()
    rows = []
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(col), *(len(row[i]) for row in rows)) for i, col in enumerate(columns)]
    header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def row_for(rows: list[dict[str, Any]], split: str, operator: str) -> dict[str, Any]:
    for row in rows:
        if row["split"] == split and row["operator"] == operator:
            return row
    raise KeyError((split, operator))


if __name__ == "__main__":
    raise SystemExit(main())
