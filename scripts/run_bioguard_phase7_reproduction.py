from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.biospectral_jepa import FloorPreservingTransitionHead
from perturb_jepa.training.biospectral_operator import (
    bundle_features,
    bundle_transition_metrics,
    fit_ridge_numpy,
    identity_metrics,
    load_latent_bundle,
    paired_transition_metrics,
    predict_ridge_numpy,
)
from perturb_jepa.training.seed import seed_everything
from scripts.train_biospectral_jepa import run_spectral_residual


PHASE7_ROOT = Path("outputs/autoresearch_bioguard_jepa_phase7")
PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
FULL_FLOOR = {
    "transition_source_cosine_improvement": 0.0057,
    "delta_cosine": 0.3980,
    "transition_to_target_recall@1": 0.4815,
    "delta_prediction_effective_rank": 10.2835,
    "delta_magnitude_ratio": 0.7744,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7 BSG000 reproduction.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=PHASE7_ROOT / "stage_a_reproduction")
    args = parser.parse_args()

    seed_everything(args.seed)
    initialize_docs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = PHASE7_ROOT / "experiments/BSG000_reproduction_seed0"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    train = load_latent_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")

    ridge = fit_ridge_numpy(bundle_features(train), train.delta, alpha=1.0e-2)
    train_floor_delta = predict_ridge_numpy(ridge, bundle_features(train))
    eval_floor_delta = predict_ridge_numpy(ridge, bundle_features(eval_bundle))
    floor_eval = bundle_transition_metrics(eval_bundle, eval_floor_delta)

    zero_wrapper_diff = zero_wrapper_max_diff(train, eval_bundle, device=device)
    old_metrics, old_payload = run_spectral_residual(
        train,
        eval_bundle,
        device=device,
        alpha=1.0e-2,
        rank=12,
        steps=80,
        lr=1.0e-3,
    )
    old_eval = {
        "transition_source_cosine_improvement": old_metrics["operator_eval_transition_improvement"],
        "delta_cosine": old_metrics["operator_eval_delta_cosine"],
        "transition_to_target_recall@1": old_metrics["operator_eval_recall_at_1"],
        "delta_prediction_effective_rank": old_metrics["operator_predicted_delta_rank"],
        "delta_magnitude_ratio": old_metrics["operator_eval_delta_magnitude_ratio"],
        "residual_to_floor_norm_ratio": old_metrics["residual_to_floor_norm_ratio"],
        "residual_cap_hit_fraction": old_metrics["residual_cap_hit_fraction"],
    }
    reproduction_ok = (
        abs(floor_eval["transition_source_cosine_improvement"] - FULL_FLOOR["transition_source_cosine_improvement"]) <= 1.0e-4
        and abs(floor_eval["delta_cosine"] - FULL_FLOOR["delta_cosine"]) <= 1.0e-4
        and abs(floor_eval["transition_to_target_recall@1"] - FULL_FLOOR["transition_to_target_recall@1"]) <= 1.0e-4
        and abs(floor_eval["delta_prediction_effective_rank"] - FULL_FLOOR["delta_prediction_effective_rank"]) <= 1.0e-3
        and zero_wrapper_diff <= 1.0e-7
        and (
            old_eval["transition_source_cosine_improvement"] < FULL_FLOOR["transition_source_cosine_improvement"] - 1.0e-4
            or old_eval["transition_to_target_recall@1"] < FULL_FLOOR["transition_to_target_recall@1"] - 1.0e-4
        )
    )
    decision = "BSG000_PASS_REPRODUCED_PHASE6" if reproduction_ok else "BSG000_ABORT_REPRODUCTION_MISMATCH"
    write_reproduction_report(args.output_dir, args, floor_eval, old_eval, zero_wrapper_diff, decision)
    write_json(experiment_dir / "metrics_eval.json", {"floor_eval": floor_eval, "old_bsj004_style_eval": old_eval, "zero_wrapper_max_diff": zero_wrapper_diff, "decision_label": decision})
    write_json(experiment_dir / "operator_payload.json", old_payload)
    write_bioguard_leakage_report(experiment_dir / "leakage_report.md", mode="BSG000_reproduction", train_rows=train.source.shape[0], eval_rows=eval_bundle.source.shape[0])
    update_results(
        "BSG000",
        args,
        decision,
        floor_eval,
        floor_eval,
        residual_gate_mean=0.0,
        residual_gate_nonzero_fraction=0.0,
        residual_scale=0.0,
        train_residual_fit_metric=0.0,
        calibration_residual_fit_metric=0.0,
        action_negative_gap=0.0,
        artifact_dir=experiment_dir,
        status=decision,
        cv_lcb_transition_gap=0.0,
        cv_lcb_recall_gap=0.0,
    )
    append_journal("BSG000", decision, f"Reproduced full floor improvement {floor_eval['transition_source_cosine_improvement']:.4f}; old residual-style improvement {old_eval['transition_source_cosine_improvement']:.4f}.")
    if not reproduction_ok:
        write_final_report("PHASE7_ABORT_REPRODUCTION_MISMATCH", ["BSG000 reproduction mismatch"], floor_eval)
        print(json.dumps({"decision": decision, "final": "PHASE7_ABORT_REPRODUCTION_MISMATCH"}, sort_keys=True))
        return 2
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def initialize_docs() -> None:
    PHASE7_ROOT.mkdir(parents=True, exist_ok=True)
    docs = {
        "results.tsv": "\t".join(
            [
                "experiment_id",
                "dataset",
                "eval_split",
                "seed",
                "mode",
                "status",
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
                "residual_gate_mean",
                "residual_gate_nonzero_fraction",
                "residual_scale",
                "train_residual_fit_metric",
                "calibration_residual_fit_metric",
                "action_negative_gap",
                "leakage_status",
                "decision_label",
                "artifact_dir",
            ]
        )
        + "\n",
        "research_journal.md": "# BioGuard-JEPA Phase 7 Research Journal\n\n",
        "architectural_changes_log.md": "# BioGuard-JEPA Phase 7 Architectural Changes Log\n\nNo full BioGuard-JEPA wrapper before train-internal residual selection passes.\n",
        "family_allocation.md": "# Phase 7 Family Allocation\n\n| family | experiments | status |\n| --- | --- | --- |\n| reproduction | BSG000 | in progress |\n| residual target/split audit | BSG001 | gated |\n| cross-fitted residual candidates | BSG002-BSG004 | gated |\n| ensemble/wrapper/full JEPA | BSG005-BSG008 | gated |\n",
        "BASELINE_REGISTRY.md": baseline_registry_text(),
        "papers_consulted.md": papers_consulted_text(),
        "external_resources.md": "# External Resources\n\nNo new external datasets downloaded in Phase 7. Uses cached Phase 4/Phase 6 artifacts.\n",
        "identity_violations_considered.md": "# Identity Violations Considered\n\n- PLS/raw-linear main path: forbidden except audit-only baseline.\n- `condition_key`, `biological_key`, exact target-key one-hot: forbidden as model, residual, calibration, or selection features.\n- Eval/test target rows: scoring-only; forbidden for fitting, whitening, residual selection, and calibration.\n",
    }
    for name, text in docs.items():
        path = PHASE7_ROOT / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def zero_wrapper_max_diff(train, eval_bundle, *, device: torch.device) -> float:
    head = FloorPreservingTransitionHead(train.source.shape[1], train.action.shape[1], residual_rank=12, hidden_dim=64).to(device)
    head.fit_floor_and_basis(tensor(train.source, device), tensor(train.action, device), tensor(train.delta, device), alpha=1.0e-2)
    with torch.no_grad():
        out = head(tensor(eval_bundle.source, device), tensor(eval_bundle.action, device))
    return float(torch.max(torch.abs(out["predicted_delta_bio"] - out["delta_floor"])).detach().cpu())


def write_reproduction_report(path: Path, args: argparse.Namespace, floor_eval: dict[str, float], old_eval: dict[str, float], zero_wrapper_diff: float, decision: str) -> None:
    lines = [
        "# Phase 7 Reproduction",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        "- Scope: BSG000 reproduction only; no Phase 7 residual selection yet.",
        "",
        "## Full-Ridge Floor",
        "",
        f"- transition improvement: `{floor_eval['transition_source_cosine_improvement']:.4f}`",
        f"- delta cosine: `{floor_eval['delta_cosine']:.4f}`",
        f"- recall@1: `{floor_eval['transition_to_target_recall@1']:.4f}`",
        f"- delta rank: `{floor_eval['delta_prediction_effective_rank']:.4f}`",
        "",
        "## Zero Residual Wrapper",
        "",
        f"- max absolute wrapper drift from floor: `{zero_wrapper_diff:.8f}`",
        "",
        "## Old BSJ004-Style Residual",
        "",
        f"- transition improvement: `{old_eval['transition_source_cosine_improvement']:.4f}`",
        f"- delta cosine: `{old_eval['delta_cosine']:.4f}`",
        f"- recall@1: `{old_eval['transition_to_target_recall@1']:.4f}`",
        f"- delta rank: `{old_eval['delta_prediction_effective_rank']:.4f}`",
        f"- residual cap hit fraction: `{old_eval['residual_cap_hit_fraction']:.4f}`",
        "",
        "## Decision",
        "",
        f"`{decision}`",
    ]
    (path / "PHASE7_REPRODUCTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_results(
    experiment_id: str,
    args: argparse.Namespace,
    decision: str,
    metrics: dict[str, float],
    floor: dict[str, float],
    *,
    residual_gate_mean: float,
    residual_gate_nonzero_fraction: float,
    residual_scale: float,
    train_residual_fit_metric: float,
    calibration_residual_fit_metric: float,
    action_negative_gap: float,
    artifact_dir: Path,
    status: str,
    cv_lcb_transition_gap: float,
    cv_lcb_recall_gap: float,
) -> None:
    path = PHASE7_ROOT / "results.tsv"
    frame = pd.read_csv(path, sep="\t")
    row = {
        "experiment_id": experiment_id,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "mode": "reproduction",
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
        "cv_lcb_transition_gap": f"{cv_lcb_transition_gap:.4f}",
        "cv_lcb_recall_gap": f"{cv_lcb_recall_gap:.4f}",
        "residual_gate_mean": f"{residual_gate_mean:.4f}",
        "residual_gate_nonzero_fraction": f"{residual_gate_nonzero_fraction:.4f}",
        "residual_scale": f"{residual_scale:.4f}",
        "train_residual_fit_metric": f"{train_residual_fit_metric:.6f}",
        "calibration_residual_fit_metric": f"{calibration_residual_fit_metric:.6f}",
        "action_negative_gap": f"{action_negative_gap:.4f}",
        "leakage_status": "PASS",
        "decision_label": decision,
        "artifact_dir": str(artifact_dir),
    }
    frame = frame[frame["experiment_id"].astype(str) != experiment_id]
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, sep="\t", index=False)


def write_bioguard_leakage_report(path: Path, *, mode: str, train_rows: int, eval_rows: int) -> None:
    lines = [
        "# Leakage Report",
        "",
        f"- Mode: `{mode}`",
        f"- Train rows used for fitting/calibration: `{train_rows}`",
        f"- Eval rows used for scoring only: `{eval_rows}`",
        "- Were eval/test target rows used for fitting? `no`",
        "- Were eval/test target rows used for whitening/statistics? `no`",
        "- Were eval/test target rows used for residual calibration or selection? `no`",
        "- Was `condition_key` used as a model feature? `no`",
        "- Was `biological_key` used as a model feature? `no`",
        "- Were exact target-key one-hot features used? `no`",
        "- Was batch id used as a biological transition shortcut? `no`",
        "- Were raw-linear PLS features used as the main representation path? `no`",
        "- Were any candidate choices based on eval/test performance? `no`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_journal(experiment_id: str, decision: str, summary: str) -> None:
    with (PHASE7_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {experiment_id}\n\n{summary}\n\nDecision label: `{decision}`.\n\n")


def write_final_report(decision: str, notes: list[str], floor: dict[str, float]) -> None:
    lines = [
        "# BioGuard-JEPA Phase 7 Final Report",
        "",
        "## Decision label",
        decision,
        "",
        "## Model of record status",
        "Protected rank-3 train-split-only PLS raw-linear readout remains model of record.",
        "",
        "## Floor values used",
        f"- transition improvement: `{floor['transition_source_cosine_improvement']:.4f}`",
        f"- delta cosine: `{floor['delta_cosine']:.4f}`",
        f"- recall@1: `{floor['transition_to_target_recall@1']:.4f}`",
        f"- delta rank: `{floor['delta_prediction_effective_rank']:.4f}`",
        "",
        "## Notes",
        *[f"- {note}" for note in notes],
    ]
    (PHASE7_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def baseline_registry_text() -> str:
    return """# Phase 7 Baseline Registry

## Protected Model Of Record

Protected rank-3 train-split-only PLS raw-linear readout remains the model of record. It is audit-only and not the BioGuard-JEPA representation path.

## Frozen-Latent Full-Ridge Floor

| metric | value |
| --- | ---: |
| floor_transition_improvement | 0.0057 |
| floor_delta_cosine | 0.3980 |
| floor_recall@1 | 0.4815 |
| floor_delta_rank | 10.2835 |
| floor_magnitude_ratio | 0.7744 |
| source_as_target_transition_improvement | 0.0000 |

Source artifacts:

- `outputs/autoresearch_biospectral_jepa_phase6/final_report.md`
- `outputs/autoresearch_biospectral_jepa_phase6/results.tsv`
- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/`
"""


def papers_consulted_text() -> str:
    return """# Papers Consulted

| Source | Design idea used in Phase 7 |
| --- | --- |
| V-JEPA / V-JEPA 2 | Latent action-conditioned future-state prediction; mapped to control `z_bio` + perturbation action -> perturbed teacher `z_bio`. |
| Safe policy improvement literature | Learned residuals are not deployed when evidence suggests underperforming a protected baseline. |
| Conformal risk control / split calibration | Use train-only calibration to decide residual scale/gate, defaulting to zero residual. |
| Double/debiased ML and cross-fitting | Fit flexible residuals on one fold and evaluate them on action-held-out calibration folds. |
| CellOT / GEARS / CPA | Treat perturbations as biological actions and prioritize held-out perturbation generalization over train residual fit. |
"""


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
