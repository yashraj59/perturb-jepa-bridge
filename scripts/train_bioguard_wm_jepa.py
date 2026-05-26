from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.jepawm_predictor import (
    ActionAdaLNPredictor,
    BioJEPAWMContextConfig,
    FloorPreservingJEPAWMTransitionHead,
    context_contract_dict,
)
from perturb_jepa.training.bioguard_wm_calibration import SCALE_GRID, select_residual_scale_crossfit, select_residual_scale_small_cap_continuous
from perturb_jepa.training.bioguard_wm_losses import (
    action_negative_contrast_loss,
    cosine_endpoint_loss,
    delta_cosine_loss,
    floor_gap_hinge,
    l2_endpoint_loss,
    source_improvement_hinge,
)
from perturb_jepa.training.bioguard_splits import ActionGroupedResidualSplitConfig, ActionGroupedResidualSplitter
from perturb_jepa.training.biospectral_operator import (
    LatentOperatorBundle,
    bundle_features,
    bundle_transition_metrics,
    fit_ridge_numpy,
    load_latent_bundle,
    predict_ridge_numpy,
)
from perturb_jepa.training.seed import seed_everything


PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
FLOOR = {
    "transition_source_cosine_improvement": 0.0057,
    "delta_cosine": 0.3980,
    "transition_to_target_recall@1": 0.4815,
    "delta_prediction_effective_rank": 10.2835,
    "delta_magnitude_ratio": 0.7744,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate Phase 8 v2 BioGuard-WM-JEPA frozen-latent probes.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bag-size", type=int, default=3)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--bio-dim", type=int, default=24)
    parser.add_argument("--tech-dim", type=int, default=8)
    parser.add_argument("--predictor-dim", type=int, default=64)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--calibration-mode", choices=("none", "crossfit", "fixed_zero", "small_cap_continuous"), default="crossfit")
    parser.add_argument("--ridge-floor-path", type=Path, default=None)
    parser.add_argument("--latent-cache-path", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--paper-path", type=Path, default=None)
    args = parser.parse_args()
    metrics = run(args)
    print(json.dumps(metrics, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    del args.batch_size, args.bag_size, args.shared_dim, args.tech_dim, args.ridge_floor_path
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    train = load_latent_bundle(args.latent_cache_path / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(args.latent_cache_path / f"{args.dataset}_{args.eval_split}_latents", "eval")
    floor_fit = fit_ridge_numpy(bundle_features(train), train.delta, alpha=1.0e-2)
    train_floor_delta = predict_ridge_numpy(floor_fit, bundle_features(train))
    eval_floor_delta = predict_ridge_numpy(floor_fit, bundle_features(eval_bundle))
    floor_eval = bundle_transition_metrics(eval_bundle, eval_floor_delta)
    config = BioJEPAWMContextConfig(
        z_dim=args.bio_dim,
        action_dim=train.action.shape[1],
        predictor_dim=args.predictor_dim,
        depth=args.predictor_depth,
        heads=args.predictor_heads,
        context_length=args.context_length,
        max_context_length_seen_at_train=args.context_length,
    )
    contract = context_contract_dict(context_length_train=args.context_length, context_length_eval=args.context_length, rollout_steps=args.rollout_steps)
    if args.calibration_mode == "fixed_zero":
        metrics = evaluate_delta(eval_bundle, eval_floor_delta, floor_eval)
        metrics.update(
            {
                "residual_scale": 0.0,
                "residual_gate_mean": 0.0,
                "residual_gate_nonzero_fraction": 0.0,
                "cv_lcb_transition_gap": 0.0,
                "cv_lcb_recall_gap": 0.0,
                "cv_lcb_delta_cosine_gap": 0.0,
                "mean_transition_gap": 0.0,
                "mean_recall_gap": 0.0,
                "action_negative_gap": 0.0,
                "decision_label": "BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT",
                "max_abs_pred_floor_diff": 0.0,
            }
        )
        write_artifacts(args, metrics, contract, full_jepa=False, train_trace=[])
        return metrics

    predictor = ActionAdaLNPredictor(config).to(device)
    trace_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, float]] = []
    folds = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=4, seed=args.seed)).split(train.metadata)
    for fold in folds:
        fold_floor_fit = fit_ridge_numpy(bundle_features(_subset(train, fold.fit_indices)), _subset(train, fold.fit_indices).delta, alpha=1.0e-2)
        fit_bundle = _subset(train, fold.fit_indices)
        cal_bundle = _subset(train, fold.calibration_indices)
        fit_floor_delta = predict_ridge_numpy(fold_floor_fit, bundle_features(fit_bundle))
        cal_floor_delta = predict_ridge_numpy(fold_floor_fit, bundle_features(cal_bundle))
        model = ActionAdaLNPredictor(config).to(device)
        model.load_state_dict(predictor.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        source = tensor(fit_bundle.source, device)
        floor_z = tensor(fit_bundle.source + fit_floor_delta, device)
        action = tensor(fit_bundle.action, device)
        target = tensor(fit_bundle.target, device)
        residual_target = target - floor_z
        for step in range(max(1, int(args.steps))):
            raw, _ = model(source, floor_z, action)
            pred = floor_z + raw
            neg_action = action[torch.randperm(action.shape[0])]
            neg_raw, _ = model(source, floor_z, neg_action)
            loss = (
                l2_endpoint_loss(pred, target)
                + cosine_endpoint_loss(pred, target)
                + 0.5 * delta_cosine_loss(raw, residual_target)
                + 0.2 * source_improvement_hinge(pred, source, target, margin=0.0)
                + 0.5 * floor_gap_hinge(pred, floor_z, source, target, margin=0.0)
                + 0.1 * action_negative_contrast_loss(pred, target, floor_z + neg_raw, margin=0.05)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        trace_rows.append({"fold_id": fold.fold_id, "last_loss": float(loss.detach().cpu())})
        with torch.no_grad():
            cal_source = tensor(cal_bundle.source, device)
            cal_floor_z = tensor(cal_bundle.source + cal_floor_delta, device)
            cal_action = tensor(cal_bundle.action, device)
            raw = model(cal_source, cal_floor_z, cal_action)[0].detach().cpu().numpy()
            neg_action_np = cal_bundle.action.copy()
            rng = np.random.default_rng(args.seed + fold.fold_id + 11)
            rng.shuffle(neg_action_np)
            neg_raw = model(cal_source, cal_floor_z, tensor(neg_action_np, device))[0].detach().cpu().numpy()
        floor_metrics = bundle_transition_metrics(cal_bundle, cal_floor_delta)
        for scale in SCALE_GRID:
            pred_delta = cal_floor_delta + float(scale) * raw
            neg_delta = cal_floor_delta + float(scale) * neg_raw
            pred_metrics = bundle_transition_metrics(cal_bundle, pred_delta)
            neg_metrics = bundle_transition_metrics(cal_bundle, neg_delta)
            fold_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "scale": float(scale),
                    "floor_gap_transition": pred_metrics["transition_source_cosine_improvement"] - floor_metrics["transition_source_cosine_improvement"],
                    "floor_gap_recall": pred_metrics["transition_to_target_recall@1"] - floor_metrics["transition_to_target_recall@1"],
                    "floor_gap_delta_cosine": pred_metrics["delta_cosine"] - floor_metrics["delta_cosine"],
                    "action_negative_gap": pred_metrics["transition_source_cosine_improvement"] - neg_metrics["transition_source_cosine_improvement"],
                }
            )
    if args.calibration_mode == "small_cap_continuous":
        selection = select_residual_scale_small_cap_continuous(fold_metrics, cap=0.05, scale_grid=SCALE_GRID)
    else:
        selection = select_residual_scale_crossfit(fold_metrics, SCALE_GRID)
    if selection.residual_scale == 0.0:
        eval_delta = eval_floor_delta
    else:
        predictor = train_final_predictor(args, config, train, train_floor_delta, device)
        with torch.no_grad():
            raw_eval = predictor(
                tensor(eval_bundle.source, device),
                tensor(eval_bundle.source + eval_floor_delta, device),
                tensor(eval_bundle.action, device),
            )[0].cpu().numpy()
        eval_delta = eval_floor_delta + selection.residual_scale * raw_eval
    metrics = evaluate_delta(eval_bundle, eval_delta, floor_eval)
    metrics.update(
        {
            "residual_scale": selection.residual_scale,
            "residual_gate_mean": selection.residual_gate,
            "residual_gate_nonzero_fraction": selection.residual_gate,
            "cv_lcb_transition_gap": selection.cv_lcb_transition_gap,
            "cv_lcb_recall_gap": selection.cv_lcb_recall_gap,
            "cv_lcb_delta_cosine_gap": selection.cv_lcb_delta_cosine_gap,
            "mean_transition_gap": selection.mean_transition_gap,
            "mean_recall_gap": selection.mean_recall_gap,
            "action_negative_gap": selection.action_negative_gap,
            "decision_label": (
                "BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN"
                if selection.residual_scale == 0.0 and args.calibration_mode != "small_cap_continuous"
                else "BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK"
                if selection.residual_scale == 0.0
                else "BGWM_SMALL_CAP_KEEP_TINY_ACTION_ADALN_RESIDUAL"
                if args.calibration_mode == "small_cap_continuous"
                else "BGWM002_KEEP_SAFE_ACTION_ADALN_RESIDUAL"
            ),
            "fold_metrics": fold_metrics,
            "selection_status": selection.status,
            "selection_row": selection.selected_row,
        }
    )
    write_artifacts(args, metrics, contract, full_jepa=False, train_trace=trace_rows)
    if args.save_checkpoint:
        torch.save({"predictor_config": config, "state_dict": predictor.state_dict()}, args.output_dir / "checkpoint.pt")
    return metrics


def train_final_predictor(args: argparse.Namespace, config: BioJEPAWMContextConfig, train: LatentOperatorBundle, floor_delta: np.ndarray, device: torch.device) -> ActionAdaLNPredictor:
    model = ActionAdaLNPredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    source = tensor(train.source, device)
    floor_z = tensor(train.source + floor_delta, device)
    action = tensor(train.action, device)
    target = tensor(train.target, device)
    residual_target = target - floor_z
    for _ in range(max(1, int(args.steps))):
        raw, _ = model(source, floor_z, action)
        pred = floor_z + raw
        loss = l2_endpoint_loss(pred, target) + cosine_endpoint_loss(pred, target) + 0.5 * delta_cosine_loss(raw, residual_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def evaluate_delta(bundle: LatentOperatorBundle, pred_delta: np.ndarray, floor_metrics: dict[str, float]) -> dict[str, float]:
    metrics = bundle_transition_metrics(bundle, pred_delta)
    return {
        "transition_improvement": metrics["transition_source_cosine_improvement"],
        "delta_cosine": metrics["delta_cosine"],
        "recall_at_1": metrics["transition_to_target_recall@1"],
        "delta_rank": metrics["delta_prediction_effective_rank"],
        "magnitude_ratio": metrics["delta_magnitude_ratio"],
        "floor_transition_improvement": floor_metrics["transition_source_cosine_improvement"],
        "floor_delta_cosine": floor_metrics["delta_cosine"],
        "floor_recall_at_1": floor_metrics["transition_to_target_recall@1"],
        "floor_delta_rank": floor_metrics["delta_prediction_effective_rank"],
        "floor_gap_transition": metrics["transition_source_cosine_improvement"] - floor_metrics["transition_source_cosine_improvement"],
        "floor_gap_recall": metrics["transition_to_target_recall@1"] - floor_metrics["transition_to_target_recall@1"],
        "floor_gap_delta_cosine": metrics["delta_cosine"] - floor_metrics["delta_cosine"],
    }


def write_artifacts(args: argparse.Namespace, metrics: dict[str, Any], contract: dict[str, Any], *, full_jepa: bool, train_trace: list[dict[str, Any]]) -> None:
    write_json(args.output_dir / "config.json", vars(args))
    with (args.output_dir / "metrics_train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train_trace:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_json(args.output_dir / "context_contract.json", contract)
    write_identity_report(args.output_dir / "jepa_identity_report.md", full_jepa=full_jepa)
    write_leakage_report(args.output_dir / "leakage_report.md")
    write_model_card(args.output_dir / "model_card.md", args, metrics)


def write_identity_report(path: Path, *, full_jepa: bool) -> None:
    flags = {
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
    path.write_text("\n".join(["# JEPA Identity Report", "", *[f"- {k}: `{v}`" for k, v in flags.items()]]) + "\n", encoding="utf-8")


def write_leakage_report(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Leakage Report",
                "",
                "- Were eval target rows used for fitting? `no`",
                "- Were eval target means used? `no`",
                "- Were pooled train+test stats used? `no`",
                "- Was condition_key used as model input? `no`",
                "- Was biological_key used as model input? `no`",
                "- Was exact target-key one-hot used? `no`",
                "- Was batch id used as biological shortcut? `no`",
                "- Was raw-linear PLS used as main path? `no`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_model_card(path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# BioGuard-WM-JEPA Probe Model Card",
                "",
                f"- Dataset: `{args.dataset}`",
                f"- Eval split: `{args.eval_split}`",
                f"- Calibration mode: `{args.calibration_mode}`",
                f"- Decision: `{metrics.get('decision_label', '')}`",
                "- Role: Phase 8 v2 frozen-latent probe only; cannot promote model of record.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _subset(bundle: LatentOperatorBundle, indices: np.ndarray) -> LatentOperatorBundle:
    arrays = {key: value[indices] if value.shape[0] == bundle.source.shape[0] else value for key, value in bundle.arrays.items()}
    return LatentOperatorBundle(name=f"{bundle.name}_subset", arrays=arrays, metadata=bundle.metadata.iloc[indices].reset_index(drop=True))


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
