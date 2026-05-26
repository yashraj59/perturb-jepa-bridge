from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.training.trainer import BridgeTrainer
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FROZEN_PLS_JEPA")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train JEPA around a frozen prefit PLS shared readout.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=args.rank)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        model_dim=max(4, args.rank),
        shared_dim=args.rank,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.2,
        image_mask_weight=0.2,
        jepa_weight=1.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
    )
    model = experiment_config.build_model().to(args.device)
    install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
    initial_state = _frozen_readout_state(model)

    before = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    before["tier1_pass_gate"] = _tier1_pass(before)

    optimizer = experiment_config.build_optimizer(parameter for parameter in model.parameters() if parameter.requires_grad)
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=experiment_config.loss,
        ema_decay=experiment_config.training.ema_decay,
        device=args.device,
        grad_clip_norm=experiment_config.training.grad_clip_norm,
    )
    history = trainer.fit(
        dataset.iter_condition_batches(
            split="train",
            batch_size=args.batch_size,
            bag_size=bag_size,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
        ),
        steps=args.steps,
    )

    after = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    after["tier1_pass_gate"] = _tier1_pass(after)
    drift = _max_frozen_readout_drift(initial_state, _frozen_readout_state(model))
    protected = _protected_metric_deltas(before, after)
    after["frozen_readout_max_abs_drift"] = drift
    after["training_steps_completed"] = float(len(history))
    after["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after["protected_geometry_preserved"] = bool(drift <= 1e-7 and after["tier1_pass_gate"] and _protected_deltas_ok(protected))

    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(args.output_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, args.output_dir / "prefit_pls_readout.json")
    (args.output_dir / "BEFORE_METRICS.json").write_text(
        json.dumps(_jsonable(before), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "AFTER_METRICS.json").write_text(
        json.dumps(_jsonable(after), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "TRAIN_HISTORY.json").write_text(
        json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = trainer.save_checkpoint(
        args.output_dir / "frozen_pls_jepa.pt",
        experiment_config=experiment_config,
        metadata={
            "stage": "frozen_pls_jepa",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "protected_metric_deltas": protected,
            "frozen_readout_max_abs_drift": drift,
        },
    )
    _write_report(
        args.output_dir / "REPORT.md",
        args=args,
        before=before,
        after=after,
        protected=protected,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after), sort_keys=True))
    return 0


def _tier1_pass(metrics: dict[str, Any]) -> bool:
    batch_balanced = float(metrics.get("model_batch_probe_balanced_accuracy", float("nan")))
    batch_majority = float(metrics.get("model_batch_probe_majority_accuracy", 0.5))
    return bool(
        not metrics.get("collapse_flag", True)
        and float(metrics.get("model_rna_to_image_recall@1", 0.0)) >= RANDOM_RECALL1 + 0.05
        and float(metrics.get("model_bio_latent_r2_rna_shared", -1.0)) > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= batch_majority + 0.10)
    )


def _frozen_readout_state(model) -> dict[str, torch.Tensor]:
    return {
        "rna_weight": model.rna_raw_linear_projection.weight.detach().cpu().clone(),
        "rna_bias": model.rna_raw_linear_projection.bias.detach().cpu().clone(),
        "image_weight": model.image_raw_linear_projection.weight.detach().cpu().clone(),
        "image_bias": model.image_raw_linear_projection.bias.detach().cpu().clone(),
    }


def _max_frozen_readout_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return float(max((after[key] - before[key]).abs().max().item() for key in before))


def _protected_metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_to_image_recall@1",
        "model_rna_to_image_recall@5",
        "model_bio_latent_r2_rna_shared",
        "model_bio_latent_r2_image_shared",
        "model_rna_shared_min_std",
        "model_image_shared_min_std",
        "model_batch_probe_balanced_accuracy",
    )
    return {key: float(after.get(key, 0.0)) - float(before.get(key, 0.0)) for key in keys}


def _protected_deltas_ok(deltas: dict[str, float]) -> bool:
    return (
        deltas["model_rna_to_image_recall@1"] >= -1e-6
        and deltas["model_bio_latent_r2_rna_shared"] >= -1e-4
        and deltas["model_rna_shared_min_std"] >= -1e-6
        and deltas["model_image_shared_min_std"] >= -1e-6
        and deltas["model_batch_probe_balanced_accuracy"] <= 1e-6
    )


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before: dict[str, Any],
    after: dict[str, Any],
    protected: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# Frozen PLS JEPA Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Protected geometry preserved: `{bool(after['protected_geometry_preserved'])}`",
        f"- Frozen readout max abs drift: `{after['frozen_readout_max_abs_drift']:.8f}`",
        "",
        "## Before Training",
        "",
        f"- pass: `{bool(before['tier1_pass_gate'])}`",
        f"- recall@1: `{before['model_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{before['model_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{before['model_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## After Training",
        "",
        f"- pass: `{bool(after['tier1_pass_gate'])}`",
        f"- recall@1: `{after['model_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{after['model_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{after['model_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Protected Deltas",
        "",
    ]
    for key, value in protected.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `BEFORE_METRICS.json`",
            "- `AFTER_METRICS.json`",
            "- `TRAIN_HISTORY.json`",
            "- `prefit_pls_readout.json`",
            f"- `{checkpoint_path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
