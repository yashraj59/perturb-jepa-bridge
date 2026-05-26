from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.prefit_readout import (
    PrefitPLSReadout,
    fit_pls_readout,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, _latent_r2, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PREFIT_PLS_READOUT")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate closed-form PLS whitening readouts inside the bridge model.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-standardize", action="store_true")
    args = parser.parse_args()
    dim = int(args.rank if args.rank is not None else args.dim)

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(
        train_arrays["rna_mean"],
        train_arrays["image_mean"],
        rank=dim,
        output_standardize=args.output_standardize,
    )
    direct_metrics = _direct_metrics(dataset, readout, eval_split=args.eval_split)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=0,
        device=args.device,
        model_dim=max(4, dim),
        shared_dim=dim,
        dropout=0.0,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
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

    metrics = evaluate_step0(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=dataset.config.cells_per_condition,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    metrics["training_steps_completed"] = 0.0
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["model_dim"] = float(dim)
    metrics["tier1_pass_gate"] = _tier1_pass(metrics)
    metrics["readout"] = "prefit_pls_linear"

    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(args.output_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, args.output_dir / "prefit_pls_readout.json")
    (args.output_dir / "DIRECT_PLS_METRICS.json").write_text(
        json.dumps(_jsonable(direct_metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MODEL_METRICS.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "REPORT.md", args=args, metrics=metrics, direct_metrics=direct_metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _direct_metrics(dataset, readout: PrefitPLSReadout, *, eval_split: str = "test") -> dict[str, float]:
    arrays = {split: _condition_arrays(dataset, split) for split in ("train", eval_split)}
    train_rna = readout.rna.transform(arrays["train"]["rna_mean"])
    train_image = readout.image.transform(arrays["train"]["image_mean"])
    test_rna = readout.rna.transform(arrays[eval_split]["rna_mean"])
    test_image = readout.image.transform(arrays[eval_split]["image_mean"])
    retrieval = cross_modal_retrieval_metrics(
        test_rna,
        test_image,
        arrays[eval_split]["metadata"],
        arrays[eval_split]["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    batch = batch_probe_metrics(test_rna, arrays[eval_split]["metadata"], label_col="batch")
    return {
        "direct_rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "direct_rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "direct_image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "direct_rna_bio_latent_r2": _latent_r2(train_rna, arrays["train"]["z_bio"], test_rna, arrays[eval_split]["z_bio"]),
        "direct_image_bio_latent_r2": _latent_r2(train_image, arrays["train"]["z_bio"], test_image, arrays[eval_split]["z_bio"]),
        "direct_batch_balanced_accuracy": batch.get("batch_probe_balanced_accuracy", float("nan")),
        "direct_batch_majority_accuracy": batch.get("batch_probe_majority_accuracy", float("nan")),
        "direct_rna_min_std": float(test_rna.std(axis=0).min()),
        "direct_image_min_std": float(test_image.std(axis=0).min()),
    }


def _tier1_pass(metrics: dict[str, Any]) -> bool:
    batch_balanced = float(metrics.get("model_batch_probe_balanced_accuracy", float("nan")))
    batch_majority = float(metrics.get("model_batch_probe_majority_accuracy", 0.5))
    return bool(
        not metrics.get("collapse_flag", True)
        and float(metrics.get("model_rna_to_image_recall@1", 0.0)) >= RANDOM_RECALL1 + 0.05
        and float(metrics.get("model_bio_latent_r2_rna_shared", -1.0)) > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= batch_majority + 0.10)
    )


def _write_report(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], direct_metrics: dict[str, float]) -> None:
    lines = [
        "# Prefit PLS Linear Readout Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Dim: `{args.rank if args.rank is not None else args.dim}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Tier 1 pass: `{bool(metrics['tier1_pass_gate'])}`",
        f"- Wallclock minutes: `{metrics['wallclock_minutes']:.3f}`",
        "",
        "## Direct PLS Check",
        "",
        f"- recall@1: `{direct_metrics['direct_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{direct_metrics['direct_rna_bio_latent_r2']:.4f}`",
        f"- Image latent R2: `{direct_metrics['direct_image_bio_latent_r2']:.4f}`",
        f"- RNA min std: `{direct_metrics['direct_rna_min_std']:.4f}`",
        f"- Image min std: `{direct_metrics['direct_image_min_std']:.4f}`",
        "",
        "## Bridge Result",
        "",
        f"- collapse flag: `{bool(metrics.get('collapse_flag'))}`",
        f"- recall@1: `{metrics.get('model_rna_to_image_recall@1', float('nan')):.4f}`",
        f"- recall@5: `{metrics.get('model_rna_to_image_recall@5', float('nan')):.4f}`",
        f"- RNA latent R2: `{metrics.get('model_bio_latent_r2_rna_shared', float('nan')):.4f}`",
        f"- Image latent R2: `{metrics.get('model_bio_latent_r2_image_shared', float('nan')):.4f}`",
        f"- RNA min std: `{metrics.get('model_rna_shared_min_std', float('nan')):.4f}`",
        f"- Image min std: `{metrics.get('model_image_shared_min_std', float('nan')):.4f}`",
        f"- Batch balanced accuracy: `{metrics.get('model_batch_probe_balanced_accuracy', float('nan')):.4f}`",
        "",
        "## Artifacts",
        "",
        "- `MODEL_METRICS.json`",
        "- `DIRECT_PLS_METRICS.json`",
        "- `prefit_pls_readout.json`",
        "- `bridge_config.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
