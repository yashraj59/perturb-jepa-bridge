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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.losses import info_nce_loss, variance_floor_loss
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_distillation_head,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import SyntheticBridgeBatch
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, _latent_r2, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PLS_DISTILLED_HEAD")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a separate neural head to distill frozen PLS geometry without replacing retrieval."
    )
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.05)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--student-head", default="raw_mlp", choices=("raw_mlp", "linear_clone", "residual_mlp"))
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    run_dir = args.output_dir / f"{args.dataset}_{args.student_head}_seed{args.seed}_rank{args.rank}_s{args.steps}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
        dropout=0.0,
        model_dim=max(4, args.rank),
        shared_dim=args.rank,
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
    if args.student_head in {"linear_clone", "residual_mlp"}:
        install_prefit_pls_distillation_head(model, readout, freeze=False, device=args.device)
    _freeze_all_parameters(model)
    _unfreeze_distilled_heads(model, args.student_head)
    initial_readout = _frozen_readout_state(model)

    before_student = _evaluate_student_head(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        student_head=args.student_head,
    )
    protected_before = evaluate_step0(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )

    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    history = _train_distilled_heads(
        model,
        optimizer,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        seed=args.seed,
        device=args.device,
        alignment_weight=args.alignment_weight,
        variance_weight=args.variance_weight,
        student_head=args.student_head,
    )

    after_student = _evaluate_student_head(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        student_head=args.student_head,
    )
    protected_after = evaluate_step0(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    readout_drift = _max_frozen_readout_drift(initial_readout, _frozen_readout_state(model))
    protected_deltas = _protected_metric_deltas(protected_before, protected_after)
    after_student["training_steps_completed"] = float(len(history))
    after_student["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after_student["frozen_readout_max_abs_drift"] = readout_drift
    after_student["protected_geometry_preserved"] = bool(readout_drift <= 1e-7 and _protected_deltas_ok(protected_deltas))

    (run_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(run_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, run_dir / "prefit_pls_readout.json")
    _write_json(run_dir / "STUDENT_BEFORE_METRICS.json", before_student)
    _write_json(run_dir / "STUDENT_AFTER_METRICS.json", after_student)
    _write_json(run_dir / "PROTECTED_BEFORE_METRICS.json", protected_before)
    _write_json(run_dir / "PROTECTED_AFTER_METRICS.json", protected_after)
    _write_json(run_dir / "TRAIN_HISTORY.json", history)
    checkpoint_path = save_checkpoint(
        run_dir / "pls_distilled_head.pt",
        model=model,
        optimizer=optimizer,
        trainer_state={"steps": len(history)},
        experiment_config=experiment_config,
        metadata={
            "stage": "pls_distilled_head",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "student_head": args.student_head,
            "protected_metric_deltas": protected_deltas,
            "frozen_readout_max_abs_drift": readout_drift,
        },
    )
    _write_report(
        run_dir / "REPORT.md",
        args=args,
        before_student=before_student,
        after_student=after_student,
        protected_after=protected_after,
        protected_deltas=protected_deltas,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after_student), sort_keys=True))
    return 0


def _train_distilled_heads(
    model,
    optimizer: torch.optim.Optimizer,
    dataset,
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    seed: int,
    device: str,
    alignment_weight: float,
    variance_weight: float,
    student_head: str,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    batches = dataset.iter_condition_batches(
        split="train",
        batch_size=batch_size,
        bag_size=bag_size,
        steps=steps,
        seed=seed,
        device=device,
        rna_mask_prob=0.0,
        image_mask_prob=0.0,
    )
    model.train()
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_batch(model, batch)
        rna_student_key, image_student_key = _student_output_keys(student_head)
        rna_teacher = outputs["rna_raw_linear_shared"].detach()
        image_teacher = outputs["image_raw_linear_shared"].detach()
        rna_student = outputs[rna_student_key]
        image_student = outputs[image_student_key]
        rna_loss = _normalized_mse(rna_student, rna_teacher)
        image_loss = _normalized_mse(image_student, image_teacher)
        align_loss = info_nce_loss(rna_student, image_student) if alignment_weight else _zero_like(rna_loss)
        variance_loss = (
            variance_floor_loss(rna_student, target_std=0.05) + variance_floor_loss(image_student, target_std=0.05)
            if variance_weight
            else _zero_like(rna_loss)
        )
        total = rna_loss + image_loss + float(alignment_weight) * align_loss + float(variance_weight) * variance_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), 1.0)
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "total": float(total.detach().cpu()),
                "rna_teacher_mse": float(rna_loss.detach().cpu()),
                "image_teacher_mse": float(image_loss.detach().cpu()),
                "student_align": float(align_loss.detach().cpu()),
                "student_variance": float(variance_loss.detach().cpu()),
            }
        )
    return history


def _evaluate_student_head(
    dataset,
    model,
    *,
    split: str,
    train_split: str,
    device: str,
    bag_size: int,
    seed: int,
    student_head: str = "raw_mlp",
) -> dict[str, Any]:
    test = _collect_head_outputs(
        dataset,
        model,
        split=split,
        device=device,
        bag_size=bag_size,
        seed=seed,
        student_head=student_head,
    )
    train = _collect_head_outputs(
        dataset,
        model,
        split=train_split,
        device=device,
        bag_size=bag_size,
        seed=seed + 101,
        student_head=student_head,
    )
    retrieval = cross_modal_retrieval_metrics(
        test["rna_student"],
        test["image_student"],
        test["metadata"],
        test["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    batch = batch_probe_metrics(test["rna_student"], test["metadata"], label_col="batch")
    rna_std = test["rna_student"].std(axis=0)
    image_std = test["image_student"].std(axis=0)
    metrics: dict[str, Any] = {
        "student_rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "student_rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "student_image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "student_bio_latent_r2_rna_shared": _latent_r2(
            train["rna_student"],
            train["z_bio_mean"],
            test["rna_student"],
            test["z_bio_mean"],
        ),
        "student_bio_latent_r2_image_shared": _latent_r2(
            train["image_student"],
            train["z_bio_mean"],
            test["image_student"],
            test["z_bio_mean"],
        ),
        "student_rna_shared_min_std": float(rna_std.min()),
        "student_rna_shared_mean_std": float(rna_std.mean()),
        "student_image_shared_min_std": float(image_std.min()),
        "student_image_shared_mean_std": float(image_std.mean()),
        "student_teacher_rna_mse": float(np.mean((test["rna_student"] - test["rna_teacher"]) ** 2)),
        "student_teacher_image_mse": float(np.mean((test["image_student"] - test["image_teacher"]) ** 2)),
    }
    metrics.update({f"student_{key}": value for key, value in batch.items()})
    majority = float(batch.get("batch_probe_majority_accuracy", 0.0))
    balanced = float(batch.get("batch_probe_balanced_accuracy", float("nan")))
    metrics["student_batch_probe_balanced_accuracy_minus_majority"] = float(balanced - majority)
    metrics["student_collapse_flag"] = bool(
        metrics["student_rna_shared_min_std"] < 0.01 or metrics["student_image_shared_min_std"] < 0.01
    )
    metrics["student_tier1_pass_gate"] = _student_tier1_pass(metrics)
    if hasattr(model, "rna_distilled_residual_scale"):
        metrics["rna_distilled_residual_scale"] = float(model.rna_distilled_residual_scale.detach().cpu())
    if hasattr(model, "image_distilled_residual_scale"):
        metrics["image_distilled_residual_scale"] = float(model.image_distilled_residual_scale.detach().cpu())
    return metrics


def _collect_head_outputs(
    dataset,
    model,
    *,
    split: str,
    device: str,
    bag_size: int,
    seed: int,
    student_head: str,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    if not groups:
        raise ValueError(f"split {split!r} has no condition bags")
    outputs: dict[str, list[np.ndarray]] = {
        "rna_student": [],
        "image_student": [],
        "rna_teacher": [],
        "image_teacher": [],
    }
    z_bio_mean = []
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        rna_student_key, image_student_key = _student_output_keys(student_head)
        for start in range(0, len(groups), 16):
            selected = groups[start : start + 16]
            batch = dataset._make_bridge_batch(
                selected,
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            result = _forward_batch(model, batch)
            outputs["rna_student"].append(result[rna_student_key].detach().cpu().numpy())
            outputs["image_student"].append(result[image_student_key].detach().cpu().numpy())
            outputs["rna_teacher"].append(result["rna_raw_linear_shared"].detach().cpu().numpy())
            outputs["image_teacher"].append(result["image_raw_linear_shared"].detach().cpu().numpy())
            for group in selected:
                z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
    collected = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    collected["metadata"] = metadata
    collected["z_bio_mean"] = np.stack(z_bio_mean)
    return collected


def _forward_batch(model, batch: SyntheticBridgeBatch) -> dict[str, torch.Tensor]:
    return model(
        gene_ids=batch.gene_ids,
        expression_values=batch.expression_values,
        rna_token_mask=None,
        images=batch.images,
        image_patch_mask=None,
        perturbation_id=batch.perturbation_id,
        perturbation_type_id=batch.perturbation_type_id,
        cell_line_id=batch.cell_line_id,
        batch_id=batch.batch_id,
        dose=batch.dose,
        time=batch.time,
    )


def _student_tier1_pass(metrics: dict[str, Any]) -> bool:
    batch_balanced = float(metrics.get("student_batch_probe_balanced_accuracy", float("nan")))
    batch_majority = float(metrics.get("student_batch_probe_majority_accuracy", 0.5))
    return bool(
        not metrics.get("student_collapse_flag", True)
        and float(metrics.get("student_rna_to_image_recall@1", 0.0)) >= RANDOM_RECALL1 + 0.05
        and float(metrics.get("student_bio_latent_r2_rna_shared", -1.0)) > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= batch_majority + 0.10)
    )


def _normalized_mse(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.detach().std(dim=0, unbiased=False).clamp_min(1e-3)
    return (((student - target.detach()) / scale) ** 2).mean()


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=value.device, dtype=value.dtype)


def _freeze_all_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _unfreeze_distilled_heads(model, student_head: str) -> None:
    if student_head == "raw_mlp":
        modules = (model.rna_raw_pseudobulk_projection, model.image_raw_projection)
    elif student_head == "linear_clone":
        modules = (model.rna_distilled_linear_projection, model.image_distilled_linear_projection)
    elif student_head == "residual_mlp":
        modules = (model.rna_raw_pseudobulk_projection, model.image_raw_projection)
        model.rna_distilled_residual_scale.requires_grad_(True)
        model.image_distilled_residual_scale.requires_grad_(True)
    else:
        raise ValueError(f"unsupported student_head: {student_head}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _student_output_keys(student_head: str) -> tuple[str, str]:
    if student_head == "raw_mlp":
        return "rna_distilled_shared", "image_distilled_shared"
    if student_head == "linear_clone":
        return "rna_distilled_linear_shared", "image_distilled_linear_shared"
    if student_head == "residual_mlp":
        return "rna_distilled_residual_shared", "image_distilled_residual_shared"
    raise ValueError(f"unsupported student_head: {student_head}")


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before_student: dict[str, Any],
    after_student: dict[str, Any],
    protected_after: dict[str, Any],
    protected_deltas: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# PLS Distilled Head Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Student head: `{args.student_head}`",
        f"- Frozen readout max abs drift: `{after_student['frozen_readout_max_abs_drift']:.8f}`",
        f"- Protected geometry preserved: `{bool(after_student['protected_geometry_preserved'])}`",
        "",
        "## Student Head Before",
        "",
        f"- pass: `{bool(before_student['student_tier1_pass_gate'])}`",
        f"- recall@1: `{before_student['student_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{before_student['student_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{before_student['student_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Student Head After",
        "",
        f"- pass: `{bool(after_student['student_tier1_pass_gate'])}`",
        f"- recall@1: `{after_student['student_rna_to_image_recall@1']:.4f}`",
        f"- recall@5: `{after_student['student_rna_to_image_recall@5']:.4f}`",
        f"- RNA latent R2: `{after_student['student_bio_latent_r2_rna_shared']:.4f}`",
        f"- Image latent R2: `{after_student['student_bio_latent_r2_image_shared']:.4f}`",
        f"- RNA min std: `{after_student['student_rna_shared_min_std']:.4f}`",
        f"- Image min std: `{after_student['student_image_shared_min_std']:.4f}`",
        f"- batch balanced accuracy: `{after_student['student_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Protected PLS Retrieval Path",
        "",
        f"- recall@1: `{protected_after['model_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{protected_after['model_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{protected_after['model_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Protected Deltas",
        "",
    ]
    for key, value in protected_deltas.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `STUDENT_BEFORE_METRICS.json`",
            "- `STUDENT_AFTER_METRICS.json`",
            "- `PROTECTED_BEFORE_METRICS.json`",
            "- `PROTECTED_AFTER_METRICS.json`",
            "- `TRAIN_HISTORY.json`",
            "- `prefit_pls_readout.json`",
            f"- `{checkpoint_path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
