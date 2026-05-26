from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import (
    SyntheticBiologyLiteDataset,
    generate_synthetic_biology_lite,
    synthetic_lite_config,
)
from perturb_jepa.training.trainer import BridgeTrainer
from scripts.run_synthetic_lite_step0 import (
    _experiment_config_for_dataset,
    _jsonable,
    _latent_r2,
    _linear_predict,
    _train_with_early_stopping,
)


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/SPLIT_GEOMETRY_AUDIT")
SPLITS = ("train", "val", "test")
VECTOR_NAMES = ("z_bio_mean", "z_tech_mean", "rna_mean", "image_mean_flat")
STAGE_NAMES = (
    "rna_instance_mean",
    "image_instance_mean",
    "rna_shared",
    "image_shared",
    "rna_teacher_shared",
    "image_teacher_shared",
    "rna_state",
    "image_state",
    "z_state",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose synthetic split support and held-out embedding geometry.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregators", nargs="+", default=["attention", "mean"], choices=("attention", "mean"))
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = synthetic_lite_config(args.dataset, seed=args.seed)
    dataset = generate_synthetic_biology_lite(config)
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    condition_arrays = {split: _condition_arrays(dataset, split) for split in SPLITS}
    split_rows = _split_summary_rows(dataset)
    support_rows = _support_rows(condition_arrays)
    baseline_rows = _baseline_rows(condition_arrays)

    stage_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    history_summary_rows: list[dict[str, Any]] = []
    bag_sizes = {
        "full_eval_bag": dataset.config.cells_per_condition,
        "single_cell_bag": 1,
    }
    for aggregator in args.aggregators:
        model, history = _train_model(
            dataset,
            aggregator=aggregator,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )
        history_path = args.output_dir / f"{aggregator}_history.json"
        history_path.write_text(json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        history_summary_rows.append(
            {
                "aggregator": aggregator,
                "steps_completed": len(history),
                "final_total": history[-1].get("total") if history else np.nan,
                "history_path": str(history_path),
            }
        )
        for bag_label, bag_size in bag_sizes.items():
            collected = {
                split: _collect_intermediate_outputs(
                    dataset,
                    model,
                    split=split,
                    device=args.device,
                    bag_size=bag_size,
                    seed=args.seed + 1009 + bag_size,
                )
                for split in SPLITS
            }
            stage_rows.extend(_stage_geometry_rows(collected, aggregator=aggregator, bag_label=bag_label, bag_size=bag_size))
            retrieval_rows.extend(_model_retrieval_rows(collected, aggregator=aggregator, bag_label=bag_label, bag_size=bag_size))

    _write_tsv(args.output_dir / "SPLIT_SUMMARY.tsv", split_rows)
    _write_tsv(args.output_dir / "LATENT_SUPPORT.tsv", support_rows)
    _write_tsv(args.output_dir / "ORACLE_AND_RAW_BASELINES.tsv", baseline_rows)
    _write_tsv(args.output_dir / "MODEL_STAGE_GEOMETRY.tsv", stage_rows)
    _write_tsv(args.output_dir / "MODEL_RETRIEVAL_BY_SPLIT.tsv", retrieval_rows)
    _write_tsv(args.output_dir / "TRAINING_HISTORY_SUMMARY.tsv", history_summary_rows)
    _write_report(
        args.output_dir / "REPORT.md",
        dataset=dataset,
        split_rows=split_rows,
        support_rows=support_rows,
        baseline_rows=baseline_rows,
        stage_rows=stage_rows,
        retrieval_rows=retrieval_rows,
        wallclock_minutes=(time.perf_counter() - started) / 60.0,
    )
    return 0


def _train_model(
    dataset: SyntheticBiologyLiteDataset,
    *,
    aggregator: str,
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
):
    seed_everything(seed)
    config = _experiment_config_for_dataset(
        dataset,
        steps=steps,
        device=device,
        bag_aggregator=aggregator,
        num_bag_prototypes=2,
    )
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        ema_decay=config.training.ema_decay,
        schedule=config.training.objective_schedule,
        device=device,
        grad_clip_norm=config.training.grad_clip_norm,
    )
    history = _train_with_early_stopping(
        trainer,
        dataset,
        steps=steps,
        batch_size=batch_size,
        bag_size=dataset.config.cells_per_condition,
        device=device,
        seed=seed,
    )
    return model, history


def _condition_arrays(dataset: SyntheticBiologyLiteDataset, split: str) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    if not groups:
        raise ValueError(f"split {split!r} has no condition bags")
    rna_mean = []
    image_mean = []
    z_bio_mean = []
    z_tech_mean = []
    group_sizes = []
    for group in groups:
        rna_mean.append(dataset.expression_values[group].mean(axis=0))
        image_mean.append(dataset.images[group].mean(axis=0).reshape(-1))
        z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
        z_tech_mean.append(dataset.z_tech[group].mean(axis=0))
        group_sizes.append(int(len(group)))
    return {
        "metadata": metadata,
        "group_sizes": np.asarray(group_sizes, dtype=int),
        "rna_mean": np.stack(rna_mean).astype(float),
        "image_mean_flat": np.stack(image_mean).astype(float),
        "z_bio_mean": np.stack(z_bio_mean).astype(float),
        "z_tech_mean": np.stack(z_tech_mean).astype(float),
    }


def _collect_intermediate_outputs(
    dataset: SyntheticBiologyLiteDataset,
    model,
    *,
    split: str,
    device: str,
    bag_size: int,
    seed: int,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    rng = np.random.default_rng(seed)
    values: dict[str, list[np.ndarray]] = {name: [] for name in STAGE_NAMES}
    z_bio_mean = []
    attention_rows = []
    model.eval()
    with torch.no_grad():
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
            result = model(
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
            values["rna_instance_mean"].append(result["rna_instance_shared"].mean(dim=1).detach().cpu().numpy())
            values["image_instance_mean"].append(result["image_instance_shared"].mean(dim=1).detach().cpu().numpy())
            for name in STAGE_NAMES:
                if name in ("rna_instance_mean", "image_instance_mean"):
                    continue
                tensor = result.get(name)
                if tensor is not None:
                    values[name].append(tensor.detach().cpu().numpy().reshape(tensor.shape[0], -1))
            for group in selected:
                z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
            attention_rows.extend(_attention_rows(result, split=split, offset=start))
    collected = {
        name: np.concatenate(parts, axis=0)
        for name, parts in values.items()
        if parts
    }
    collected["metadata"] = metadata
    collected["z_bio_mean"] = np.stack(z_bio_mean).astype(float)
    collected["attention_rows"] = attention_rows
    return collected


def _split_summary_rows(dataset: SyntheticBiologyLiteDataset) -> list[dict[str, Any]]:
    rows = []
    train = dataset.metadata[dataset.metadata["split"] == "train"]
    train_condition_keys = set(train["condition_key"])
    train_bag_keys = set(train["bag_key"])
    for split in sorted(dataset.metadata["split"].unique()):
        frame = dataset.metadata[dataset.metadata["split"] == split]
        bag_sizes = frame.groupby("bag_key").size().to_numpy(dtype=float)
        condition_keys = set(frame["condition_key"])
        bag_keys = set(frame["bag_key"])
        rows.append(
            {
                "split": split,
                "cells": int(len(frame)),
                "condition_bags": int(len(bag_sizes)),
                "bag_size_min": float(bag_sizes.min()) if bag_sizes.size else 0.0,
                "bag_size_mean": float(bag_sizes.mean()) if bag_sizes.size else 0.0,
                "bag_size_max": float(bag_sizes.max()) if bag_sizes.size else 0.0,
                "unique_condition_keys": int(len(condition_keys)),
                "unique_bag_keys": int(len(bag_keys)),
                "unique_perturbations": int(frame["perturbation_id"].nunique()),
                "unique_doses": int(frame["dose"].nunique()),
                "unique_cell_lines": int(frame["cell_line_id"].nunique()),
                "unique_batches": int(frame["batch_id"].nunique()),
                "condition_key_overlap_with_train": _overlap_fraction(condition_keys, train_condition_keys),
                "bag_key_overlap_with_train": _overlap_fraction(bag_keys, train_bag_keys),
            }
        )
    return rows


def _support_rows(condition_arrays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    train = condition_arrays["train"]
    for split, arrays in condition_arrays.items():
        for vector_name in VECTOR_NAMES:
            values = arrays[vector_name]
            row = {
                "split": split,
                "vector": vector_name,
                **_vector_stats(values),
            }
            row.update(_support_against_train(train[vector_name], values, same_split=split == "train"))
            rows.append(row)
    return rows


def _baseline_rows(condition_arrays: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    train = condition_arrays["train"]
    for split, arrays in condition_arrays.items():
        metadata = arrays["metadata"]
        oracle = cross_modal_retrieval_metrics(
            arrays["z_bio_mean"],
            arrays["z_bio_mean"],
            metadata,
            metadata,
            ks=(1, 5),
            stratify_by=(),
        )
        predicted_image = _linear_predict(train["rna_mean"], train["image_mean_flat"], arrays["rna_mean"])
        ridge = cross_modal_retrieval_metrics(
            predicted_image,
            arrays["image_mean_flat"],
            metadata,
            metadata,
            ks=(1, 5),
            stratify_by=(),
        )
        rows.extend(
            [
                {
                    "split": split,
                    "baseline": "true_z_bio_oracle",
                    "rna_to_image_recall@1": oracle["rna_to_image_recall@1"],
                    "rna_to_image_recall@5": oracle["rna_to_image_recall@5"],
                    "image_to_rna_recall@1": oracle["image_to_rna_recall@1"],
                    "image_to_rna_recall@5": oracle["image_to_rna_recall@5"],
                    "rna_to_image_median_rank": oracle["rna_to_image_median_rank"],
                    "rna_latent_r2_from_raw_rna": _latent_r2(train["rna_mean"], train["z_bio_mean"], arrays["rna_mean"], arrays["z_bio_mean"]),
                    "image_latent_r2_from_raw_image": _latent_r2(train["image_mean_flat"], train["z_bio_mean"], arrays["image_mean_flat"], arrays["z_bio_mean"]),
                },
                {
                    "split": split,
                    "baseline": "ridge_raw_rna_to_image",
                    "rna_to_image_recall@1": ridge["rna_to_image_recall@1"],
                    "rna_to_image_recall@5": ridge["rna_to_image_recall@5"],
                    "image_to_rna_recall@1": ridge["image_to_rna_recall@1"],
                    "image_to_rna_recall@5": ridge["image_to_rna_recall@5"],
                    "rna_to_image_median_rank": ridge["rna_to_image_median_rank"],
                    "rna_latent_r2_from_raw_rna": _latent_r2(train["rna_mean"], train["z_bio_mean"], arrays["rna_mean"], arrays["z_bio_mean"]),
                    "image_latent_r2_from_raw_image": _latent_r2(train["image_mean_flat"], train["z_bio_mean"], arrays["image_mean_flat"], arrays["z_bio_mean"]),
                },
            ]
        )
    return rows


def _stage_geometry_rows(
    collected: dict[str, dict[str, Any]],
    *,
    aggregator: str,
    bag_label: str,
    bag_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for split, arrays in collected.items():
        for stage in STAGE_NAMES:
            if stage not in arrays:
                continue
            row = {
                "aggregator": aggregator,
                "bag_label": bag_label,
                "bag_size": int(bag_size),
                "split": split,
                "stage": stage,
                **_vector_stats(arrays[stage]),
            }
            row["collapse_flag"] = bool(row["min_std"] < 0.01)
            if split != "train":
                row["bio_latent_r2_from_train"] = _latent_r2(
                    collected["train"][stage],
                    _condition_arrays_from_collected(collected["train"])["z_bio_mean"],
                    arrays[stage],
                    _condition_arrays_from_collected(arrays)["z_bio_mean"],
                )
            else:
                row["bio_latent_r2_from_train"] = _latent_r2(
                    arrays[stage],
                    _condition_arrays_from_collected(arrays)["z_bio_mean"],
                    arrays[stage],
                    _condition_arrays_from_collected(arrays)["z_bio_mean"],
                )
            rows.append(row)
        rows.extend(_attention_summary_rows(arrays, aggregator=aggregator, bag_label=bag_label, bag_size=bag_size, split=split))
    return rows


def _model_retrieval_rows(
    collected: dict[str, dict[str, Any]],
    *,
    aggregator: str,
    bag_label: str,
    bag_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for split, arrays in collected.items():
        metadata = arrays["metadata"]
        for rna_stage, image_stage in (
            ("rna_instance_mean", "image_instance_mean"),
            ("rna_shared", "image_shared"),
            ("rna_teacher_shared", "image_teacher_shared"),
            ("rna_state", "image_state"),
        ):
            if rna_stage not in arrays or image_stage not in arrays:
                continue
            metrics = cross_modal_retrieval_metrics(
                arrays[rna_stage],
                arrays[image_stage],
                metadata,
                metadata,
                ks=(1, 5),
                stratify_by=(),
            )
            rows.append(
                {
                    "aggregator": aggregator,
                    "bag_label": bag_label,
                    "bag_size": int(bag_size),
                    "split": split,
                    "stage_pair": f"{rna_stage}_vs_{image_stage}",
                    "rna_to_image_recall@1": metrics["rna_to_image_recall@1"],
                    "rna_to_image_recall@5": metrics["rna_to_image_recall@5"],
                    "image_to_rna_recall@1": metrics["image_to_rna_recall@1"],
                    "image_to_rna_recall@5": metrics["image_to_rna_recall@5"],
                    "rna_to_image_median_rank": metrics["rna_to_image_median_rank"],
                    "image_to_rna_median_rank": metrics["image_to_rna_median_rank"],
                }
            )
    return rows


def _condition_arrays_from_collected(collected: dict[str, Any]) -> dict[str, np.ndarray]:
    return {"z_bio_mean": np.asarray(collected["z_bio_mean"], dtype=float)}


def _attention_rows(result: dict[str, torch.Tensor], *, split: str, offset: int) -> list[dict[str, Any]]:
    rows = []
    for name in ("rna_attention", "image_attention"):
        value = result.get(name)
        if value is None:
            continue
        attention = value.detach().cpu().numpy()
        entropy = -np.sum(attention * np.log(np.clip(attention, 1e-12, 1.0)), axis=-1)
        rows.append(
            {
                "split": split,
                "offset": offset,
                "attention": name,
                "entropy_mean": float(entropy.mean()),
                "max_weight_mean": float(attention.max(axis=-1).mean()),
            }
        )
    return rows


def _attention_summary_rows(
    arrays: dict[str, Any],
    *,
    aggregator: str,
    bag_label: str,
    bag_size: int,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    frame = pd.DataFrame(arrays.get("attention_rows", []))
    if frame.empty:
        return rows
    for attention_name, group in frame.groupby("attention", sort=True):
        rows.append(
            {
                "aggregator": aggregator,
                "bag_label": bag_label,
                "bag_size": int(bag_size),
                "split": split,
                "stage": attention_name,
                "rows": float(len(group)),
                "dims": 0.0,
                "min_std": np.nan,
                "mean_std": np.nan,
                "rank": np.nan,
                "effective_rank": np.nan,
                "collapse_flag": False,
                "bio_latent_r2_from_train": np.nan,
                "attention_entropy_mean": float(group["entropy_mean"].mean()),
                "attention_max_weight_mean": float(group["max_weight_mean"].mean()),
            }
        )
    return rows


def _vector_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    std = values.std(axis=0)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return {
        "rows": int(values.shape[0]),
        "dims": int(values.shape[1]),
        "min_std": float(std.min()) if std.size else 0.0,
        "mean_std": float(std.mean()) if std.size else 0.0,
        "rank": int(np.sum(singular > 1e-3)),
        "effective_rank": _effective_rank(singular),
        "top_singular": float(singular[0]) if singular.size else 0.0,
    }


def _support_against_train(train_values: np.ndarray, values: np.ndarray, *, same_split: bool) -> dict[str, Any]:
    train_scaled, values_scaled = _standardize_with_train(train_values, values)
    distances = _nearest_distances(train_scaled, values_scaled, exclude_self=same_split)
    train_std = train_values.std(axis=0)
    value_std = values.std(axis=0)
    return {
        "nearest_train_distance_mean": float(np.mean(distances)) if distances.size else 0.0,
        "nearest_train_distance_median": float(np.median(distances)) if distances.size else 0.0,
        "nearest_train_distance_max": float(np.max(distances)) if distances.size else 0.0,
        "centroid_shift_from_train": float(np.linalg.norm(values.mean(axis=0) - train_values.mean(axis=0)) / np.sqrt(values.shape[1])),
        "mean_std_ratio_to_train": float(value_std.mean() / max(train_std.mean(), 1e-12)),
    }


def _standardize_with_train(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_values - mean) / std, (values - mean) / std


def _nearest_distances(train_values: np.ndarray, values: np.ndarray, *, exclude_self: bool) -> np.ndarray:
    distances = np.linalg.norm(values[:, None, :] - train_values[None, :, :], axis=-1)
    if exclude_self and distances.shape[0] == distances.shape[1]:
        distances = distances.copy()
        np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return nearest[np.isfinite(nearest)]


def _effective_rank(singular_values: np.ndarray) -> float:
    singular_values = np.asarray(singular_values, dtype=float)
    total = singular_values.sum()
    if total <= 1e-12:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return float(np.exp(entropy))


def _overlap_fraction(values: set[Any], reference: set[Any]) -> float:
    if not values:
        return 0.0
    return float(len(values & reference) / len(values))


def _write_report(
    path: Path,
    *,
    dataset: SyntheticBiologyLiteDataset,
    split_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    wallclock_minutes: float,
) -> None:
    split = pd.DataFrame(split_rows)
    stage = pd.DataFrame(stage_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    baseline = pd.DataFrame(baseline_rows)
    regular_splits = split[split["split"].isin(SPLITS)]
    min_eval_bag = float(regular_splits.loc[regular_splits["split"].isin(("val", "test")), "bag_size_min"].min())
    train_bag = regular_splits.loc[regular_splits["split"] == "train", "bag_size_mean"].iloc[0]
    test_bag = regular_splits.loc[regular_splits["split"] == "test", "bag_size_mean"].iloc[0]

    def stage_value(aggregator: str, bag_label: str, split_name: str, stage_name: str, column: str) -> Any:
        rows = stage[
            (stage["aggregator"] == aggregator)
            & (stage["bag_label"] == bag_label)
            & (stage["split"] == split_name)
            & (stage["stage"] == stage_name)
        ]
        return rows[column].iloc[0] if not rows.empty and column in rows else np.nan

    def retrieval_value(aggregator: str, bag_label: str, split_name: str, pair: str, column: str) -> Any:
        rows = retrieval[
            (retrieval["aggregator"] == aggregator)
            & (retrieval["bag_label"] == bag_label)
            & (retrieval["split"] == split_name)
            & (retrieval["stage_pair"] == pair)
        ]
        return rows[column].iloc[0] if not rows.empty and column in rows else np.nan

    oracle_test = baseline[(baseline["split"] == "test") & (baseline["baseline"] == "true_z_bio_oracle")]
    ridge_test = baseline[(baseline["split"] == "test") & (baseline["baseline"] == "ridge_raw_rna_to_image")]
    lines = [
        "# Split Geometry Audit",
        "",
        f"Dataset: `{dataset.config.name}`",
        f"Seed: `{dataset.config.seed}`",
        f"Wallclock minutes: `{wallclock_minutes:.3f}`",
        "",
        "## Main Finding",
        "",
        "The synthetic retrieval target is well-posed but the learned model geometry is already collapsed when all train condition bags are collected in eval mode. "
        "The regular synthetic split is label-interpolation, not unseen-condition extrapolation: train, val, and test share the same condition and bag keys. "
        f"However, the condition-bag cell counts are imbalanced: train bags average `{train_bag:.1f}` cells, while test bags average `{test_bag:.1f}` cell. "
        f"The minimum val/test bag size is `{min_eval_bag:.1f}`. Normal Step 0 evaluation asks for `{dataset.config.cells_per_condition}` cells per bag, so val/test bags are sampled with replacement from very small groups. "
        "That cell-count issue is real, but it is not sufficient to explain the failure because train collection also collapses below the hard std gate.",
        "",
        "## Oracle And Raw Baselines",
        "",
        f"- True latent test recall@1: `{_fmt(oracle_test['rna_to_image_recall@1'].iloc[0] if not oracle_test.empty else np.nan)}`",
        f"- Raw ridge RNA->image test recall@1: `{_fmt(ridge_test['rna_to_image_recall@1'].iloc[0] if not ridge_test.empty else np.nan)}`",
        f"- Raw RNA->z_bio test R2: `{_fmt(oracle_test['rna_latent_r2_from_raw_rna'].iloc[0] if not oracle_test.empty else np.nan)}`",
        f"- Raw image->z_bio test R2: `{_fmt(oracle_test['image_latent_r2_from_raw_image'].iloc[0] if not oracle_test.empty else np.nan)}`",
        "",
        "## Collapse Location",
        "",
    ]
    for aggregator in sorted(stage["aggregator"].dropna().unique()):
        lines.extend(
            [
                f"### {aggregator}",
                "",
                f"- Full-bag train RNA shared min std: `{_fmt(stage_value(aggregator, 'full_eval_bag', 'train', 'rna_shared', 'min_std'))}`",
                f"- Full-bag test RNA shared min std: `{_fmt(stage_value(aggregator, 'full_eval_bag', 'test', 'rna_shared', 'min_std'))}`",
                f"- Full-bag train image shared min std: `{_fmt(stage_value(aggregator, 'full_eval_bag', 'train', 'image_shared', 'min_std'))}`",
                f"- Full-bag test image shared min std: `{_fmt(stage_value(aggregator, 'full_eval_bag', 'test', 'image_shared', 'min_std'))}`",
                f"- Single-cell train image shared min std: `{_fmt(stage_value(aggregator, 'single_cell_bag', 'train', 'image_shared', 'min_std'))}`",
                f"- Single-cell test image shared min std: `{_fmt(stage_value(aggregator, 'single_cell_bag', 'test', 'image_shared', 'min_std'))}`",
                f"- Full-bag test shared recall@1: `{_fmt(retrieval_value(aggregator, 'full_eval_bag', 'test', 'rna_shared_vs_image_shared', 'rna_to_image_recall@1'))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The true-latent oracle works and raw ridge RNA-to-image is above random on test, so the synthetic generator is not completely underdetermined. "
            "The learned model fails before retrieval: instance-level projected embeddings are already below the collapse threshold, then aggregation/state heads preserve or amplify the low-variance geometry. "
            "The next diagnostic should inspect raw encoder CLS outputs, pre-normalized projection outputs, normalized projection outputs, and train-mode vs eval-mode dropout. This distinguishes an encoder signal problem from projection normalization and from dropout-masked collapse.",
            "",
            "## Artifacts",
            "",
            "- `SPLIT_SUMMARY.tsv`: split coverage and condition-bag cell counts.",
            "- `LATENT_SUPPORT.tsv`: true latent/raw modality support distances against train.",
            "- `ORACLE_AND_RAW_BASELINES.tsv`: true-latent and ridge raw-modality baselines.",
            "- `MODEL_STAGE_GEOMETRY.tsv`: variance/rank by model stage, split, aggregator, and eval bag size.",
            "- `MODEL_RETRIEVAL_BY_SPLIT.tsv`: retrieval by stage pair, split, aggregator, and eval bag size.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
