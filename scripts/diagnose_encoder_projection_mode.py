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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.training.trainer import BridgeTrainer
from scripts.run_synthetic_lite_step0 import (
    _experiment_config_for_dataset,
    _jsonable,
    _latent_r2,
    _train_with_early_stopping,
)


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/ENCODER_PROJECTION_MODE_AUDIT")
SPLITS = ("train", "val", "test")
MODES = ("eval", "train_no_grad")
STAGE_PAIRS = (
    ("raw_cls", "rna_raw_cls", "image_raw_cls"),
    ("token_patch_mean", "rna_token_mean", "image_patch_mean"),
    ("projection_pre_norm", "rna_projection_pre_norm", "image_projection_pre_norm"),
    ("projection_norm_instance_mean", "rna_projection_norm_instance_mean", "image_projection_norm_instance_mean"),
    ("shared_after_aggregator", "rna_shared", "image_shared"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit encoder/projection/dropout geometry on synthetic lite data.")
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
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    geometry_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
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
        history_rows.append(
            {
                "aggregator": aggregator,
                "steps_completed": len(history),
                "final_total": history[-1].get("total") if history else np.nan,
                "history_path": str(history_path),
            }
        )
        for mode in MODES:
            collected = {
                split: _collect_stages(
                    dataset,
                    model,
                    split=split,
                    mode=mode,
                    device=args.device,
                    bag_size=dataset.config.cells_per_condition,
                    seed=args.seed + 707,
                )
                for split in SPLITS
            }
            geometry_rows.extend(_geometry_rows(collected, aggregator=aggregator, mode=mode))
            retrieval_rows.extend(_retrieval_rows(collected, aggregator=aggregator, mode=mode))

    _write_tsv(args.output_dir / "ENCODER_PROJECTION_GEOMETRY.tsv", geometry_rows)
    _write_tsv(args.output_dir / "ENCODER_PROJECTION_RETRIEVAL.tsv", retrieval_rows)
    _write_tsv(args.output_dir / "TRAINING_HISTORY_SUMMARY.tsv", history_rows)
    _write_report(
        args.output_dir / "REPORT.md",
        geometry_rows=geometry_rows,
        retrieval_rows=retrieval_rows,
        history_rows=history_rows,
        wallclock_minutes=(time.perf_counter() - started) / 60.0,
    )
    return 0


def _train_model(dataset, *, aggregator: str, steps: int, batch_size: int, device: str, seed: int):
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


def _collect_stages(
    dataset,
    model,
    *,
    split: str,
    mode: str,
    device: str,
    bag_size: int,
    seed: int,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    rng = np.random.default_rng(seed)
    values: dict[str, list[np.ndarray]] = {
        "rna_raw_cls": [],
        "image_raw_cls": [],
        "rna_token_mean": [],
        "image_patch_mean": [],
        "rna_projection_pre_norm": [],
        "image_projection_pre_norm": [],
        "rna_projection_norm_instance_mean": [],
        "image_projection_norm_instance_mean": [],
        "rna_shared": [],
        "image_shared": [],
        "z_bio_mean": [],
    }
    if mode == "eval":
        model.eval()
    elif mode == "train_no_grad":
        model.train()
    else:
        raise ValueError(f"unknown mode: {mode}")

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
            flat_gene_ids = batch.gene_ids.reshape(-1, batch.gene_ids.shape[-1])
            flat_expression = batch.expression_values.reshape(-1, batch.expression_values.shape[-1])
            flat_images = batch.images.reshape(-1, *batch.images.shape[-3:])

            rna = model.rna_encoder(flat_gene_ids, flat_expression, token_mask=None)
            image = model.image_encoder(flat_images, patch_mask=None)
            rna_raw = rna.cell_embedding.reshape(batch.gene_ids.shape[0], bag_size, -1)
            image_raw = image.image_embedding.reshape(batch.images.shape[0], bag_size, -1)
            rna_token_mean = rna.token_embeddings.mean(dim=1).reshape(batch.gene_ids.shape[0], bag_size, -1)
            image_patch_mean = image.patch_embeddings.mean(dim=1).reshape(batch.images.shape[0], bag_size, -1)

            rna_projection_pre = model.rna_projection.net(rna.cell_embedding).reshape(batch.gene_ids.shape[0], bag_size, -1)
            image_projection_pre = model.image_projection.net(image.image_embedding).reshape(batch.images.shape[0], bag_size, -1)
            rna_projection_norm = F.normalize(rna_projection_pre, dim=-1)
            image_projection_norm = F.normalize(image_projection_pre, dim=-1)
            rna_shared = model.rna_bag_aggregator(rna_projection_norm).bag_embedding
            image_shared = model.image_bag_aggregator(image_projection_norm).bag_embedding

            values["rna_raw_cls"].append(rna_raw.mean(dim=1).detach().cpu().numpy())
            values["image_raw_cls"].append(image_raw.mean(dim=1).detach().cpu().numpy())
            values["rna_token_mean"].append(rna_token_mean.mean(dim=1).detach().cpu().numpy())
            values["image_patch_mean"].append(image_patch_mean.mean(dim=1).detach().cpu().numpy())
            values["rna_projection_pre_norm"].append(rna_projection_pre.mean(dim=1).detach().cpu().numpy())
            values["image_projection_pre_norm"].append(image_projection_pre.mean(dim=1).detach().cpu().numpy())
            values["rna_projection_norm_instance_mean"].append(rna_projection_norm.mean(dim=1).detach().cpu().numpy())
            values["image_projection_norm_instance_mean"].append(image_projection_norm.mean(dim=1).detach().cpu().numpy())
            values["rna_shared"].append(rna_shared.detach().cpu().numpy())
            values["image_shared"].append(image_shared.detach().cpu().numpy())
            values["z_bio_mean"].extend(dataset.z_bio[group].mean(axis=0) for group in selected)

    return {
        key: np.stack(parts).reshape(-1, np.stack(parts).shape[-1]) if key == "z_bio_mean" else np.concatenate(parts, axis=0)
        for key, parts in values.items()
        if parts
    } | {"metadata": metadata}


def _geometry_rows(collected: dict[str, dict[str, Any]], *, aggregator: str, mode: str) -> list[dict[str, Any]]:
    rows = []
    for split, arrays in collected.items():
        for key, values in arrays.items():
            if key == "metadata" or not isinstance(values, np.ndarray):
                continue
            if key == "z_bio_mean":
                continue
            row = {
                "aggregator": aggregator,
                "mode": mode,
                "split": split,
                "stage": key,
                **_vector_stats(values),
            }
            row["collapse_flag"] = bool(row["min_std"] < 0.01)
            row["bio_latent_r2_from_train"] = _latent_r2(
                collected["train"][key],
                collected["train"]["z_bio_mean"],
                values,
                arrays["z_bio_mean"],
            )
            rows.append(row)
    return rows


def _retrieval_rows(collected: dict[str, dict[str, Any]], *, aggregator: str, mode: str) -> list[dict[str, Any]]:
    rows = []
    for split, arrays in collected.items():
        metadata = arrays["metadata"]
        for pair_name, rna_stage, image_stage in STAGE_PAIRS:
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
                    "mode": mode,
                    "split": split,
                    "stage_pair": pair_name,
                    "rna_to_image_recall@1": metrics["rna_to_image_recall@1"],
                    "rna_to_image_recall@5": metrics["rna_to_image_recall@5"],
                    "image_to_rna_recall@1": metrics["image_to_rna_recall@1"],
                    "image_to_rna_recall@5": metrics["image_to_rna_recall@5"],
                    "rna_to_image_median_rank": metrics["rna_to_image_median_rank"],
                    "image_to_rna_median_rank": metrics["image_to_rna_median_rank"],
                }
            )
    return rows


def _vector_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values.reshape(values.shape[0], -1)
    std = values.std(axis=0)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices_from(cosine, k=1)] if cosine.shape[0] > 1 else np.asarray([], dtype=float)
    return {
        "rows": int(values.shape[0]),
        "dims": int(values.shape[1]),
        "min_std": float(std.min()) if std.size else 0.0,
        "mean_std": float(std.mean()) if std.size else 0.0,
        "rank": int(np.sum(singular > 1e-3)),
        "effective_rank": _effective_rank(singular),
        "mean_norm": float(np.linalg.norm(values, axis=1).mean()),
        "pairwise_cosine_mean": float(upper.mean()) if upper.size else 0.0,
        "pairwise_cosine_std": float(upper.std()) if upper.size else 0.0,
    }


def _effective_rank(singular_values: np.ndarray) -> float:
    total = float(np.sum(singular_values))
    if total <= 1e-12:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return float(np.exp(entropy))


def _write_report(
    path: Path,
    *,
    geometry_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    wallclock_minutes: float,
) -> None:
    geometry = pd.DataFrame(geometry_rows)
    retrieval = pd.DataFrame(retrieval_rows)

    def value(frame: pd.DataFrame, filters: dict[str, Any], column: str) -> Any:
        rows = frame
        for key, item in filters.items():
            rows = rows[rows[key] == item]
        return rows[column].iloc[0] if not rows.empty and column in rows else np.nan

    lines = [
        "# Encoder Projection Mode Audit",
        "",
        f"Wallclock minutes: `{wallclock_minutes:.3f}`",
        "",
        "## Main Finding",
        "",
        "This audit localizes whether collapse is already present in raw encoder CLS embeddings, appears in the projection head, appears after L2 normalization, or is hidden by train-mode dropout.",
        "",
        "## Key Rows",
        "",
    ]
    for aggregator in sorted(geometry["aggregator"].unique()):
        lines.extend([f"### {aggregator}", ""])
        for mode in MODES:
            lines.append(f"Mode `{mode}`:")
            for stage in (
                "rna_raw_cls",
                "rna_token_mean",
                "rna_projection_pre_norm",
                "rna_projection_norm_instance_mean",
                "rna_shared",
                "image_raw_cls",
                "image_patch_mean",
                "image_projection_pre_norm",
                "image_projection_norm_instance_mean",
                "image_shared",
            ):
                filters = {"aggregator": aggregator, "mode": mode, "split": "test", "stage": stage}
                lines.append(
                    f"- `{stage}` test min_std={_fmt(value(geometry, filters, 'min_std'))}, "
                    f"mean_std={_fmt(value(geometry, filters, 'mean_std'))}, "
                    f"bio_R2={_fmt(value(geometry, filters, 'bio_latent_r2_from_train'))}"
                )
            recall = value(
                retrieval,
                {"aggregator": aggregator, "mode": mode, "split": "test", "stage_pair": "shared_after_aggregator"},
                "rna_to_image_recall@1",
            )
            token_recall = value(
                retrieval,
                {"aggregator": aggregator, "mode": mode, "split": "test", "stage_pair": "token_patch_mean"},
                "rna_to_image_recall@1",
            )
            lines.append(f"- shared test recall@1={_fmt(recall)}; token/patch mean test recall@1={_fmt(token_recall)}")
            lines.append("")
    lines.extend(
        [
            "## Interpretation Rules",
            "",
            "- Raw CLS collapsed but token/patch mean not collapsed: use token/patch pooled condition embeddings instead of CLS.",
            "- Projection pre-norm healthy but normalized projection collapsed: change projection normalization/temperature geometry.",
            "- Eval collapsed but train-no-grad healthy: dropout is masking eval collapse and train-batch traces are overoptimistic.",
            "- Both eval and train-no-grad collapsed at raw CLS: encoder global token is the first failure point.",
            "",
            "## Artifacts",
            "",
            "- `ENCODER_PROJECTION_GEOMETRY.tsv`",
            "- `ENCODER_PROJECTION_RETRIEVAL.tsv`",
            "- `TRAINING_HISTORY_SUMMARY.tsv`",
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
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
