from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.losses import bridge_loss
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.training.trainer import forward_batch, patchify_batch_images
from scripts.diagnose_pseudobulk_whitening_probe import (
    _cca_probe,
    _condition_arrays,
    _dual_ridge_probe,
    _fit_pca,
    _pls_probe,
    _ridge_probe,
)
from scripts.run_synthetic_lite_step0 import (
    _experiment_config_for_dataset,
    _jsonable,
    _latent_r2,
    evaluate_step0,
)


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PLS_TEACHER_REPAIR")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a low-compute PLS-whitened pseudobulk teacher repair.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--teacher", default="pls", choices=("pls", "ridge_image_pca", "regularized_cca", "supervised_z_bio"))
    parser.add_argument("--teacher-weight", type=float, default=5.0)
    parser.add_argument("--teacher-ridge", type=float, default=1e-2)
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bag-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rna-condition-readout", default="raw_pseudobulk")
    parser.add_argument("--image-condition-readout", default="raw_pooled")
    parser.add_argument("--no-rna-pseudobulk-normalize", action="store_true")
    parser.add_argument("--no-image-raw-normalize", action="store_true")
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--rna-mask-weight", type=float, default=0.05)
    parser.add_argument("--image-mask-weight", type=float, default=0.05)
    parser.add_argument("--perturbation-cls-weight", type=float, default=0.02)
    parser.add_argument("--batch-adv-weight", type=float, default=0.0)
    parser.add_argument("--shared-variance-weight", type=float, default=0.05)
    parser.add_argument("--shared-covariance-weight", type=float, default=0.01)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    bag_size = args.bag_size or dataset.config.cells_per_condition
    teacher = _fit_teacher(dataset, name=args.teacher, dim=args.model_dim, ridge=args.teacher_ridge)
    teacher_metrics = _teacher_probe_metrics(dataset, teacher)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        model_dim=args.model_dim,
        rna_condition_readout=args.rna_condition_readout,
        rna_pseudobulk_normalize=not args.no_rna_pseudobulk_normalize,
        image_condition_readout=args.image_condition_readout,
        image_raw_normalize=not args.no_image_raw_normalize,
        align_weight=args.align_weight,
        jepa_weight=args.jepa_weight,
        rna_mask_weight=args.rna_mask_weight,
        image_mask_weight=args.image_mask_weight,
        perturbation_cls_weight=args.perturbation_cls_weight,
        batch_adv_weight=args.batch_adv_weight,
        shared_variance_weight=args.shared_variance_weight,
        shared_covariance_weight=args.shared_covariance_weight,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        bag_aggregator="mean",
        num_bag_prototypes=1,
    )
    model = experiment_config.build_model().to(args.device)
    optimizer = experiment_config.build_optimizer(model.parameters())
    history = _train_with_teacher(
        model,
        optimizer,
        dataset,
        teacher,
        weights=experiment_config.loss,
        teacher_weight=args.teacher_weight,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        device=args.device,
        seed=args.seed,
        ema_decay=experiment_config.training.ema_decay,
        grad_clip_norm=experiment_config.training.grad_clip_norm,
    )
    metrics = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    metrics["training_steps_completed"] = float(len(history))
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["teacher"] = args.teacher
    metrics["teacher_weight"] = float(args.teacher_weight)
    metrics["model_dim"] = float(args.model_dim)
    metrics["tier1_pass_gate"] = _tier1_pass(metrics)

    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(args.output_dir / "bridge_config.json")
    (args.output_dir / "TEACHER_PROBE_METRICS.json").write_text(
        json.dumps(_jsonable(teacher_metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "TRAIN_HISTORY.json").write_text(
        json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MODEL_METRICS.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "REPORT.md", args=args, metrics=metrics, teacher_metrics=teacher_metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


class SharedTeacher:
    def __init__(self, transform: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]], *, dim: int) -> None:
        self._transform = transform
        self.dim = int(dim)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit_scale(self, train_arrays: dict[str, Any]) -> None:
        rna, image = self.transform_unscaled(train_arrays)
        pooled = np.concatenate([rna, image], axis=0)
        self._mean = pooled.mean(axis=0, keepdims=True)
        self._std = np.where(pooled.std(axis=0, keepdims=True) < 1e-6, 1.0, pooled.std(axis=0, keepdims=True))

    def transform_unscaled(self, arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        rna, image = self._transform(arrays)
        return _match_dim(rna, self.dim), _match_dim(image, self.dim)

    def transform(self, arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is None or self._std is None:
            raise RuntimeError("teacher scale has not been fitted")
        rna, image = self.transform_unscaled(arrays)
        return (rna - self._mean) / self._std, (image - self._mean) / self._std

    def batch_targets(self, batch, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = {
            "rna_mean": batch.expression_values.mean(dim=1).detach().cpu().numpy(),
            "image_mean": batch.images.mean(dim=1).detach().cpu().numpy().reshape(batch.images.shape[0], -1),
        }
        rna, image = self.transform(arrays)
        return (
            torch.as_tensor(rna, dtype=torch.float32, device=device),
            torch.as_tensor(image, dtype=torch.float32, device=device),
        )


def _fit_teacher(dataset, *, name: str, dim: int, ridge: float) -> SharedTeacher:
    train = _condition_arrays(dataset, "train")
    x = train["rna_mean"]
    y = train["image_mean"]
    if name == "pls":
        transform = _pls_probe(x, y, dim=dim)
    elif name == "ridge_image_pca":
        image_pca = _fit_pca(y, dim=dim, whiten=True)
        transform = _ridge_probe(
            x,
            image_pca(y),
            image_pca,
            ridge=ridge,
            x_key="rna_mean",
            y_key="image_mean",
        )
    elif name == "regularized_cca":
        transform = _cca_probe(x, y, dim=dim, ridge=ridge)
    elif name == "supervised_z_bio":
        transform = _dual_ridge_probe(
            x,
            train["z_bio"],
            y,
            train["z_bio"],
            ridge=ridge,
            x_key="rna_mean",
            y_key="image_mean",
        )
    else:
        raise ValueError(f"unsupported teacher: {name}")
    teacher = SharedTeacher(transform, dim=dim)
    teacher.fit_scale(train)
    return teacher


def _match_dim(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[1] == dim:
        return values
    if values.shape[1] > dim:
        return values[:, :dim]
    return np.pad(values, ((0, 0), (0, dim - values.shape[1])), mode="constant")


def _train_with_teacher(
    model,
    optimizer: torch.optim.Optimizer,
    dataset,
    teacher: SharedTeacher,
    *,
    weights,
    teacher_weight: float,
    steps: int,
    batch_size: int,
    bag_size: int,
    device: str,
    seed: int,
    ema_decay: float,
    grad_clip_norm: float | None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    train_batches = dataset.iter_condition_batches(
        split="train",
        batch_size=batch_size,
        bag_size=bag_size,
        steps=steps,
        seed=seed,
        device=device,
    )
    for step, batch in enumerate(train_batches, start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_batch(model, batch)
        image_patches = patchify_batch_images(batch.images, model.config.image.patch_size)
        total, terms = bridge_loss(
            outputs,
            rna_values=batch.expression_values,
            rna_mask=batch.rna_token_mask,
            image_patches=image_patches,
            image_patch_mask=batch.image_patch_mask,
            perturbation_id=batch.perturbation_id,
            batch_id=batch.batch_id,
            temperature=weights.temperature,
            weights=weights,
        )
        rna_target, image_target = teacher.batch_targets(batch, device=device)
        teacher_loss = _normalized_mse(outputs["rna_shared"], rna_target) + _normalized_mse(outputs["image_shared"], image_target)
        total = total + float(teacher_weight) * teacher_loss
        total.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        model.update_teachers(decay=ema_decay)
        row = {name: float(value.detach().cpu()) for name, value in terms.items()}
        row["teacher_loss"] = float(teacher_loss.detach().cpu())
        row["total_with_teacher"] = float(total.detach().cpu())
        row["step"] = float(step)
        history.append(row)
    return history


def _normalized_mse(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(values, dim=-1), F.normalize(target.detach(), dim=-1))


def _teacher_probe_metrics(dataset, teacher: SharedTeacher) -> dict[str, float]:
    from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
    from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics

    arrays = {split: _condition_arrays(dataset, split) for split in ("train", "test")}
    train_rna, train_image = teacher.transform(arrays["train"])
    test_rna, test_image = teacher.transform(arrays["test"])
    retrieval = cross_modal_retrieval_metrics(
        test_rna,
        test_image,
        arrays["test"]["metadata"],
        arrays["test"]["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    batch = batch_probe_metrics(test_rna, arrays["test"]["metadata"], label_col="batch")
    return {
        "teacher_rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "teacher_rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "teacher_image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "teacher_rna_bio_latent_r2": _latent_r2(train_rna, arrays["train"]["z_bio"], test_rna, arrays["test"]["z_bio"]),
        "teacher_image_bio_latent_r2": _latent_r2(train_image, arrays["train"]["z_bio"], test_image, arrays["test"]["z_bio"]),
        "teacher_batch_balanced_accuracy": batch.get("batch_probe_balanced_accuracy", float("nan")),
        "teacher_batch_majority_accuracy": batch.get("batch_probe_majority_accuracy", float("nan")),
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


def _write_report(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], teacher_metrics: dict[str, float]) -> None:
    lines = [
        "# PLS Teacher Repair Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Teacher: `{args.teacher}`",
        f"- Teacher weight: `{args.teacher_weight}`",
        f"- Model dim: `{args.model_dim}`",
        f"- Steps: `{args.steps}`",
        f"- Tier 1 pass: `{bool(metrics['tier1_pass_gate'])}`",
        "",
        "## Teacher Probe",
        "",
        f"- recall@1: `{teacher_metrics['teacher_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{teacher_metrics['teacher_rna_bio_latent_r2']:.4f}`",
        f"- Image latent R2: `{teacher_metrics['teacher_image_bio_latent_r2']:.4f}`",
        "",
        "## Model Result",
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
        "- `TEACHER_PROBE_METRICS.json`",
        "- `TRAIN_HISTORY.json`",
        "- `bridge_config.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
