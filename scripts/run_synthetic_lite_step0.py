from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.baselines import (
    batch_only_retrieval_metrics,
    mean_prototype_alignment_metrics,
    metadata_only_retrieval_metrics,
)
from perturb_jepa.config import ExperimentConfig, ObjectiveScheduleConfig, OptimizerConfig, TrainingConfig
from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.losses import BridgeLossWeights
from perturb_jepa.models.bridge import PerturbJEPABridgeConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import (
    SyntheticBiologyLiteDataset,
    generate_synthetic_biology_lite,
    synthetic_lite_config,
)
from perturb_jepa.training.trainer import BridgeTrainer, loss_for_batch, move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 0 synthetic biology lite baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/autoresearch_synth_lite"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bag-size", type=int, default=None)
    parser.add_argument("--label-shuffle-repeats", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--rna-pooling", default="cls", choices=("cls", "mean_tokens"))
    parser.add_argument("--image-pooling", default="cls", choices=("cls", "mean_patches"))
    parser.add_argument(
        "--rna-condition-readout",
        default="encoder",
        choices=(
            "encoder",
            "pseudobulk",
            "encoder_plus_pseudobulk",
            "raw_pseudobulk",
            "encoder_plus_raw_pseudobulk",
            "raw_linear_pseudobulk",
            "encoder_plus_raw_linear_pseudobulk",
        ),
    )
    parser.add_argument("--no-rna-pseudobulk-normalize", action="store_true")
    parser.add_argument(
        "--image-condition-readout",
        default="encoder",
        choices=("encoder", "raw_pooled", "encoder_plus_raw_pooled", "raw_linear_pooled", "encoder_plus_raw_linear_pooled"),
    )
    parser.add_argument("--no-image-raw-normalize", action="store_true")
    parser.add_argument("--adversary-scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rna-mask-weight", type=float, default=0.2)
    parser.add_argument("--image-mask-weight", type=float, default=0.2)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--mmd-weight", type=float, default=0.05)
    parser.add_argument("--sliced-wasserstein-weight", type=float, default=0.02)
    parser.add_argument("--perturbation-cls-weight", type=float, default=0.05)
    parser.add_argument("--batch-adv-weight", type=float, default=0.02)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--cycle-weight", type=float, default=0.05)
    parser.add_argument("--response-bottleneck-weight", type=float, default=0.005)
    parser.add_argument("--shared-variance-weight", type=float, default=0.0)
    parser.add_argument("--shared-covariance-weight", type=float, default=0.0)
    parser.add_argument("--cross-correlation-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-rna-residual", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--bag-aggregator", default="attention", choices=("attention", "mean"))
    parser.add_argument("--num-bag-prototypes", type=int, default=2)
    parser.add_argument("--multi-positive-alignment", action="store_true")
    parser.add_argument("--schedule-reconstruction-warmup-steps", type=int, default=0)
    parser.add_argument("--schedule-reconstruction-anneal-steps", type=int, default=0)
    parser.add_argument("--schedule-reconstruction-final-scale", type=float, default=1.0)
    parser.add_argument("--schedule-warmup-non-reconstruction-scale", type=float, default=0.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    started = time.perf_counter()
    config = synthetic_lite_config(args.dataset, seed=args.seed)
    dataset = generate_synthetic_biology_lite(config)
    dataset_dir = args.output_root / "step0_baselines" / args.dataset
    dataset.export(dataset_dir)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        model_dim=args.model_dim,
        shared_dim=args.shared_dim,
        rna_pooling=args.rna_pooling,
        image_pooling=args.image_pooling,
        rna_condition_readout=args.rna_condition_readout,
        rna_pseudobulk_normalize=not args.no_rna_pseudobulk_normalize,
        image_condition_readout=args.image_condition_readout,
        image_raw_normalize=not args.no_image_raw_normalize,
        adversary_scale=args.adversary_scale,
        temperature=args.temperature,
        rna_mask_weight=args.rna_mask_weight,
        image_mask_weight=args.image_mask_weight,
        jepa_weight=args.jepa_weight,
        align_weight=args.align_weight,
        mmd_weight=args.mmd_weight,
        sliced_wasserstein_weight=args.sliced_wasserstein_weight,
        perturbation_cls_weight=args.perturbation_cls_weight,
        batch_adv_weight=args.batch_adv_weight,
        counterfactual_weight=args.counterfactual_weight,
        cycle_weight=args.cycle_weight,
        response_bottleneck_weight=args.response_bottleneck_weight,
        shared_variance_weight=args.shared_variance_weight,
        shared_covariance_weight=args.shared_covariance_weight,
        cross_correlation_weight=args.cross_correlation_weight,
        counterfactual_rna_residual=args.counterfactual_rna_residual,
        ema_decay=args.ema_decay,
        bag_aggregator=args.bag_aggregator,
        num_bag_prototypes=args.num_bag_prototypes,
        objective_schedule=_objective_schedule_from_args(args),
    )
    config_dir = args.output_root / "step0_baselines" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save_json(config_dir / f"{args.dataset}_seed{args.seed}_bridge.json")
    (config_dir / f"{args.dataset}_seed{args.seed}_generator.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model = experiment_config.build_model()
    optimizer = experiment_config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=experiment_config.loss,
        ema_decay=experiment_config.training.ema_decay,
        schedule=experiment_config.training.objective_schedule,
        device=args.device,
        grad_clip_norm=experiment_config.training.grad_clip_norm,
        multi_positive_alignment=args.multi_positive_alignment,
    )
    history = _train_with_early_stopping(
        trainer,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=args.bag_size or config.cells_per_condition,
        device=args.device,
        seed=args.seed,
    )
    split = args.eval_split
    metrics = evaluate_step0(
        dataset,
        model,
        split=split,
        train_split="train",
        device=args.device,
        bag_size=args.bag_size or config.cells_per_condition,
        seed=args.seed,
        label_shuffle_repeats=args.label_shuffle_repeats,
    )
    metrics["training_steps_completed"] = float(len(history))
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["device_used"] = args.device
    metrics["max_gpu_memory_gb"] = 0.0

    metrics_path = dataset_dir / f"step0_seed{args.seed}_metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history_path = dataset_dir / f"step0_seed{args.seed}_history.json"
    history_path.write_text(json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dataset_report(dataset_dir / f"{args.dataset}_baseline.md", args.dataset, args.seed, metrics, history)
    print(json.dumps(_jsonable(metrics), sort_keys=True))


def evaluate_step0(
    dataset: SyntheticBiologyLiteDataset,
    model,
    *,
    split: str,
    train_split: str,
    device: str,
    bag_size: int,
    seed: int,
    label_shuffle_repeats: int = 20,
) -> dict[str, Any]:
    model.eval()
    test = _collect_model_outputs(dataset, model, split=split, device=device, bag_size=bag_size, seed=seed)
    train = _collect_model_outputs(dataset, model, split=train_split, device=device, bag_size=bag_size, seed=seed + 101)
    metrics: dict[str, Any] = {}

    retrieval = cross_modal_retrieval_metrics(
        test["rna_shared"],
        test["image_shared"],
        test["metadata"],
        test["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    metrics.update(_prefix_dict("model", retrieval))

    rng = np.random.default_rng(seed)
    random_rna = rng.normal(size=test["rna_shared"].shape)
    random_image = rng.normal(size=test["image_shared"].shape)
    metrics.update(
        _prefix_dict(
            "random_embedding",
            cross_modal_retrieval_metrics(
                random_rna,
                random_image,
                test["metadata"],
                test["metadata"],
                ks=(1, 5),
                stratify_by=(),
            ),
        )
    )
    zeros = np.zeros_like(test["rna_shared"])
    metrics.update(
        _prefix_dict(
            "dataset_mean",
            cross_modal_retrieval_metrics(
                zeros,
                zeros,
                test["metadata"],
                test["metadata"],
                ks=(1, 5),
                stratify_by=(),
            ),
        )
    )
    metrics.update(_prefix_dict("metadata_only", metadata_only_retrieval_metrics(test["metadata"], test["metadata"], ks=(1, 5))))
    metrics.update(_prefix_dict("batch_only", batch_only_retrieval_metrics(test["metadata"], test["metadata"], ks=(1, 5))))
    image_flat = test["image_mean"].reshape(test["image_mean"].shape[0], -1)
    metrics.update(
        _prefix_dict(
            "mean_prototype_alignment",
            mean_prototype_alignment_metrics(test["metadata"], image_flat, test["metadata"], ks=(1, 5)),
        )
    )

    cf_arrays = _counterfactual_arrays(dataset, model, split=split, device=device, bag_size=bag_size, seed=seed)
    gene_sets = _gene_sets(dataset)
    if cf_arrays["observed"].shape[0] > 0:
        metrics.update(
            _prefix_dict(
                "model",
                rna_counterfactual_metrics(
                    cf_arrays["model_predicted"],
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )
        metrics.update(
            _prefix_dict(
                "source_as_target",
                rna_counterfactual_metrics(
                    cf_arrays["control"],
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )
        dataset_mean = np.broadcast_to(train["rna_mean"].mean(axis=0, keepdims=True), cf_arrays["observed"].shape)
        metrics.update(
            _prefix_dict(
                "dataset_mean_cf",
                rna_counterfactual_metrics(
                    dataset_mean,
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )

    metrics["model_bio_latent_r2_rna_shared"] = _latent_r2(
        train["rna_shared"],
        train["z_bio_mean"],
        test["rna_shared"],
        test["z_bio_mean"],
    )
    metrics["model_bio_latent_r2_image_shared"] = _latent_r2(
        train["image_shared"],
        train["z_bio_mean"],
        test["image_shared"],
        test["z_bio_mean"],
    )
    mapped_test = _linear_predict(train["rna_shared"], train["z_bio_mean"], test["rna_shared"])
    metrics["model_perturbation_direction_cosine"] = _perturbation_direction_cosine(test["metadata"], mapped_test, test["z_bio_mean"])
    metrics["model_dose_response_rank_correlation"] = _dose_response_rank_corr(test["metadata"], mapped_test)
    metrics["model_cell_line_baseline_recovery"] = _cell_line_baseline_recovery(
        test["metadata"],
        mapped_test,
        dataset.cell_line_baselines,
    )
    if cf_arrays["observed"].shape[0] > 0:
        metrics["model_program_level_effect_recovery"] = _program_effect_recovery(
            cf_arrays["model_predicted"],
            cf_arrays["observed"],
            cf_arrays["control"],
            dataset.gene_program_assignment,
        )
        metrics["source_as_target_program_level_effect_recovery"] = _program_effect_recovery(
            cf_arrays["control"],
            cf_arrays["observed"],
            cf_arrays["control"],
            dataset.gene_program_assignment,
        )

    batch_probe = batch_probe_metrics(test["rna_shared"], test["metadata"], label_col="batch")
    metrics.update(_prefix_dict("model", batch_probe))
    majority = batch_probe.get("batch_probe_majority_accuracy", 0.0)
    balanced = batch_probe.get("batch_probe_balanced_accuracy", float("nan"))
    metrics["model_batch_probe_balanced_accuracy_minus_majority"] = float(balanced - majority)
    metrics["model_bio_latent_r2_on_heldout_batch"] = float("nan")
    metrics["model_retrieval_drop_on_heldout_batch"] = float("nan")

    label_shuffle_values = []
    for _ in range(max(1, int(label_shuffle_repeats))):
        shuffled_metadata = test["metadata"].copy()
        shuffled_metadata["condition_key"] = rng.permutation(shuffled_metadata["condition_key"].to_numpy())
        label_shuffle_values.append(
            cross_modal_retrieval_metrics(
                test["rna_shared"],
                test["image_shared"],
                test["metadata"],
                shuffled_metadata,
                ks=(1, 5),
                stratify_by=(),
            )
        )
    label_shuffle = {
        key: float(np.mean([values[key] for values in label_shuffle_values]))
        for key in label_shuffle_values[0]
    }
    metrics.update(_prefix_dict("label_shuffle", label_shuffle))

    collapse = _collapse_diagnostics(test)
    metrics.update(collapse)
    metrics["collapse_flag"] = bool(collapse["model_rna_shared_min_std"] < 0.01 or collapse["model_image_shared_min_std"] < 0.01)
    return metrics


def _experiment_config_for_dataset(
    dataset: SyntheticBiologyLiteDataset,
    *,
    steps: int,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.05,
    model_dim: int = 32,
    shared_dim: int | None = None,
    rna_pooling: str = "cls",
    image_pooling: str = "cls",
    rna_condition_readout: str = "encoder",
    rna_pseudobulk_normalize: bool = True,
    image_condition_readout: str = "encoder",
    image_raw_normalize: bool = True,
    adversary_scale: float = 0.5,
    temperature: float = 0.1,
    rna_mask_weight: float = 0.2,
    image_mask_weight: float = 0.2,
    jepa_weight: float = 1.0,
    align_weight: float = 1.0,
    mmd_weight: float = 0.05,
    sliced_wasserstein_weight: float = 0.02,
    perturbation_cls_weight: float = 0.05,
    batch_adv_weight: float = 0.02,
    counterfactual_weight: float = 0.0,
    cycle_weight: float = 0.05,
    response_bottleneck_weight: float = 0.005,
    shared_variance_weight: float = 0.0,
    shared_covariance_weight: float = 0.0,
    cross_correlation_weight: float = 0.0,
    counterfactual_rna_residual: bool = False,
    counterfactual_rna_program_factorized: bool = False,
    counterfactual_rna_num_programs: int = 0,
    counterfactual_rna_program_assignment: tuple[int, ...] = (),
    counterfactual_rna_within_program_residual: bool = False,
    counterfactual_rna_program_conditioned: bool = False,
    counterfactual_rna_program_metadata_context: bool = False,
    counterfactual_rna_program_decoder_depth: int = 2,
    ema_decay: float = 0.996,
    bag_aggregator: str = "attention",
    num_bag_prototypes: int = 2,
    objective_schedule: ObjectiveScheduleConfig | None = None,
) -> ExperimentConfig:
    config = dataset.config
    dim = int(model_dim)
    shared_width = int(shared_dim) if shared_dim is not None else dim
    model = PerturbJEPABridgeConfig(
        rna=RNAEncoderConfig(
            vocab_size=config.genes,
            dim=dim,
            depth=1,
            heads=4,
            max_genes=config.genes,
            pooling=rna_pooling,
        ),
        image=ImageEncoderConfig(
            in_channels=config.image_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=dim,
            depth=1,
            heads=4,
            pooling=image_pooling,
        ),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=config.num_perturbations,
            num_types=2,
            num_cell_lines=config.num_cell_lines,
            num_batches=config.num_batches,
            dim=dim,
        ),
        shared_dim=shared_width,
        num_bag_prototypes=num_bag_prototypes,
        dropout=dropout,
        adversary_scale=adversary_scale,
        bag_aggregator=bag_aggregator,
        rna_condition_readout=rna_condition_readout,
        rna_pseudobulk_normalize=rna_pseudobulk_normalize,
        image_condition_readout=image_condition_readout,
        image_raw_normalize=image_raw_normalize,
        counterfactual_rna_residual=counterfactual_rna_residual,
        counterfactual_rna_program_factorized=counterfactual_rna_program_factorized,
        counterfactual_rna_num_programs=counterfactual_rna_num_programs,
        counterfactual_rna_program_assignment=counterfactual_rna_program_assignment,
        counterfactual_rna_within_program_residual=counterfactual_rna_within_program_residual,
        counterfactual_rna_program_conditioned=counterfactual_rna_program_conditioned,
        counterfactual_rna_program_metadata_context=counterfactual_rna_program_metadata_context,
        counterfactual_rna_program_decoder_depth=counterfactual_rna_program_decoder_depth,
    )
    return ExperimentConfig(
        name=f"{config.name}-step0",
        model=model,
        optimizer=OptimizerConfig(lr=lr, weight_decay=weight_decay),
        training=TrainingConfig(
            steps=steps,
            batch_size=8,
            device=device,
            seed=config.seed,
            ema_decay=ema_decay,
            grad_clip_norm=1.0,
            log_every=0,
            objective_schedule=objective_schedule or ObjectiveScheduleConfig(),
        ),
        loss=BridgeLossWeights(
            temperature=temperature,
            rna_mask=rna_mask_weight,
            image_mask=image_mask_weight,
            jepa=jepa_weight,
            align=align_weight,
            mmd=mmd_weight,
            sliced_wasserstein=sliced_wasserstein_weight,
            perturbation_cls=perturbation_cls_weight,
            batch_adv=batch_adv_weight,
            counterfactual=counterfactual_weight,
            cycle=cycle_weight,
            response_bottleneck=response_bottleneck_weight,
            shared_variance=shared_variance_weight,
            shared_covariance=shared_covariance_weight,
            cross_correlation=cross_correlation_weight,
        ),
    )


def _train_with_early_stopping(
    trainer: BridgeTrainer,
    dataset: SyntheticBiologyLiteDataset,
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    device: str,
    seed: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    best_val = float("inf")
    stale_steps = 0
    patience = max(10, steps // 5)
    eval_every = max(10, min(25, steps // 4))
    train_batches = dataset.iter_condition_batches(
        split="train",
        batch_size=batch_size,
        bag_size=bag_size,
        steps=steps,
        seed=seed,
        device=device,
    )
    for step, batch in enumerate(train_batches, start=1):
        terms = trainer.step(batch)
        history.append(terms)
        if step % eval_every != 0:
            continue
        val_loss = _validation_loss(trainer, dataset, batch_size=batch_size, bag_size=bag_size, device=device, seed=seed + step)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            stale_steps = 0
        else:
            stale_steps += eval_every
        if stale_steps >= patience:
            break
    return history


def _validation_loss(
    trainer: BridgeTrainer,
    dataset: SyntheticBiologyLiteDataset,
    *,
    batch_size: int,
    bag_size: int,
    device: str,
    seed: int,
) -> float:
    trainer.model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataset.iter_condition_batches(
            split="val",
            batch_size=batch_size,
            bag_size=bag_size,
            steps=3,
            seed=seed,
            device=device,
            shuffle=True,
        ):
            batch = move_batch_to_device(batch, device)
            total, _ = loss_for_batch(
                trainer.model,
                batch,
                weights=trainer.weights,
                schedule=trainer.schedule,
                step=trainer.global_step,
                uncertainty_weighting=trainer.uncertainty_weighting,
                multi_positive_alignment=trainer.multi_positive_alignment,
            )
            losses.append(float(total.detach().cpu()))
    trainer.model.train()
    return float(np.mean(losses)) if losses else float("inf")


def _objective_schedule_from_args(args: argparse.Namespace) -> ObjectiveScheduleConfig:
    enabled = (
        args.schedule_reconstruction_warmup_steps > 0
        or args.schedule_reconstruction_anneal_steps > 0
        or args.schedule_reconstruction_final_scale != 1.0
        or args.schedule_warmup_non_reconstruction_scale != 0.0
    )
    return ObjectiveScheduleConfig(
        enabled=enabled,
        reconstruction_warmup_steps=args.schedule_reconstruction_warmup_steps,
        reconstruction_anneal_steps=args.schedule_reconstruction_anneal_steps,
        reconstruction_final_scale=args.schedule_reconstruction_final_scale,
        warmup_non_reconstruction_scale=args.schedule_warmup_non_reconstruction_scale,
    )


def _collect_model_outputs(
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
    if not groups:
        raise ValueError(f"split {split!r} has no condition bags")
    outputs: dict[str, list[np.ndarray]] = {
        "rna_shared": [],
        "image_shared": [],
        "rna_teacher_shared": [],
        "image_teacher_shared": [],
        "counterfactual_delta": [],
        "z_state": [],
        "rna_token_prediction": [],
    }
    rna_mean = []
    image_mean = []
    z_bio_mean = []
    rng = np.random.default_rng(seed)
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
            for key in outputs:
                value = result.get(key)
                if value is None:
                    continue
                outputs[key].append(value.detach().cpu().numpy().reshape(value.shape[0], -1))
            for group in selected:
                rna_mean.append(dataset.expression_values[group].mean(axis=0))
                image_mean.append(dataset.images[group].mean(axis=0))
                z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
    collected = {key: np.concatenate(value, axis=0) for key, value in outputs.items() if value}
    collected["metadata"] = metadata
    collected["rna_mean"] = np.stack(rna_mean)
    collected["image_mean"] = np.stack(image_mean)
    collected["z_bio_mean"] = np.stack(z_bio_mean)
    return collected


def _counterfactual_arrays(
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
    key_to_group = {
        (
            int(row.perturbation_id),
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        ): group
        for row, group in zip(metadata.itertuples(index=False), groups, strict=True)
    }
    rows = []
    predicted = []
    observed = []
    control = []
    rng = np.random.default_rng(seed + 303)
    model.eval()
    with torch.no_grad():
        for row, target_group in zip(metadata.itertuples(index=False), groups, strict=True):
            if int(row.perturbation_id) == dataset.config.control_perturbation_id:
                continue
            control_key = (
                dataset.config.control_perturbation_id,
                int(row.cell_line_id),
                float(row.dose),
                int(row.batch_id),
            )
            control_group = key_to_group.get(control_key)
            if control_group is None:
                continue
            batch = dataset._make_bridge_batch(
                [control_group],
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            perturbation_id = torch.tensor([int(row.perturbation_id)], dtype=torch.long, device=device)
            perturbation_type_id = torch.tensor([int(row.perturbation_type_id)], dtype=torch.long, device=device)
            dose = torch.tensor([float(row.dose)], dtype=torch.float32, device=device)
            result = model(
                gene_ids=batch.gene_ids,
                expression_values=batch.expression_values,
                rna_token_mask=None,
                images=batch.images,
                image_patch_mask=None,
                perturbation_id=perturbation_id,
                perturbation_type_id=perturbation_type_id,
                cell_line_id=batch.cell_line_id,
                batch_id=batch.batch_id,
                dose=dose,
                time=batch.time,
            )
            predicted.append(result["counterfactual_rna"].detach().cpu().numpy()[0])
            observed.append(dataset.expression_values[target_group].mean(axis=0))
            control.append(dataset.expression_values[control_group].mean(axis=0))
            rows.append(row._asdict())
    if not predicted:
        genes = dataset.config.genes
        return {
            "model_predicted": np.empty((0, genes), dtype=float),
            "observed": np.empty((0, genes), dtype=float),
            "control": np.empty((0, genes), dtype=float),
            "metadata": pd.DataFrame(rows),
        }
    return {
        "model_predicted": np.asarray(predicted, dtype=float).reshape(len(predicted), -1),
        "observed": np.asarray(observed, dtype=float).reshape(len(observed), -1),
        "control": np.asarray(control, dtype=float).reshape(len(control), -1),
        "metadata": pd.DataFrame(rows),
    }


def _gene_sets(dataset: SyntheticBiologyLiteDataset) -> dict[str, list[int]]:
    result = {}
    for program in sorted(np.unique(dataset.gene_program_assignment)):
        result[f"program_{int(program)}"] = np.flatnonzero(dataset.gene_program_assignment == program).astype(int).tolist()
    return result


def _latent_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    pred = _linear_predict(x_train, y_train, x_test)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)


def _linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_eval = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(x_aug.shape[1])
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return x_eval @ coef


def _perturbation_direction_cosine(metadata: pd.DataFrame, mapped: np.ndarray, true_latent: np.ndarray) -> float:
    values = []
    control = metadata["perturbation_id"].to_numpy() == 0
    for perturbation_id in sorted(set(metadata["perturbation_id"])):
        if perturbation_id == 0:
            continue
        mask = metadata["perturbation_id"].to_numpy() == perturbation_id
        if not mask.any() or not control.any():
            continue
        pred_delta = mapped[mask].mean(axis=0) - mapped[control].mean(axis=0)
        true_delta = true_latent[mask].mean(axis=0) - true_latent[control].mean(axis=0)
        denom = np.linalg.norm(pred_delta) * np.linalg.norm(true_delta)
        if denom > 1e-12:
            values.append(float(np.dot(pred_delta, true_delta) / denom))
    return float(np.mean(values)) if values else 0.0


def _dose_response_rank_corr(metadata: pd.DataFrame, mapped: np.ndarray) -> float:
    correlations = []
    for (_, _), group in metadata.groupby(["perturbation_id", "cell_line_id"]):
        if group["dose"].nunique() < 2:
            continue
        indices = group.index.to_numpy()
        doses = group["dose"].to_numpy(dtype=float)
        norms = np.linalg.norm(mapped[indices] - mapped[indices].mean(axis=0, keepdims=True), axis=1)
        correlations.append(_spearman(doses, norms))
    return float(np.nanmean(correlations)) if correlations else 0.0


def _cell_line_baseline_recovery(metadata: pd.DataFrame, mapped: np.ndarray, baselines: np.ndarray) -> float:
    predictions = []
    targets = []
    for cell_line_id, group in metadata[metadata["perturbation_id"] == 0].groupby("cell_line_id"):
        predictions.append(mapped[group.index.to_numpy()].mean(axis=0))
        targets.append(baselines[int(cell_line_id)])
    if len(predictions) < 2:
        return 0.0
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    ss_res = float(np.sum((targets - predictions) ** 2))
    ss_tot = float(np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)


def _program_effect_recovery(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray, programs: np.ndarray) -> float:
    pred_delta = predicted - control
    obs_delta = observed - control
    values = []
    for program in sorted(np.unique(programs)):
        mask = programs == program
        values.append(_safe_corr(pred_delta[:, mask].mean(axis=1), obs_delta[:, mask].mean(axis=1)))
    return float(np.nanmean(values)) if values else 0.0


def _collapse_diagnostics(outputs: dict[str, Any]) -> dict[str, Any]:
    rna = outputs["rna_shared"]
    image = outputs["image_shared"]
    delta = outputs.get("counterfactual_delta", np.zeros_like(rna))
    state = outputs.get("z_state", np.ones_like(rna))
    rna_std = rna.std(axis=0)
    image_std = image.std(axis=0)
    rank = _rank(rna)
    image_rank = _rank(image)
    teacher_cos = _paired_cosine(outputs.get("rna_shared", rna), outputs.get("rna_teacher_shared", rna))
    image_teacher_cos = _paired_cosine(outputs.get("image_shared", image), outputs.get("image_teacher_shared", image))
    delta_ratio = float(np.linalg.norm(delta, axis=1).mean() / max(np.linalg.norm(state, axis=1).mean(), 1e-12))
    predictor_variance = float(outputs.get("rna_token_prediction", rna).var())
    spectrum = np.linalg.svd(rna - rna.mean(axis=0, keepdims=True), full_matrices=False, compute_uv=False)
    return {
        "model_rna_shared_min_std": float(rna_std.min()),
        "model_rna_shared_mean_std": float(rna_std.mean()),
        "model_image_shared_min_std": float(image_std.min()),
        "model_image_shared_mean_std": float(image_std.mean()),
        "model_embedding_rank": float(rank),
        "model_image_embedding_rank": float(image_rank),
        "model_covariance_spectrum_top5": [float(value) for value in spectrum[:5]],
        "model_student_teacher_cosine_mean": float(np.mean([teacher_cos[0], image_teacher_cos[0]])),
        "model_student_teacher_cosine_std": float(np.mean([teacher_cos[1], image_teacher_cos[1]])),
        "model_delta_norm_ratio": delta_ratio,
        "model_predictor_output_variance": predictor_variance,
    }


def _rank(values: np.ndarray) -> int:
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return int(np.sum(singular_values > 1e-3))


def _paired_cosine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y_norm = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    values = np.sum(x_norm * y_norm, axis=1)
    return float(values.mean()), float(values.std())


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_corr(_rankdata(x), _rankdata(y))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _write_dataset_report(path: Path, dataset_name: str, seed: int, metrics: dict[str, Any], history: list[dict[str, float]]) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {dataset_name} Step 0 Baseline",
                "",
                f"Seed: `{seed}`",
                "",
                f"Training steps completed: `{len(history)}`",
                f"Device: `{metrics.get('device_used', 'cpu')}`",
                f"Wallclock minutes: `{metrics.get('wallclock_minutes', 0.0):.3f}`",
                "",
                "## Key Metrics",
                "",
                f"- Model RNA->image recall@1: `{metrics.get('model_rna_to_image_recall@1', float('nan')):.4f}`",
                f"- Random RNA->image recall@1: `{metrics.get('random_embedding_rna_to_image_recall@1', float('nan')):.4f}`",
                f"- Batch-only recall@1: `{metrics.get('batch_only_batch_only_recall@1', float('nan')):.4f}`",
                f"- Model counterfactual direction accuracy: `{metrics.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
                f"- Source-as-target direction accuracy: `{metrics.get('source_as_target_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
                f"- RNA shared biological latent R2: `{metrics.get('model_bio_latent_r2_rna_shared', float('nan')):.4f}`",
                f"- Batch probe balanced accuracy: `{metrics.get('model_batch_probe_balanced_accuracy', float('nan')):.4f}`",
                f"- Embedding rank: `{metrics.get('model_embedding_rank', float('nan')):.1f}`",
                f"- Delta norm ratio: `{metrics.get('model_delta_norm_ratio', float('nan')):.4f}`",
                f"- Collapse flag: `{metrics.get('collapse_flag', False)}`",
                "",
                "Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
