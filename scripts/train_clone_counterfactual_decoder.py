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

from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_distillation_head,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train counterfactual decoder heads with PLS clone geometry frozen.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--program-weight", type=float, default=0.0)
    parser.add_argument("--program-factorized-rna", action="store_true")
    parser.add_argument("--within-program-residual", action="store_true")
    parser.add_argument("--program-condition-source", action="store_true")
    parser.add_argument("--program-metadata-context", action="store_true")
    parser.add_argument("--linear-program-decoder", action="store_true")
    parser.add_argument("--prefit-program-ridge", action="store_true")
    parser.add_argument("--prefit-program-ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--prefit-program-ridge-repeats", type=int, default=8)
    parser.add_argument("--delta-mse", action="store_true")
    parser.add_argument("--train-perturbation-encoder", action="store_true")
    parser.add_argument("--residual-rna", action="store_true")
    args = parser.parse_args()
    if args.linear_program_decoder and not args.program_factorized_rna:
        parser.error("--linear-program-decoder requires --program-factorized-rna")
    if args.prefit_program_ridge and not args.program_factorized_rna:
        parser.error("--prefit-program-ridge requires --program-factorized-rna")
    if args.prefit_program_ridge and not args.linear_program_decoder:
        parser.error("--prefit-program-ridge requires --linear-program-decoder")
    if args.prefit_program_ridge_repeats < 1:
        parser.error("--prefit-program-ridge-repeats must be >= 1")

    started = time.perf_counter()
    seed_everything(args.seed)
    perturb_mode = "perttrain" if args.train_perturbation_encoder else "pertfrozen"
    residual_mode = "rnaresidual" if args.residual_rna else "rnagenerative"
    direction_tag = f"dw{str(args.direction_weight).replace('.', 'p')}"
    program_tag = f"pw{str(args.program_weight).replace('.', 'p')}"
    factorized_tag = "pfact" if args.program_factorized_rna else "genehead"
    within_tag = "wres" if args.within_program_residual else "nowres"
    context_tag = "srcprog" if args.program_condition_source else "nostx"
    metadata_tag = "metactx" if args.program_metadata_context else "nometactx"
    depth_tag = "linearprog" if args.linear_program_decoder else "mlpprog"
    init_tag = "prefitridge" if args.prefit_program_ridge else "sgdinit"
    loss_tag = "deltamse" if args.delta_mse else "absmse"
    lr_tag = f"lr{_slug_float(args.lr)}"
    wd_tag = f"wd{_slug_float(args.weight_decay)}"
    batch_tag = f"bs{args.batch_size}"
    ridge_tag = ""
    if args.prefit_program_ridge:
        ridge_tag = f"_ra{_slug_float(args.prefit_program_ridge_alpha)}_rr{args.prefit_program_ridge_repeats}"
    run_dir = (
        args.output_dir
        / (
            f"{args.dataset}_{perturb_mode}_{residual_mode}_{factorized_tag}_{within_tag}_{context_tag}"
            f"_{metadata_tag}_{depth_tag}_{init_tag}_{loss_tag}"
            f"_{direction_tag}_{program_tag}"
            f"_{lr_tag}_{wd_tag}_{batch_tag}{ridge_tag}"
            f"_seed{args.seed}_rank{args.rank}_s{args.steps}"
        )
    )
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
        counterfactual_rna_residual=args.residual_rna,
        counterfactual_rna_program_factorized=args.program_factorized_rna,
        counterfactual_rna_num_programs=dataset.config.num_programs if args.program_factorized_rna else 0,
        counterfactual_rna_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment)
        if args.program_factorized_rna
        else (),
        counterfactual_rna_within_program_residual=args.within_program_residual,
        counterfactual_rna_program_conditioned=args.program_condition_source,
        counterfactual_rna_program_metadata_context=args.program_metadata_context,
        counterfactual_rna_program_decoder_depth=1 if args.linear_program_decoder else 2,
    )
    model = experiment_config.build_model().to(args.device)
    install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
    install_prefit_pls_distillation_head(model, readout, freeze=True, device=args.device)
    if args.program_factorized_rna:
        _zero_init_program_factorized_decoder(model, zero_within=args.within_program_residual)
    elif args.residual_rna:
        _zero_init_rna_delta_decoder(model)
    _freeze_all_parameters(model)
    _unfreeze_counterfactual_modules(
        model,
        train_perturbation_encoder=args.train_perturbation_encoder,
        program_factorized=args.program_factorized_rna,
        within_program_residual=args.within_program_residual,
    )
    initial_readout = _frozen_readout_state(model)

    before = evaluate_step0(
        dataset,
        model,
        split=args.eval_split,
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    pairs = _counterfactual_pairs(dataset, split="train")
    if not pairs:
        raise RuntimeError("no train counterfactual pairs were found")

    prefit_program_ridge = {}
    if args.prefit_program_ridge:
        prefit_program_ridge = _prefit_program_ridge_decoder(
            model,
            dataset,
            pairs,
            bag_size=bag_size,
            seed=args.seed,
            device=args.device,
            ridge=args.prefit_program_ridge_alpha,
            repeats=args.prefit_program_ridge_repeats,
        )

    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    history = _train_counterfactual(
        model,
        optimizer,
        dataset,
        pairs,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        seed=args.seed,
        device=args.device,
        direction_weight=args.direction_weight,
        program_weight=args.program_weight,
        delta_mse=args.delta_mse,
    )
    after = evaluate_step0(
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
    protected_deltas = _protected_metric_deltas(before, after)
    counterfactual_deltas = _counterfactual_deltas(before, after)
    after["training_steps_completed"] = float(len(history))
    after["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after["device_used"] = str(args.device)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        after["max_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    else:
        after["max_gpu_memory_gb"] = 0.0
    after["frozen_readout_max_abs_drift"] = readout_drift
    after["protected_geometry_preserved"] = bool(readout_drift <= 1e-7 and _protected_deltas_ok(protected_deltas))
    after["counterfactual_gate_pass"] = _counterfactual_gate_pass(before, after, protected_deltas, readout_drift)
    after["counterfactual_program_factorized"] = bool(args.program_factorized_rna)
    after["counterfactual_within_program_residual"] = bool(args.within_program_residual)
    after["counterfactual_program_conditioned"] = bool(args.program_condition_source)
    after["counterfactual_program_metadata_context"] = bool(args.program_metadata_context)
    after["counterfactual_program_decoder_depth"] = int(1 if args.linear_program_decoder else 2)
    after["counterfactual_prefit_program_ridge"] = bool(args.prefit_program_ridge)
    after["counterfactual_prefit_program_ridge_alpha"] = float(args.prefit_program_ridge_alpha)
    after["counterfactual_prefit_program_ridge_repeats"] = float(args.prefit_program_ridge_repeats)
    after["counterfactual_delta_mse_loss"] = bool(args.delta_mse)
    after.update({f"prefit_program_ridge_{key}": value for key, value in prefit_program_ridge.items()})
    after.update(_training_diagnostics(history))

    (run_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(run_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, run_dir / "prefit_pls_readout.json")
    if prefit_program_ridge:
        _write_json(run_dir / "PREFIT_PROGRAM_RIDGE.json", prefit_program_ridge)
    _write_json(run_dir / "BEFORE_METRICS.json", before)
    _write_json(run_dir / "AFTER_METRICS.json", after)
    _write_json(run_dir / "TRAIN_HISTORY.json", history)
    checkpoint_path = save_checkpoint(
        run_dir / "clone_counterfactual_decoder.pt",
        model=model,
        optimizer=optimizer,
        trainer_state={"steps": len(history)},
        experiment_config=experiment_config,
        metadata={
            "stage": "clone_counterfactual_decoder",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "protected_metric_deltas": protected_deltas,
            "counterfactual_metric_deltas": counterfactual_deltas,
            "frozen_readout_max_abs_drift": readout_drift,
            "counterfactual_program_factorized": bool(args.program_factorized_rna),
            "counterfactual_within_program_residual": bool(args.within_program_residual),
            "counterfactual_program_conditioned": bool(args.program_condition_source),
            "counterfactual_program_metadata_context": bool(args.program_metadata_context),
            "counterfactual_program_decoder_depth": int(1 if args.linear_program_decoder else 2),
            "counterfactual_prefit_program_ridge": bool(args.prefit_program_ridge),
            "prefit_program_ridge": prefit_program_ridge,
            "counterfactual_delta_mse_loss": bool(args.delta_mse),
        },
    )
    _write_report(
        run_dir / "REPORT.md",
        args=args,
        before=before,
        after=after,
        protected_deltas=protected_deltas,
        counterfactual_deltas=counterfactual_deltas,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after), sort_keys=True))
    return 0


def _train_counterfactual(
    model,
    optimizer: torch.optim.Optimizer,
    dataset,
    pairs: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    seed: int,
    device: str,
    direction_weight: float,
    program_weight: float,
    delta_mse: bool,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    program_assignment = torch.as_tensor(dataset.gene_program_assignment, dtype=torch.long, device=device)
    model.train()
    for step in range(1, steps + 1):
        selected = [pairs[int(index)] for index in rng.integers(0, len(pairs), size=batch_size)]
        batch = dataset._make_bridge_batch(
            [item["control_group"] for item in selected],
            bag_size=bag_size,
            rng=rng,
            rna_mask_prob=0.0,
            image_mask_prob=0.0,
            device=device,
        )
        target = torch.as_tensor(
            np.stack([dataset.expression_values[item["target_group"]].mean(axis=0) for item in selected]),
            dtype=torch.float32,
            device=device,
        )
        target_metadata = _target_metadata_tensors(selected, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            gene_ids=batch.gene_ids,
            expression_values=batch.expression_values,
            rna_token_mask=None,
            images=batch.images,
            image_patch_mask=None,
            perturbation_id=target_metadata["perturbation_id"],
            perturbation_type_id=target_metadata["perturbation_type_id"],
            cell_line_id=target_metadata["cell_line_id"],
            batch_id=target_metadata["batch_id"],
            dose=target_metadata["dose"],
            time=target_metadata["time"],
        )
        prediction = outputs["counterfactual_rna"]
        control = batch.expression_values.mean(dim=1)
        target_delta = target - control
        predicted_delta = prediction - control
        absolute_mse = F.mse_loss(prediction, target)
        delta_loss = F.mse_loss(predicted_delta, target_delta)
        mse = delta_loss if delta_mse else absolute_mse
        direction = 1.0 - F.cosine_similarity(predicted_delta, target_delta, dim=-1).mean()
        program_loss = _program_delta_mse(
            predicted_delta,
            target_delta,
            program_assignment=program_assignment,
        )
        total = mse + float(direction_weight) * direction + float(program_weight) * program_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), 1.0)
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "total": float(total.detach().cpu()),
                "counterfactual_mse": float(mse.detach().cpu()),
                "absolute_mse": float(absolute_mse.detach().cpu()),
                "delta_mse": float(delta_loss.detach().cpu()),
                "direction_loss": float(direction.detach().cpu()),
                "program_delta_mse": float(program_loss.detach().cpu()),
                "final_delta_to_target_ratio": _norm_ratio(predicted_delta, target_delta),
                "program_contribution_ratio": _norm_ratio(
                    outputs.get("counterfactual_rna_program_gene_delta"),
                    predicted_delta,
                ),
                "within_program_residual_ratio": _norm_ratio(
                    outputs.get("counterfactual_rna_within_program_residual"),
                    predicted_delta,
                ),
                "cap_hit_fraction": 0.0,
            }
        )
    return history


def _program_delta_mse(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    program_assignment: torch.Tensor,
) -> torch.Tensor:
    if prediction_delta.shape != target_delta.shape:
        raise ValueError("program delta loss inputs must have matching shapes")
    if program_assignment.shape != (prediction_delta.shape[-1],):
        program_assignment = program_assignment[: prediction_delta.shape[-1]]
    values = []
    for program in program_assignment.unique(sorted=True):
        mask = program_assignment.eq(program)
        values.append(F.mse_loss(prediction_delta[:, mask].mean(dim=1), target_delta[:, mask].mean(dim=1)))
    if not values:
        return torch.zeros((), device=prediction_delta.device, dtype=prediction_delta.dtype)
    return torch.stack(values).mean()


def _counterfactual_pairs(
    dataset,
    *,
    split: str,
    control_splits: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    if control_splits is None:
        control_splits = (split,)
    control_splits = tuple(dict.fromkeys(control_splits))
    control_groups: list[Any] = []
    control_metadata_rows: list[Any] = []
    control_split_by_key: dict[tuple[int, int, float, int], str] = {}
    for control_split in control_splits:
        try:
            split_groups = dataset.condition_bag_indices(split=control_split)
            split_metadata = dataset.metadata_for_condition_bags(split=control_split)
        except ValueError:
            continue
        for row, group in zip(split_metadata.itertuples(index=False), split_groups, strict=True):
            if int(row.perturbation_id) != dataset.config.control_perturbation_id:
                continue
            key = (
                int(row.perturbation_id),
                int(row.cell_line_id),
                float(row.dose),
                int(row.batch_id),
            )
            control_groups.append(group)
            control_metadata_rows.append(row)
            control_split_by_key.setdefault(key, control_split)
    key_to_group: dict[tuple[int, int, float, int], Any] = {}
    for row, group in zip(control_metadata_rows, control_groups, strict=True):
        key = (
            int(row.perturbation_id),
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        )
        key_to_group.setdefault(key, group)
    pairs: list[dict[str, Any]] = []
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
            control_group, control_split = _fallback_control_group(
                dataset,
                row,
                control_metadata_rows,
                control_groups,
                control_splits=control_splits,
            )
            if control_group is None:
                continue
        else:
            control_split = control_split_by_key.get(control_key, split)
        pairs.append(
            {
                "control_group": control_group,
                "target_group": target_group,
                "perturbation_id": int(row.perturbation_id),
                "perturbation_type_id": int(row.perturbation_type_id),
                "cell_line_id": int(row.cell_line_id),
                "batch_id": int(row.batch_id),
                "dose": float(row.dose),
                "time": float(row.time),
                "target_split": split,
                "control_split": control_split,
            }
        )
    return pairs


def _fallback_control_group(
    dataset,
    target_row,
    control_metadata_rows: list[Any],
    control_groups: list[Any],
    *,
    control_splits: tuple[str, ...],
) -> tuple[Any | None, str | None]:
    candidates = []
    split_order = {value: index for index, value in enumerate(control_splits)}
    for row, group in zip(control_metadata_rows, control_groups, strict=True):
        if int(row.perturbation_id) != dataset.config.control_perturbation_id:
            continue
        if int(row.cell_line_id) != int(target_row.cell_line_id):
            continue
        if abs(float(row.dose) - float(target_row.dose)) > 1e-8:
            continue
        batch_distance = abs(int(row.batch_id) - int(target_row.batch_id))
        split_value = getattr(row, "split", None)
        split_rank = split_order.get(str(split_value), len(split_order)) if split_value is not None else len(split_order)
        candidates.append((split_rank, batch_distance, int(row.batch_id), group, str(split_value or "")))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, group, split_value = candidates[0]
    return group, split_value


def _target_metadata_tensors(selected: list[dict[str, Any]], *, device: str) -> dict[str, torch.Tensor]:
    return {
        "perturbation_id": torch.tensor([item["perturbation_id"] for item in selected], dtype=torch.long, device=device),
        "perturbation_type_id": torch.tensor(
            [item["perturbation_type_id"] for item in selected],
            dtype=torch.long,
            device=device,
        ),
        "cell_line_id": torch.tensor([item["cell_line_id"] for item in selected], dtype=torch.long, device=device),
        "batch_id": torch.tensor([item["batch_id"] for item in selected], dtype=torch.long, device=device),
        "dose": torch.tensor([item["dose"] for item in selected], dtype=torch.float32, device=device),
        "time": torch.tensor([item["time"] for item in selected], dtype=torch.float32, device=device),
    }


def _prefit_program_ridge_decoder(
    model,
    dataset,
    pairs: list[dict[str, Any]],
    *,
    bag_size: int,
    seed: int,
    device: str,
    ridge: float,
    repeats: int,
) -> dict[str, float]:
    if not getattr(model.config, "counterfactual_rna_program_factorized", False):
        raise ValueError("program ridge prefit requires a program-factorized decoder")
    linear = model.counterfactual_program_decoder.net[-1]
    if len(model.counterfactual_program_decoder.net) != 1 or not isinstance(linear, torch.nn.Linear):
        raise TypeError("program ridge prefit requires a depth-1 linear program decoder")

    rng = np.random.default_rng(seed + 19_001)
    assignment = np.asarray(dataset.gene_program_assignment, dtype=np.int64)
    num_programs = int(model.config.counterfactual_rna_num_programs)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            batch = dataset._make_bridge_batch(
                [item["control_group"] for item in pairs],
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            target_metadata = _target_metadata_tensors(pairs, device=device)
            outputs = model(
                gene_ids=batch.gene_ids,
                expression_values=batch.expression_values,
                rna_token_mask=None,
                images=batch.images,
                image_patch_mask=None,
                perturbation_id=target_metadata["perturbation_id"],
                perturbation_type_id=target_metadata["perturbation_type_id"],
                cell_line_id=target_metadata["cell_line_id"],
                batch_id=target_metadata["batch_id"],
                dose=target_metadata["dose"],
                time=target_metadata["time"],
            )
            decoder_input = outputs.get("counterfactual_rna_program_decoder_input")
            if decoder_input is None:
                raise RuntimeError("model did not expose counterfactual_rna_program_decoder_input")
            x_blocks.append(decoder_input.detach().cpu().numpy())
            y_blocks.append(
                np.stack(
                    [
                        _program_means_np(
                            dataset.expression_values[item["target_group"]].mean(axis=0)
                            - dataset.expression_values[item["control_group"]].mean(axis=0),
                            assignment=assignment,
                            num_programs=num_programs,
                        )
                        for item in pairs
                    ],
                    axis=0,
                )
            )
    if was_training:
        model.train()
    x_train = np.concatenate(x_blocks, axis=0).astype(np.float64)
    y_train = np.concatenate(y_blocks, axis=0).astype(np.float64)
    if x_train.shape[1] != linear.in_features:
        raise ValueError(f"decoder input width {x_train.shape[1]} does not match linear width {linear.in_features}")
    x_aug = np.concatenate((x_train, np.ones((x_train.shape[0], 1), dtype=np.float64)), axis=1)
    penalty = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    prediction = x_aug @ coef
    residual = prediction - y_train
    total = y_train - y_train.mean(axis=0, keepdims=True)
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(total).sum())

    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(coef[:-1].T, dtype=linear.weight.dtype, device=linear.weight.device))
        linear.bias.copy_(torch.as_tensor(coef[-1], dtype=linear.bias.dtype, device=linear.bias.device))

    return {
        "train_rows": float(x_train.shape[0]),
        "feature_dim": float(x_train.shape[1]),
        "target_dim": float(y_train.shape[1]),
        "ridge_alpha": float(ridge),
        "repeats": float(repeats),
        "train_mse": float(np.square(residual).mean()),
        "train_program_r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "mean_target_program_delta_norm": float(np.linalg.norm(y_train, axis=1).mean()),
        "mean_predicted_program_delta_norm": float(np.linalg.norm(prediction, axis=1).mean()),
    }


def _program_means_np(values: np.ndarray, *, assignment: np.ndarray, num_programs: int) -> np.ndarray:
    if values.shape[-1] != assignment.shape[0]:
        raise ValueError("program assignment length must match values last dimension")
    result = np.zeros(num_programs, dtype=np.float64)
    for program in range(num_programs):
        mask = assignment == program
        if np.any(mask):
            result[program] = float(values[mask].mean())
    return result


def _freeze_all_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _zero_init_rna_delta_decoder(model) -> None:
    final = model.rna_distribution_decoder.net[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("rna_distribution_decoder must end with a Linear layer for zero init")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)


def _zero_init_program_factorized_decoder(model, *, zero_within: bool) -> None:
    final = model.counterfactual_program_decoder.net[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("counterfactual_program_decoder must end with a Linear layer for zero init")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)
    if zero_within:
        _zero_init_rna_delta_decoder(model)


def _unfreeze_counterfactual_modules(
    model,
    *,
    train_perturbation_encoder: bool,
    program_factorized: bool,
    within_program_residual: bool,
) -> None:
    modules = [
        model.state_head,
        model.response_head,
        model.delta_gate,
        model.delta_effect,
    ]
    if program_factorized:
        modules.append(model.counterfactual_program_decoder)
        if within_program_residual:
            modules.append(model.rna_distribution_decoder)
    else:
        modules.append(model.rna_distribution_decoder)
    if train_perturbation_encoder:
        modules.append(model.perturbation_encoder)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


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


def _counterfactual_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_counterfactual_direction_accuracy",
        "model_rna_counterfactual_logfc_correlation",
        "model_rna_counterfactual_pseudobulk_correlation",
        "model_program_level_effect_recovery",
        "model_rna_counterfactual_top50_de_overlap",
    )
    return {key: float(after.get(key, 0.0)) - float(before.get(key, 0.0)) for key in keys}


def _norm_ratio(numerator: torch.Tensor | None, denominator: torch.Tensor | None, *, eps: float = 1e-8) -> float:
    if numerator is None or denominator is None:
        return 0.0
    num = torch.linalg.vector_norm(numerator.detach(), dim=-1).mean()
    den = torch.linalg.vector_norm(denominator.detach(), dim=-1).mean().clamp_min(eps)
    return float((num / den).cpu())


def _training_diagnostics(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "mean_final_delta_to_target_ratio": 0.0,
            "mean_program_contribution_ratio": 0.0,
            "mean_within_program_residual_ratio": 0.0,
            "mean_cap_hit_fraction": 0.0,
        }
    keys = (
        "final_delta_to_target_ratio",
        "program_contribution_ratio",
        "within_program_residual_ratio",
        "cap_hit_fraction",
    )
    return {f"mean_{key}": float(np.mean([row.get(key, 0.0) for row in history])) for key in keys}


def _protected_deltas_ok(deltas: dict[str, float]) -> bool:
    return (
        deltas["model_rna_to_image_recall@1"] >= -1e-6
        and deltas["model_bio_latent_r2_rna_shared"] >= -1e-4
        and deltas["model_rna_shared_min_std"] >= -1e-6
        and deltas["model_image_shared_min_std"] >= -1e-6
        and deltas["model_batch_probe_balanced_accuracy"] <= 1e-6
    )


def _counterfactual_gate_pass(
    before: dict[str, Any],
    after: dict[str, Any],
    protected_deltas: dict[str, float],
    readout_drift: float,
) -> bool:
    return bool(
        readout_drift <= 1e-7
        and _protected_deltas_ok(protected_deltas)
        and float(after.get("model_rna_counterfactual_pseudobulk_correlation", -1.0))
        >= float(before.get("model_rna_counterfactual_pseudobulk_correlation", -1.0)) - 0.02
        and float(after.get("model_rna_counterfactual_direction_accuracy", -1.0))
        >= float(before.get("model_rna_counterfactual_direction_accuracy", -1.0)) + 0.10
        and float(after.get("model_rna_counterfactual_logfc_correlation", -1.0))
        >= float(before.get("model_rna_counterfactual_logfc_correlation", -1.0)) + 0.05
        and float(after.get("model_program_level_effect_recovery", -1.0))
        >= float(before.get("model_program_level_effect_recovery", -1.0)) + 0.05
        and float(after.get("model_rna_counterfactual_top50_de_overlap", -1.0))
        >= float(before.get("model_rna_counterfactual_top50_de_overlap", -1.0)) + 0.03
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before: dict[str, Any],
    after: dict[str, Any],
    protected_deltas: dict[str, float],
    counterfactual_deltas: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# Clone Counterfactual Decoder Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Train perturbation encoder: `{bool(args.train_perturbation_encoder)}`",
        f"- Residual RNA prediction: `{bool(args.residual_rna)}`",
        f"- Program-factorized RNA decoder: `{bool(args.program_factorized_rna)}`",
        f"- Within-program residual: `{bool(args.within_program_residual)}`",
        f"- Source-conditioned program decoder: `{bool(args.program_condition_source)}`",
        f"- Direct biological metadata context: `{bool(args.program_metadata_context)}`",
        f"- Delta MSE training loss: `{bool(args.delta_mse)}`",
        f"- Device used: `{args.device}`",
        f"- Max GPU memory GB: `{after.get('max_gpu_memory_gb', 0.0):.4f}`",
        f"- Protected geometry preserved: `{bool(after['protected_geometry_preserved'])}`",
        f"- Counterfactual gate pass: `{bool(after['counterfactual_gate_pass'])}`",
        f"- Frozen readout max abs drift: `{after['frozen_readout_max_abs_drift']:.8f}`",
        "",
        "## Counterfactual Metrics",
        "",
        f"- direction accuracy before: `{before.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
        f"- direction accuracy after: `{after.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
        f"- logFC correlation before: `{before.get('model_rna_counterfactual_logfc_correlation', float('nan')):.4f}`",
        f"- logFC correlation after: `{after.get('model_rna_counterfactual_logfc_correlation', float('nan')):.4f}`",
        f"- pseudobulk correlation before: `{before.get('model_rna_counterfactual_pseudobulk_correlation', float('nan')):.4f}`",
        f"- pseudobulk correlation after: `{after.get('model_rna_counterfactual_pseudobulk_correlation', float('nan')):.4f}`",
        f"- program recovery before: `{before.get('model_program_level_effect_recovery', float('nan')):.4f}`",
        f"- program recovery after: `{after.get('model_program_level_effect_recovery', float('nan')):.4f}`",
        f"- top50 overlap before: `{before.get('model_rna_counterfactual_top50_de_overlap', float('nan')):.4f}`",
        f"- top50 overlap after: `{after.get('model_rna_counterfactual_top50_de_overlap', float('nan')):.4f}`",
        "",
        "## Decoder Diagnostics",
        "",
        f"- mean final delta/target delta ratio: `{after.get('mean_final_delta_to_target_ratio', 0.0):.4f}`",
        f"- mean program contribution ratio: `{after.get('mean_program_contribution_ratio', 0.0):.4f}`",
        f"- mean within-program residual ratio: `{after.get('mean_within_program_residual_ratio', 0.0):.4f}`",
        f"- mean cap-hit fraction: `{after.get('mean_cap_hit_fraction', 0.0):.4f}`",
        "",
        "## Protected Deltas",
        "",
    ]
    for key, value in protected_deltas.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(["", "## Counterfactual Deltas", ""])
    for key, value in counterfactual_deltas.items():
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
