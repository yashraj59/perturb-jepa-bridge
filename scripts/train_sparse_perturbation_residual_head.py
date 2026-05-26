from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.models.sparse_perturbation_residual import (
    SparsePerturbationResidualHead,
    SparseResidualLossWeights,
    match_last_dim,
    program_means,
    sparse_delta_losses,
)
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
from scripts.run_synthetic_lite_step0 import (
    _experiment_config_for_dataset,
    _gene_sets,
    _jsonable,
    _program_effect_recovery,
    evaluate_step0,
)
from scripts.train_clone_counterfactual_decoder import (
    _counterfactual_deltas,
    _counterfactual_gate_pass,
    _counterfactual_pairs,
    _frozen_readout_state,
    _max_frozen_readout_drift,
    _norm_ratio,
    _protected_deltas_ok,
    _protected_metric_deltas,
    _target_metadata_tensors,
)


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/SPARSE_PERTURBATION_RESIDUAL_HEAD")


class SparseResidualBridgeWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, head: SparsePerturbationResidualHead) -> None:
        super().__init__()
        self.base_model = base_model
        self.sparse_residual_head = head

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        *,
        gene_ids: torch.Tensor | None = None,
        expression_values: torch.Tensor | None = None,
        rna_token_mask: torch.Tensor | None = None,
        rna_bag_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_patch_mask: torch.Tensor | None = None,
        image_bag_mask: torch.Tensor | None = None,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        descriptor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(
                gene_ids=gene_ids,
                expression_values=expression_values,
                rna_token_mask=rna_token_mask,
                rna_bag_mask=rna_bag_mask,
                images=images,
                image_patch_mask=image_patch_mask,
                image_bag_mask=image_bag_mask,
                perturbation_id=perturbation_id,
                perturbation_type_id=perturbation_type_id,
                cell_line_id=cell_line_id,
                batch_id=batch_id,
                dose=dose,
                time=time,
                descriptor=descriptor,
            )
        if expression_values is None or "rna_shared" not in outputs:
            return dict(outputs)
        source_rna = _condition_values(expression_values, self.sparse_residual_head.num_genes)
        head_outputs = self.sparse_residual_head(
            source_rna=source_rna,
            perturbation_id=perturbation_id,
            cell_line_id=cell_line_id,
            dose=dose,
            time=time,
        )
        counterfactual_delta = head_outputs["sparse_delta_hat"]
        counterfactual_rna = match_last_dim(source_rna, counterfactual_delta.shape[-1]) + counterfactual_delta
        combined = dict(outputs)
        combined.update(head_outputs)
        combined.update(
            {
                "z_inv": outputs["rna_shared"].detach(),
                "counterfactual_rna_delta": counterfactual_delta,
                "counterfactual_rna": counterfactual_rna,
                "counterfactual_sparse_delta": counterfactual_delta,
            }
        )
        return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a sparse perturbation-specific residual RNA delta head.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--z-pert-dim", type=int, default=16)
    parser.add_argument("--dictionary-rank", type=int, default=4)
    parser.add_argument("--top-de-k", type=int, default=50)
    parser.add_argument("--top-programs", type=int, default=2)
    parser.add_argument("--global-delta-weight", type=float, default=0.05)
    parser.add_argument("--top-de-weight", type=float, default=8.0)
    parser.add_argument("--program-gene-weight", type=float, default=4.0)
    parser.add_argument("--direction-weight", type=float, default=1.0)
    parser.add_argument("--sign-weight", type=float, default=0.25)
    parser.add_argument("--sparsity-weight", type=float, default=0.02)
    parser.add_argument("--decorrelation-weight", type=float, default=0.01)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    device = str(args.device)
    loss_weights = SparseResidualLossWeights(
        global_delta=args.global_delta_weight,
        top_de=args.top_de_weight,
        program_gene=args.program_gene_weight,
        program_direction=args.direction_weight,
        program_sign=args.sign_weight,
        outside_sparsity=args.sparsity_weight,
        decorrelation=args.decorrelation_weight,
    )
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=args.rank)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=device,
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
    base_model = experiment_config.build_model().to(device)
    install_prefit_pls_readout(base_model, readout, freeze=True, device=device)
    install_prefit_pls_distillation_head(base_model, readout, freeze=True, device=device)
    _freeze_all_parameters(base_model)
    head = SparsePerturbationResidualHead(
        num_genes=dataset.config.genes,
        num_programs=dataset.config.num_programs,
        program_assignment=tuple(int(value) for value in dataset.gene_program_assignment),
        num_perturbations=dataset.config.num_perturbations,
        num_cell_lines=dataset.config.num_cell_lines,
        hidden_dim=args.hidden_dim,
        z_pert_dim=args.z_pert_dim,
        dictionary_rank=args.dictionary_rank,
        dropout=0.0,
    ).to(device)
    model = SparseResidualBridgeWrapper(base_model, head).to(device)
    initial_readout = _frozen_readout_state(base_model)

    before = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    pairs = _counterfactual_pairs(dataset, split="train")
    if not pairs:
        raise RuntimeError("no train counterfactual pairs were found")
    matching_baseline = _matching_delta_mean_baseline(dataset, train_pairs=pairs, split="test")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = _train_sparse_residual(
        model,
        optimizer,
        dataset,
        pairs,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        seed=args.seed,
        device=device,
        top_de_k=args.top_de_k,
        top_programs=args.top_programs,
        loss_weights=loss_weights,
    )
    after = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    readout_drift = _max_frozen_readout_drift(initial_readout, _frozen_readout_state(base_model))
    protected_deltas = _protected_metric_deltas(before, after)
    counterfactual_deltas = _counterfactual_deltas(before, after)
    after["training_steps_completed"] = float(len(history))
    after["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after["device_used"] = device
    if device.startswith("cuda") and torch.cuda.is_available():
        after["max_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    else:
        after["max_gpu_memory_gb"] = 0.0
    after["frozen_readout_max_abs_drift"] = readout_drift
    after["protected_geometry_preserved"] = bool(readout_drift <= 1e-7 and _protected_deltas_ok(protected_deltas))
    after["counterfactual_gate_pass"] = _counterfactual_gate_pass(before, after, protected_deltas, readout_drift)
    after["sparse_residual_head"] = True
    after["sparse_residual_dictionary_rank"] = float(args.dictionary_rank)
    after["sparse_residual_top_de_k"] = float(args.top_de_k)
    after["sparse_residual_top_programs"] = float(args.top_programs)
    after.update({f"matching_mean_{key}": value for key, value in matching_baseline.items()})
    after.update(_training_diagnostics(history))

    (run_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(run_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, run_dir / "prefit_pls_readout.json")
    _write_json(run_dir / "SPARSE_RESIDUAL_CONFIG.json", _sparse_config(args, loss_weights))
    _write_json(run_dir / "MATCHING_MEAN_BASELINE.json", matching_baseline)
    _write_json(run_dir / "BEFORE_METRICS.json", before)
    _write_json(run_dir / "AFTER_METRICS.json", after)
    _write_json(run_dir / "TRAIN_HISTORY.json", history)
    checkpoint_path = save_checkpoint(
        run_dir / "sparse_perturbation_residual_head.pt",
        model=model,
        optimizer=optimizer,
        trainer_state={"steps": len(history)},
        experiment_config=experiment_config,
        metadata={
            "stage": "sparse_perturbation_residual_head",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "protected_metric_deltas": protected_deltas,
            "counterfactual_metric_deltas": counterfactual_deltas,
            "frozen_readout_max_abs_drift": readout_drift,
            "sparse_residual_config": _sparse_config(args, loss_weights),
            "matching_mean_baseline": matching_baseline,
        },
    )
    _write_report(
        run_dir / "REPORT.md",
        args=args,
        before=before,
        after=after,
        protected_deltas=protected_deltas,
        counterfactual_deltas=counterfactual_deltas,
        matching_baseline=matching_baseline,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after), sort_keys=True))
    return 0


def _train_sparse_residual(
    model: SparseResidualBridgeWrapper,
    optimizer: torch.optim.Optimizer,
    dataset,
    pairs: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    seed: int,
    device: str,
    top_de_k: int,
    top_programs: int,
    loss_weights: SparseResidualLossWeights,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    assignment = torch.as_tensor(dataset.gene_program_assignment, dtype=torch.long, device=device)
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
        control = batch.expression_values.mean(dim=1)
        target_delta = target - control
        predicted_delta = outputs["counterfactual_rna_delta"]
        losses = sparse_delta_losses(
            predicted_delta,
            target_delta,
            z_inv=outputs["z_inv"],
            z_pert=outputs["sparse_z_pert"],
            program_assignment=assignment,
            top_de_k=top_de_k,
            top_programs=top_programs,
            weights=loss_weights,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.sparse_residual_head.parameters(), 1.0)
        optimizer.step()
        program_delta = program_means(predicted_delta, assignment)
        target_program_delta = program_means(target_delta, assignment)
        history.append(
            {
                "step": float(step),
                **{key: float(value.detach().cpu()) for key, value in losses.items()},
                "final_delta_to_target_ratio": _norm_ratio(predicted_delta, target_delta),
                "program_contribution_ratio": _norm_ratio(outputs["sparse_program_gene_delta"], predicted_delta),
                "low_rank_contribution_ratio": _norm_ratio(outputs["sparse_low_rank_delta"], predicted_delta),
                "program_delta_to_target_ratio": _norm_ratio(program_delta, target_program_delta),
                "cap_hit_fraction": 0.0,
            }
        )
    return history


def _matching_delta_mean_baseline(dataset, *, train_pairs: list[dict[str, Any]], split: str) -> dict[str, float]:
    train_delta_by_key: dict[tuple[Any, ...], list[np.ndarray]] = {}
    fallback_delta_by_key: dict[tuple[Any, ...], list[np.ndarray]] = {}
    global_deltas = []
    for pair in train_pairs:
        control = dataset.expression_values[pair["control_group"]].mean(axis=0)
        target = dataset.expression_values[pair["target_group"]].mean(axis=0)
        delta = target - control
        key = (pair["perturbation_id"], pair["cell_line_id"], pair["dose"])
        train_delta_by_key.setdefault(key, []).append(delta)
        fallback_delta_by_key.setdefault((pair["perturbation_id"], pair["dose"]), []).append(delta)
        fallback_delta_by_key.setdefault((pair["perturbation_id"],), []).append(delta)
        global_deltas.append(delta)
    global_delta = np.stack(global_deltas).mean(axis=0)

    predictions = []
    observed = []
    control_values = []
    rows = []
    exact_hits = 0
    test_pairs = _counterfactual_pairs(dataset, split=split)
    for pair in test_pairs:
        control = dataset.expression_values[pair["control_group"]].mean(axis=0)
        target = dataset.expression_values[pair["target_group"]].mean(axis=0)
        key = (pair["perturbation_id"], pair["cell_line_id"], pair["dose"])
        if key in train_delta_by_key:
            delta = np.stack(train_delta_by_key[key]).mean(axis=0)
            exact_hits += 1
        elif (pair["perturbation_id"], pair["dose"]) in fallback_delta_by_key:
            delta = np.stack(fallback_delta_by_key[(pair["perturbation_id"], pair["dose"])]).mean(axis=0)
        elif (pair["perturbation_id"],) in fallback_delta_by_key:
            delta = np.stack(fallback_delta_by_key[(pair["perturbation_id"],)]).mean(axis=0)
        else:
            delta = global_delta
        predictions.append(control + delta)
        observed.append(target)
        control_values.append(control)
        rows.append({key: value for key, value in pair.items() if not key.endswith("_group")})
    if not predictions:
        return {"rows": 0.0, "exact_match_fraction": 0.0}
    predicted = np.asarray(predictions, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    control_array = np.asarray(control_values, dtype=float)
    frame = pd.DataFrame(rows)
    metrics = rna_counterfactual_metrics(
        predicted,
        observed_array,
        control_array,
        frame,
        groupby=None,
        topk=(50,),
        gene_sets=_gene_sets(dataset),
        prefix="rna_counterfactual",
    )
    metrics["program_level_effect_recovery"] = _program_effect_recovery(
        predicted,
        observed_array,
        control_array,
        dataset.gene_program_assignment,
    )
    metrics["rows"] = float(len(predictions))
    metrics["exact_match_fraction"] = float(exact_hits / max(1, len(predictions)))
    return {key: float(value) for key, value in metrics.items()}


def _training_diagnostics(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    keys = (
        "final_delta_to_target_ratio",
        "program_contribution_ratio",
        "low_rank_contribution_ratio",
        "program_delta_to_target_ratio",
        "effect_mask_fraction",
        "top_de_mask_fraction",
        "program_mask_fraction",
        "decorrelation_loss",
        "outside_sparsity_loss",
        "cap_hit_fraction",
    )
    return {f"mean_sparse_{key}": float(np.mean([row.get(key, 0.0) for row in history])) for key in keys}


def _condition_values(expression_values: torch.Tensor, num_genes: int) -> torch.Tensor:
    if expression_values.ndim == 2:
        values = expression_values
    elif expression_values.ndim == 3:
        values = expression_values.mean(dim=1)
    else:
        raise ValueError("expression_values must be 2D or 3D")
    return match_last_dim(values, num_genes)


def _freeze_all_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / (
        f"{args.dataset}_sparsepert_deltaonly_rank{args.dictionary_rank}"
        f"_topde{args.top_de_k}_topprog{args.top_programs}"
        f"_gdw{_slug_float(args.global_delta_weight)}"
        f"_tdw{_slug_float(args.top_de_weight)}"
        f"_pgw{_slug_float(args.program_gene_weight)}"
        f"_dw{_slug_float(args.direction_weight)}"
        f"_sw{_slug_float(args.sign_weight)}"
        f"_spw{_slug_float(args.sparsity_weight)}"
        f"_dcw{_slug_float(args.decorrelation_weight)}"
        f"_lr{_slug_float(args.lr)}_wd{_slug_float(args.weight_decay)}"
        f"_bs{args.batch_size}_seed{args.seed}_pls{args.rank}_s{args.steps}"
    )


def _sparse_config(args: argparse.Namespace, weights: SparseResidualLossWeights) -> dict[str, Any]:
    return {
        "hidden_dim": int(args.hidden_dim),
        "z_pert_dim": int(args.z_pert_dim),
        "dictionary_rank": int(args.dictionary_rank),
        "top_de_k": int(args.top_de_k),
        "top_programs": int(args.top_programs),
        "loss_weights": asdict(weights),
        "metadata_context": "perturbation_id + cell_line_id + dose + time; batch_id excluded",
        "delta_form": "x_cf = x_control + delta_hat",
    }


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before: dict[str, Any],
    after: dict[str, Any],
    protected_deltas: dict[str, float],
    counterfactual_deltas: dict[str, float],
    matching_baseline: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# Sparse Perturbation Residual Head Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- PLS rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Dictionary rank: `{args.dictionary_rank}`",
        f"- Top DE genes: `{args.top_de_k}`",
        f"- Top active programs: `{args.top_programs}`",
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
        "## Matching-Mean Baseline",
        "",
        f"- rows: `{matching_baseline.get('rows', 0.0):.0f}`",
        f"- exact no-batch key coverage: `{matching_baseline.get('exact_match_fraction', 0.0):.4f}`",
        f"- direction accuracy: `{matching_baseline.get('rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
        f"- logFC correlation: `{matching_baseline.get('rna_counterfactual_logfc_correlation', float('nan')):.4f}`",
        f"- top50 overlap: `{matching_baseline.get('rna_counterfactual_top50_de_overlap', float('nan')):.4f}`",
        f"- program recovery: `{matching_baseline.get('program_level_effect_recovery', float('nan')):.4f}`",
        "",
        "## Sparse Head Diagnostics",
        "",
        f"- mean delta/target ratio: `{after.get('mean_sparse_final_delta_to_target_ratio', 0.0):.4f}`",
        f"- mean program contribution ratio: `{after.get('mean_sparse_program_contribution_ratio', 0.0):.4f}`",
        f"- mean low-rank contribution ratio: `{after.get('mean_sparse_low_rank_contribution_ratio', 0.0):.4f}`",
        f"- mean effect mask fraction: `{after.get('mean_sparse_effect_mask_fraction', 0.0):.4f}`",
        f"- mean decorrelation loss: `{after.get('mean_sparse_decorrelation_loss', 0.0):.6f}`",
        f"- mean outside sparsity loss: `{after.get('mean_sparse_outside_sparsity_loss', 0.0):.6f}`",
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
            "- `MATCHING_MEAN_BASELINE.json`",
            "- `SPARSE_RESIDUAL_CONFIG.json`",
            "- `prefit_pls_readout.json`",
            f"- `{checkpoint_path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
