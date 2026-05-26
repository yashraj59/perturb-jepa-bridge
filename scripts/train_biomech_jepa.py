from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import add_gears_simulation_split, load_norman2019_condition_data
from perturb_jepa.evaluation.biomech_metrics import evaluate_biomech_batches
from perturb_jepa.models.biomech_jepa import BioMechanisticJEPA, BioMechanisticJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch, iter_bioaction_condition_batches
from perturb_jepa.training.biomech_losses import BioMechanisticJEPALossWeights
from perturb_jepa.training.biomech_trainer import BioMechanisticJEPATrainer
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BioMechanistic-JEPA under Phase 3 gates.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite", "synth_micro", "norman"))
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--bio-dim", type=int, default=24)
    parser.add_argument("--tech-dim", type=int, default=8)
    parser.add_argument("--predictor-dim", type=int, default=64)
    parser.add_argument("--num-condition-prototypes", type=int, default=4)
    parser.add_argument("--gene-count", type=int, default=256)
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--enable-delta-jepa", action="store_true")
    parser.add_argument("--enable-program-action-encoder", action="store_true")
    parser.add_argument("--disable-perturbation-id-embedding-for-heldout-generalization", action="store_true")
    parser.add_argument("--enable-population-transition", action="store_true")
    parser.add_argument("--prototype-set-loss", default="none")
    parser.add_argument("--enable-cross-modal-repair", action="store_true")
    parser.add_argument("--modality-balanced-loss", action="store_true")
    parser.add_argument("--rna-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _dataset_payload(args)
    config = _build_config(payload, args)
    model = BioMechanisticJEPA(config).to(device)
    weights = _loss_weights(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    trainer = BioMechanisticJEPATrainer(model, optimizer, weights=weights, device=device)
    last: dict[str, float] = {}
    with (args.output_dir / "metrics_train.jsonl").open("w", encoding="utf-8") as handle:
        for result in map(trainer.train_step, _train_batches(payload, args, device=device)):
            last = result.diagnostics
            handle.write(json.dumps(_jsonable(last), sort_keys=True) + "\n")
    metrics = evaluate_biomech_batches(model, _eval_batches(payload, args, device=device), device=device)
    metrics.update({f"last_train/{key}": value for key, value in last.items() if isinstance(value, (int, float))})
    metrics["dataset"] = args.dataset  # type: ignore[assignment]
    metrics["eval_split"] = args.eval_split  # type: ignore[assignment]
    metrics["seed"] = float(args.seed)
    metrics["device_used_cuda"] = float(str(device).startswith("cuda"))
    write_json(args.output_dir / "config.json", {"args": vars(args), "device": str(device), "model": _jsonable(config), "weights": _jsonable(weights)})
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_identity_report(args.output_dir / "jepa_identity_report.md", metrics)
    write_model_card(args.output_dir / "model_card.md", args=args, metrics=metrics, device=str(device))
    if args.save_checkpoint:
        torch.save({"model_state_dict": model.state_dict(), "config": _jsonable(config), "dataset": args.dataset, "seed": args.seed}, args.output_dir / "checkpoint.pt")
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _dataset_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "norman":
        dataset = add_gears_simulation_split(load_norman2019_condition_data(args.norman_h5ad), seed=args.split_seed)
        spec = build_norman_biotech_spec(dataset, gene_count=args.gene_count)
        return {"kind": "norman", "dataset": dataset, "spec": spec, "descriptor_gene_dim": spec.descriptor_dim, "descriptor_program_dim": 0}
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    spec = synthetic_action_descriptor_spec(dataset)
    matrix = synthetic_action_descriptor_matrix(dataset, spec)
    return {
        "kind": "synthetic",
        "dataset": dataset,
        "descriptor_matrix": matrix,
        "descriptor_gene_dim": spec.gene_dim,
        "descriptor_program_dim": spec.program_dim,
    }


def _build_config(payload: dict[str, Any], args: argparse.Namespace) -> BioMechanisticJEPAConfig:
    dim = int(args.shared_dim)
    bio_dim = int(args.bio_dim)
    heads = 4 if dim % 4 == 0 else 1
    predictor_heads = 4 if int(args.predictor_dim) % 4 == 0 else 1
    descriptor_dim = int(payload["descriptor_gene_dim"]) + int(payload["descriptor_program_dim"]) if args.enable_program_action_encoder else 0
    feature_mode = "lookup"
    if args.enable_program_action_encoder and args.disable_perturbation_id_embedding_for_heldout_generalization:
        feature_mode = "feature_only"
    elif args.enable_program_action_encoder:
        feature_mode = "residual"
    if payload["kind"] == "norman":
        spec = payload["spec"]
        descriptor_dim = spec.descriptor_dim
        feature_mode = "feature_only"
        return BioMechanisticJEPAConfig(
            rna=RNAEncoderConfig(vocab_size=spec.num_genes, dim=dim, depth=1, heads=heads, max_genes=spec.num_genes, pooling="mean_tokens"),
            image=ImageEncoderConfig(in_channels=1, image_size=16, patch_size=4, dim=dim, depth=1, heads=heads, pooling="mean_patches"),
            perturbation=PerturbationEncoderConfig(
                num_perturbations=spec.num_actions,
                num_types=3,
                num_cell_lines=1,
                num_batches=1,
                dim=dim,
                descriptor_dim=descriptor_dim,
                perturbation_feature_mode=feature_mode,
            ),
            shared_dim=dim,
            bio_dim=bio_dim,
            tech_dim=int(args.tech_dim),
            predictor_dim=int(args.predictor_dim),
            target_query_dim=bio_dim,
            predictor_heads=predictor_heads,
            num_condition_prototypes=int(args.num_condition_prototypes),
            enable_delta_jepa=args.enable_delta_jepa,
            enable_program_action_encoder=args.enable_program_action_encoder,
            enable_population_transition=args.enable_population_transition,
            enable_cross_modal_repair=args.enable_cross_modal_repair,
            descriptor_gene_dim=spec.descriptor_dim,
            descriptor_program_dim=0,
            action_dim=bio_dim,
            disable_perturbation_id_embedding_for_heldout_generalization=args.disable_perturbation_id_embedding_for_heldout_generalization,
        )
    dataset = payload["dataset"]
    return BioMechanisticJEPAConfig(
        rna=RNAEncoderConfig(vocab_size=dataset.config.genes, dim=dim, depth=1, heads=heads, max_genes=dataset.config.genes, pooling="mean_tokens"),
        image=ImageEncoderConfig(
            in_channels=dataset.config.image_channels,
            image_size=dataset.config.image_size,
            patch_size=dataset.config.patch_size,
            dim=dim,
            depth=1,
            heads=heads,
            pooling="mean_patches",
        ),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=dataset.config.num_perturbations,
            num_types=3,
            num_cell_lines=dataset.config.num_cell_lines,
            num_batches=dataset.config.num_batches,
            dim=dim,
            descriptor_dim=descriptor_dim,
            perturbation_feature_mode=feature_mode,
        ),
        shared_dim=dim,
        bio_dim=bio_dim,
        tech_dim=int(args.tech_dim),
        predictor_dim=int(args.predictor_dim),
        target_query_dim=bio_dim,
        predictor_heads=predictor_heads,
        num_condition_prototypes=int(args.num_condition_prototypes),
        num_rna_program_targets=min(8, dataset.config.num_programs),
        gene_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment),
        enable_delta_jepa=args.enable_delta_jepa,
        enable_program_action_encoder=args.enable_program_action_encoder,
        enable_population_transition=args.enable_population_transition,
        enable_cross_modal_repair=args.enable_cross_modal_repair,
        descriptor_gene_dim=int(payload["descriptor_gene_dim"]) if args.enable_program_action_encoder else 0,
        descriptor_program_dim=int(payload["descriptor_program_dim"]) if args.enable_program_action_encoder else 0,
        action_dim=bio_dim,
        disable_perturbation_id_embedding_for_heldout_generalization=args.disable_perturbation_id_embedding_for_heldout_generalization,
    )


def _loss_weights(args: argparse.Namespace) -> BioMechanisticJEPALossWeights:
    cross = 2.0 if args.enable_cross_modal_repair else 1.5
    population = 1.0 if args.enable_population_transition else 0.0
    return BioMechanisticJEPALossWeights(
        rna_to_image_jepa=cross,
        image_to_rna_jepa=cross,
        prototype_transition_jepa=population,
        prototype_set=0.5 if args.enable_population_transition else 0.0,
    )


def _train_batches(payload: dict[str, Any], args: argparse.Namespace, *, device: torch.device) -> Iterable[BioActionConditionBatch]:
    if payload["kind"] == "norman":
        return iter_norman_biotech_condition_batches(payload["dataset"], payload["spec"], split="train", batch_size=args.batch_size, steps=args.steps, seed=args.seed, device=device)
    batches = iter_bioaction_condition_batches(
        payload["dataset"],
        split="train",
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        steps=args.steps,
        seed=args.seed,
        device=device,
        modality_dropout=args.modality_dropout,
        rna_only=args.rna_only,
        paired=not args.rna_only,
    )
    return _maybe_attach_descriptors(batches, payload, enabled=args.enable_program_action_encoder)


def _eval_batches(payload: dict[str, Any], args: argparse.Namespace, *, device: torch.device) -> Iterable[BioActionConditionBatch]:
    if payload["kind"] == "norman":
        return iter_norman_biotech_condition_batches(payload["dataset"], payload["spec"], split=args.eval_split, batch_size=max(4, args.batch_size), steps=args.eval_steps, seed=args.seed + 1000, device=device)
    batches = iter_bioaction_condition_batches(
        payload["dataset"],
        split=args.eval_split,
        batch_size=max(4, args.batch_size),
        bag_size=args.bag_size,
        steps=args.eval_steps,
        seed=args.seed + 1000,
        device=device,
        rna_only=args.rna_only,
        paired=not args.rna_only,
    )
    return _maybe_attach_descriptors(batches, payload, enabled=args.enable_program_action_encoder)


def _maybe_attach_descriptors(batches: Iterable[BioActionConditionBatch], payload: dict[str, Any], *, enabled: bool) -> Iterable[BioActionConditionBatch]:
    for batch in batches:
        if enabled:
            batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, payload["descriptor_matrix"])
        yield batch


def write_identity_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# BioMechanistic-JEPA Identity Report",
        "",
        f"- Real JEPA path: `{'yes' if metrics.get('encoder_path_used') == 1.0 and metrics.get('pls_raw_linear_main_path_used') == 0.0 else 'no'}`",
        f"- Stop-gradient targets verified: `{'yes' if metrics.get('teacher_stop_gradient_verified') else 'no'}`",
        f"- Separate z_bio/z_tech: `{'yes' if metrics.get('separate_bio_and_tech_latents_present') else 'no'}`",
        f"- Condition-key feature present: `{'yes' if metrics.get('condition_key_feature_present') else 'no'}`",
        f"- Held-out action descriptor valid: `{'yes' if metrics.get('heldout_action_descriptor_valid') else 'no'}`",
        "- PLS use: not used as the main representation path.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], device: str) -> None:
    lines = [
        "# BioMechanistic-JEPA Model Card",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Device: `{device}`",
        "- Role: Phase 3 diagnostic candidate; protected PLS remains model of record.",
        f"- Delta JEPA: `{args.enable_delta_jepa}`",
        f"- Program action encoder: `{args.enable_program_action_encoder}`",
        f"- Population transition: `{args.enable_population_transition}`",
        f"- Cross-modal repair: `{args.enable_cross_modal_repair}`",
        f"- Transition gain: `{float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        f"- Image->RNA recall@1: `{float(metrics.get('image_to_rna_recall@1', float('nan'))):.4f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, argparse.Namespace):
        return _jsonable(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist())
    return value


if __name__ == "__main__":
    raise SystemExit(main())
