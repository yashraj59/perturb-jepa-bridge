from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import add_gears_simulation_split, load_norman2019_condition_data
from perturb_jepa.evaluation.biotech_metrics import evaluate_biotech_batches
from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.biotech_losses import BioTechJEPALossWeights
from perturb_jepa.training.biotech_trainer import BioTechJEPATrainer
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BioTech-JEPA on synthetic or Norman condition pairs.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite", "synth_micro", "norman"))
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--bio-dim", type=int, default=24)
    parser.add_argument("--tech-dim", type=int, default=8)
    parser.add_argument("--predictor-dim", type=int, default=64)
    parser.add_argument("--num-condition-prototypes", type=int, default=4)
    parser.add_argument("--gene-count", type=int, default=512)
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--transition-jepa-weight", type=float, default=3.0)
    parser.add_argument("--cross-modal-jepa-weight", type=float, default=2.0)
    parser.add_argument("--tech-batch-weight", type=float, default=0.5)
    parser.add_argument("--orthogonality-weight", type=float, default=0.1)
    parser.add_argument("--count-aux-weight", type=float, default=0.0)
    parser.add_argument("--rna-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = _select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_payload = _build_dataset_payload(args, device=device)
    config = _build_config(dataset_payload, args)
    model = BioTechJEPA(config).to(device)
    weights = BioTechJEPALossWeights(
        rna_to_image_jepa=args.cross_modal_jepa_weight,
        image_to_rna_jepa=args.cross_modal_jepa_weight,
        transition_bio_jepa=args.transition_jepa_weight,
        z_tech_batch_ce=args.tech_batch_weight,
        bio_tech_orthogonality=args.orthogonality_weight,
        count_aux=args.count_aux_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    trainer = BioTechJEPATrainer(model, optimizer, weights=weights, device=device)
    train_batches = _train_batches(dataset_payload, args, device=device)
    last: dict[str, float] = {}
    with (args.output_dir / "metrics_train.jsonl").open("w", encoding="utf-8") as handle:
        for result in map(trainer.train_step, train_batches):
            last = result.diagnostics
            handle.write(json.dumps(_jsonable(last), sort_keys=True) + "\n")
    eval_batches = _eval_batches(dataset_payload, args, device=device)
    metrics = evaluate_biotech_batches(model, eval_batches, device=device)
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
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": _jsonable(config),
                "dataset": args.dataset,
                "seed": args.seed,
                "split_seed": args.split_seed,
            },
            args.output_dir / "checkpoint.pt",
        )
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _build_dataset_payload(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    if args.dataset == "norman":
        dataset = add_gears_simulation_split(load_norman2019_condition_data(args.norman_h5ad), seed=args.split_seed)
        spec = build_norman_biotech_spec(dataset, gene_count=args.gene_count)
        return {"kind": "norman", "dataset": dataset, "spec": spec}
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    return {"kind": "synthetic", "dataset": dataset}


def _build_config(payload: dict[str, Any], args: argparse.Namespace) -> BioTechJEPAConfig:
    dim = int(args.shared_dim)
    bio_dim = int(args.bio_dim)
    predictor_dim = int(args.predictor_dim)
    heads = 4 if dim % 4 == 0 else 1
    predictor_heads = 4 if predictor_dim % 4 == 0 else 1
    if payload["kind"] == "norman":
        spec = payload["spec"]
        return BioTechJEPAConfig(
            rna=RNAEncoderConfig(vocab_size=spec.num_genes, dim=dim, depth=1, heads=heads, max_genes=spec.num_genes, pooling="mean_tokens"),
            image=ImageEncoderConfig(in_channels=1, image_size=16, patch_size=4, dim=dim, depth=1, heads=heads, pooling="mean_patches"),
            perturbation=PerturbationEncoderConfig(
                num_perturbations=spec.num_actions,
                num_types=3,
                num_cell_lines=1,
                num_batches=1,
                dim=dim,
                descriptor_dim=spec.descriptor_dim,
                perturbation_feature_mode="feature_only",
            ),
            shared_dim=dim,
            bio_dim=bio_dim,
            tech_dim=int(args.tech_dim),
            predictor_dim=predictor_dim,
            target_query_dim=bio_dim,
            predictor_depth=1,
            predictor_heads=predictor_heads,
            num_condition_prototypes=int(args.num_condition_prototypes),
            num_rna_program_targets=8,
            num_image_region_targets=4,
            count_decoder_aux=args.count_aux_weight > 0.0,
        )
    dataset = payload["dataset"]
    return BioTechJEPAConfig(
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
        ),
        shared_dim=dim,
        bio_dim=bio_dim,
        tech_dim=int(args.tech_dim),
        predictor_dim=predictor_dim,
        target_query_dim=bio_dim,
        predictor_depth=1,
        predictor_heads=predictor_heads,
        num_condition_prototypes=int(args.num_condition_prototypes),
        num_rna_program_targets=min(8, dataset.config.num_programs),
        num_image_region_targets=4,
        gene_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment),
        count_decoder_aux=args.count_aux_weight > 0.0,
    )


def _train_batches(payload: dict[str, Any], args: argparse.Namespace, *, device: torch.device) -> Iterable:
    if payload["kind"] == "norman":
        return iter_norman_biotech_condition_batches(
            payload["dataset"],
            payload["spec"],
            split="train",
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            device=device,
        )
    return iter_bioaction_condition_batches(
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


def _eval_batches(payload: dict[str, Any], args: argparse.Namespace, *, device: torch.device) -> Iterable:
    if payload["kind"] == "norman":
        return iter_norman_biotech_condition_batches(
            payload["dataset"],
            payload["spec"],
            split=args.eval_split,
            batch_size=max(4, args.batch_size),
            steps=args.eval_steps,
            seed=args.seed + 1000,
            device=device,
        )
    return iter_bioaction_condition_batches(
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


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def write_identity_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# BioTech-JEPA Identity Report",
        "",
        f"- Is this a real JEPA path? `{'yes' if metrics.get('encoder_path_used') == 1.0 and metrics.get('pls_raw_linear_main_path_used') == 0.0 else 'no'}`",
        "- Online/context encoders: `yes`",
        "- EMA target encoders: `yes`",
        f"- Stop-gradient targets verified: `{'yes' if metrics.get('teacher_stop_gradient_verified') else 'no'}`",
        f"- Separate z_bio and z_tech branches: `{'yes' if metrics.get('separate_bio_and_tech_latents_present') else 'no'}`",
        f"- Exact condition-key features used: `{'yes' if metrics.get('condition_key_feature_present') else 'no'}`",
        "- PLS use: not used in the BioTech-JEPA representation path",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], device: str) -> None:
    lines = [
        "# BioTech-JEPA Model Card",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Device: `{device}`",
        f"- Steps: `{args.steps}`",
        "- Model role: diagnostic Tier 1 candidate only; protected PLS remains model of record",
        "- Latent contract: `z_bio` is used for JEPA retrieval/cross-modal/transition losses; `z_tech` is used for technical batch allocation when batch labels exist.",
        "- Norman metadata amendment: batch and chemical dose are ignored for Norman because the processed h5ad does not expose batch and `dose_val` is guide-count notation.",
        f"- Transition cosine improvement: `{float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        f"- RNA->image recall@1: `{float(metrics.get('rna_to_image_recall@1', float('nan'))):.4f}`",
        f"- RNA-only diagnostic: `{float(metrics.get('rna_only_diagnostic', 0.0)):.0f}`",
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
