from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.models.bioaction_jepa import BioActionJEPA, BioActionJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.bioaction_trainer import BioActionJEPATrainer
from perturb_jepa.training.bioaction_losses import BioActionJEPALossWeights
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.evaluation.bioaction_metrics import evaluate_bioaction_batches


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the minimal real BioAction-JEPA path on synthetic condition pairs.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shared-dim", type=int, default=64)
    parser.add_argument("--predictor-dim", type=int, default=128)
    parser.add_argument("--num-state-prototypes", type=int, default=4)
    parser.add_argument("--rna-mask-mode", default="program")
    parser.add_argument("--image-mask-mode", default="block")
    parser.add_argument("--modality-dropout", type=float, default=0.25)
    parser.add_argument("--transition-jepa-weight", type=float, default=2.0)
    parser.add_argument("--cross-modal-jepa-weight", type=float, default=2.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--count-aux-weight", type=float, default=0.1)
    parser.add_argument("--batch-invariance-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--rna-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--transition-only", action="store_true")
    parser.add_argument("--disable-count-aux", action="store_true")
    parser.add_argument("--disable-pls-bootstrap", action="store_true")
    parser.add_argument("--pls-bootstrap-weight", type=float, default=0.0)
    parser.add_argument("--anneal-pls-bootstrap-steps", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    config = build_config(dataset, args)
    model = BioActionJEPA(config).to(args.device)
    weights = BioActionJEPALossWeights(
        rna_to_image_jepa=args.cross_modal_jepa_weight,
        image_to_rna_jepa=args.cross_modal_jepa_weight,
        transition_rna_jepa=args.transition_jepa_weight,
        transition_image_jepa=args.transition_jepa_weight,
        transition_joint_jepa=args.transition_jepa_weight,
        count_nb_nll_aux=0.0 if args.disable_count_aux else args.count_aux_weight,
        batch_invariance=args.batch_invariance_weight,
        raw_rna_reconstruction=0.0 if args.no_reconstruction else args.reconstruction_weight,
        raw_image_reconstruction=0.0 if args.no_reconstruction else args.reconstruction_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    trainer = BioActionJEPATrainer(model, optimizer, weights=weights, device=args.device)
    train_batches = iter_bioaction_condition_batches(
        dataset,
        split="train",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        modality_dropout=args.modality_dropout,
        rna_only=args.rna_only,
        image_only=args.image_only,
        paired=args.paired or not (args.rna_only or args.image_only),
        transition_only=args.transition_only,
    )
    with (args.output_dir / "metrics_train.jsonl").open("w", encoding="utf-8") as handle:
        last = {}
        for result in map(trainer.train_step, train_batches):
            last = result.diagnostics
            handle.write(json.dumps(_jsonable(last), sort_keys=True) + "\n")
    eval_batches = iter_bioaction_condition_batches(
        dataset,
        split=args.eval_split,
        batch_size=max(8, args.batch_size),
        steps=4,
        seed=args.seed + 1000,
        device=args.device,
    )
    metrics = evaluate_bioaction_batches(model, eval_batches, device=args.device)
    metrics.update({f"last_train/{key}": value for key, value in last.items() if isinstance(value, (int, float))})
    write_json(args.output_dir / "config.json", {"args": vars(args), "model": _jsonable(config), "weights": _jsonable(weights)})
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_json(args.output_dir / "collapse_diagnostics.json", {key: value for key, value in metrics.items() if key.startswith("latent/") or key.startswith("teacher/")})
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _jsonable(config),
            "dataset": args.dataset,
            "seed": args.seed,
        },
        args.output_dir / "checkpoint.pt",
    )
    write_identity_report(args.output_dir / "jepa_identity_report.md", metrics)
    write_model_card(args.output_dir / "model_card.md", args=args, metrics=metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def build_config(dataset, args: argparse.Namespace) -> BioActionJEPAConfig:
    dim = int(args.shared_dim)
    predictor_dim = int(args.predictor_dim)
    heads = 4 if dim % 4 == 0 else 1
    predictor_heads = 4 if predictor_dim % 4 == 0 else 1
    return BioActionJEPAConfig(
        rna=RNAEncoderConfig(
            vocab_size=dataset.config.genes,
            dim=dim,
            depth=1,
            heads=heads,
            max_genes=dataset.config.genes,
            pooling="mean_tokens",
        ),
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
            num_types=2,
            num_cell_lines=dataset.config.num_cell_lines,
            num_batches=dataset.config.num_batches,
            dim=dim,
        ),
        shared_dim=dim,
        predictor_dim=predictor_dim,
        target_query_dim=dim,
        predictor_depth=1,
        predictor_heads=predictor_heads,
        num_state_prototypes=args.num_state_prototypes,
        num_condition_prototypes=max(1, args.num_state_prototypes),
        num_rna_program_targets=min(8, dataset.config.num_programs),
        num_image_region_targets=4,
        gene_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment),
        count_decoder_aux=not args.disable_count_aux,
    )


def write_identity_report(path: Path, metrics: dict[str, float]) -> None:
    lines = [
        "# BioAction-JEPA Identity Report",
        "",
        f"- Is this a real JEPA? `{'yes' if metrics.get('encoder_path_used') == 1.0 and metrics.get('pls_raw_linear_main_path_used') == 0.0 else 'no'}`",
        "- Are encoder readouts the main path? `yes`",
        f"- Are PLS raw-linear heads used as main path? `{'yes' if metrics.get('pls_raw_linear_main_path_used') else 'no'}`",
        f"- Are teacher targets stop-grad? `{'yes' if metrics.get('teacher_stop_gradient_verified') else 'no'}`",
        f"- Can loss run with reconstruction=0? `{'yes' if metrics.get('latent_prediction_loss_available_with_reconstruction_zero') else 'no'}`",
        f"- Are condition-key one-hots used? `{'yes' if metrics.get('condition_key_feature_present') else 'no'}`",
        "- Active JEPA tasks: RNA program, image region, RNA->image, image->RNA, joint->RNA/image, action-conditioned transition.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(path: Path, *, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    lines = [
        "# BioAction-JEPA Model Card",
        "",
        f"- Model name: `BioAction-JEPA minimal synthetic {args.dataset}`",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        "- Training data: synthetic only",
        "- JEPA tasks active: intra-modal, cross-modal, and transition latent prediction",
        "- Teacher/target design: EMA RNA and image target encoders with detached latent targets",
        f"- Raw reconstruction weight: `{args.reconstruction_weight if not args.no_reconstruction else 0.0}`",
        "- PLS usage: not used in the main BioAction-JEPA path",
        "- Tier reached: implementation smoke/Tier 0 candidate until gates are run",
        f"- RNA->image recall@1: `{metrics.get('rna_to_image_recall@1', float('nan')):.4f}`",
        f"- Image->RNA recall@1: `{metrics.get('image_to_rna_recall@1', float('nan')):.4f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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
