from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import add_gears_simulation_split, load_norman2019_condition_data
from perturb_jepa.evaluation.biomech_metrics import evaluate_biomech_batches
from perturb_jepa.models.biomech_jepa import BioMechanisticJEPA, BioMechanisticJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BioMechanistic-JEPA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite", "synth_micro", "norman"))
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--gene-count", type=int, default=256)
    parser.add_argument("--rna-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    config = _config_from_payload(payload["config"])
    model = BioMechanisticJEPA(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    metrics = evaluate_biomech_batches(model, _eval_batches(args, config, device=device), device=device)
    metrics["dataset"] = args.dataset  # type: ignore[assignment]
    metrics["eval_split"] = args.eval_split  # type: ignore[assignment]
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _eval_batches(args: argparse.Namespace, config: BioMechanisticJEPAConfig, *, device: torch.device):
    if args.dataset == "norman":
        dataset = add_gears_simulation_split(load_norman2019_condition_data(args.norman_h5ad), seed=args.split_seed)
        spec = build_norman_biotech_spec(dataset, gene_count=args.gene_count)
        yield from iter_norman_biotech_condition_batches(dataset, spec, split=args.eval_split, batch_size=args.batch_size, steps=args.eval_steps, seed=args.seed + 1000, device=device)
        return
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    batches = iter_bioaction_condition_batches(
        dataset,
        split=args.eval_split,
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        steps=args.eval_steps,
        seed=args.seed + 1000,
        device=device,
        rna_only=args.rna_only,
        paired=not args.rna_only,
    )
    if not config.enable_program_action_encoder:
        yield from batches
        return
    matrix = synthetic_action_descriptor_matrix(dataset, synthetic_action_descriptor_spec(dataset))
    for batch in batches:
        batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, matrix)
        yield batch


def _config_from_payload(payload: dict[str, Any]) -> BioMechanisticJEPAConfig:
    return BioMechanisticJEPAConfig(
        rna=RNAEncoderConfig(**_dataclass_kwargs(RNAEncoderConfig, payload["rna"])),
        image=ImageEncoderConfig(**_dataclass_kwargs(ImageEncoderConfig, payload["image"])),
        perturbation=PerturbationEncoderConfig(**_dataclass_kwargs(PerturbationEncoderConfig, payload["perturbation"])),
        **{
            field.name: payload[field.name]
            for field in fields(BioMechanisticJEPAConfig)
            if field.name in payload and field.name not in {"rna", "image", "perturbation"}
        },
    )


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


if __name__ == "__main__":
    raise SystemExit(main())
