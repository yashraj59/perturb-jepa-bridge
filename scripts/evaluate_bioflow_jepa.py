from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.bioflow_metrics import evaluate_bioflow_batches
from perturb_jepa.models.bioflow_jepa import BioFlowJEPA, BioFlowJEPAConfig
from perturb_jepa.models.biotech_jepa import BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch, iter_bioaction_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved BioFlow-JEPA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite",))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    config = _bioflow_config_from_payload(payload["config"])
    model = BioFlowJEPA(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    descriptor_matrix = synthetic_action_descriptor_matrix(dataset, synthetic_action_descriptor_spec(dataset))
    batches = _attach_descriptors(
        iter_bioaction_condition_batches(
            dataset,
            split=args.eval_split,
            batch_size=args.batch_size,
            bag_size=args.bag_size,
            steps=args.eval_steps,
            seed=args.seed + 1000,
            device=device,
        ),
        descriptor_matrix,
    )
    metrics = evaluate_bioflow_batches(model, batches, device=device)
    metrics["dataset"] = args.dataset  # type: ignore[assignment]
    metrics["eval_split"] = args.eval_split  # type: ignore[assignment]
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return 0


def _attach_descriptors(
    batches: Iterable[BioActionConditionBatch],
    descriptor_matrix: np.ndarray,
) -> Iterable[BioActionConditionBatch]:
    for batch in batches:
        batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, descriptor_matrix)
        yield batch


def _bioflow_config_from_payload(payload: dict[str, Any]) -> BioFlowJEPAConfig:
    base = _biotech_config_from_payload(payload["base_biotech_config"])
    names = {field.name for field in fields(BioFlowJEPAConfig)}
    values = {key: value for key, value in payload.items() if key in names and key != "base_biotech_config"}
    return BioFlowJEPAConfig(base_biotech_config=base, **values)


def _biotech_config_from_payload(payload: dict[str, Any]) -> BioTechJEPAConfig:
    return BioTechJEPAConfig(
        rna=RNAEncoderConfig(**_dataclass_kwargs(RNAEncoderConfig, payload["rna"])),
        image=ImageEncoderConfig(**_dataclass_kwargs(ImageEncoderConfig, payload["image"])),
        perturbation=PerturbationEncoderConfig(**_dataclass_kwargs(PerturbationEncoderConfig, payload["perturbation"])),
        **{
            field.name: payload[field.name]
            for field in fields(BioTechJEPAConfig)
            if field.name in payload and field.name not in {"rna", "image", "perturbation"}
        },
    )


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


if __name__ == "__main__":
    raise SystemExit(main())
