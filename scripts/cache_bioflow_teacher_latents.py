from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import add_gears_simulation_split, load_norman2019_condition_data
from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch, iter_bioaction_condition_batches
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache BioTech teacher/source/target latents for Phase 4 delta-operator audit.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite", "norman"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt"))
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--gene-count", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_biotech_model(args.checkpoint, device=device)
    if args.dataset == "norman":
        dataset = add_gears_simulation_split(load_norman2019_condition_data(args.norman_h5ad), seed=args.split_seed)
        spec = build_norman_biotech_spec(dataset, gene_count=args.gene_count)
        for split in ("train", args.eval_split):
            rows = collect_latents(
                model,
                iter_norman_biotech_condition_batches(dataset, spec, split=split, batch_size=args.batch_size, steps=None, seed=args.seed + (0 if split == "train" else 1000), device=device),
                device=device,
            )
            write_cache(args.output_dir / f"{args.dataset}_{split}_latents", rows)
    else:
        dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
        descriptor_matrix = synthetic_action_descriptor_matrix(dataset, synthetic_action_descriptor_spec(dataset))
        for split in ("train", args.eval_split):
            batches = iter_bioaction_condition_batches(
                dataset,
                split=split,
                batch_size=args.batch_size,
                bag_size=args.bag_size,
                steps=None,
                seed=args.seed + (0 if split == "train" else 1000),
                device=device,
            )
            rows = collect_latents(model, _attach_synthetic_descriptors(batches, descriptor_matrix), device=device)
            write_cache(args.output_dir / f"{args.dataset}_{split}_latents", rows)
    return 0


def load_biotech_model(checkpoint: Path, *, device: torch.device) -> BioTechJEPA:
    payload = torch.load(checkpoint, map_location=device)
    config = _config_from_payload(payload["config"])
    model = BioTechJEPA(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def collect_latents(
    model: BioTechJEPA,
    batches: Iterable[BioActionConditionBatch],
    *,
    device: torch.device,
) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {
        "source_z_bio_teacher": [],
        "source_z_bio_online": [],
        "target_z_bio_teacher": [],
        "target_z_bio_online": [],
        "source_z_tech_teacher": [],
        "target_z_tech_teacher": [],
        "action_descriptor": [],
        "perturbation_id": [],
        "cell_line_id": [],
        "batch_id": [],
    }
    labels: dict[str, list[str]] = {"condition_key": [], "split": []}
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            source_teacher = model.encode_condition(
                gene_ids=batch.control_gene_ids,
                expression_values=batch.control_expression_values,
                images=batch.control_images,
                rna_bag_mask=batch.control_rna_bag_mask,
                image_bag_mask=batch.control_image_bag_mask,
                mode="target",
            )
            source_online = model.encode_condition(
                gene_ids=batch.control_gene_ids,
                expression_values=batch.control_expression_values,
                images=batch.control_images,
                rna_bag_mask=batch.control_rna_bag_mask,
                image_bag_mask=batch.control_image_bag_mask,
                mode="context",
            )
            target_teacher = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="target",
            )
            target_online = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="context",
            )
            arrays["source_z_bio_teacher"].append(_cpu(source_teacher["joint_z_bio"]))
            arrays["source_z_bio_online"].append(_cpu(source_online["joint_z_bio"]))
            arrays["target_z_bio_teacher"].append(_cpu(target_teacher["joint_z_bio"]))
            arrays["target_z_bio_online"].append(_cpu(target_online["joint_z_bio"]))
            arrays["source_z_tech_teacher"].append(_cpu(source_teacher["joint_z_tech"]))
            arrays["target_z_tech_teacher"].append(_cpu(target_teacher["joint_z_tech"]))
            descriptor = batch.descriptor if batch.descriptor is not None else torch.zeros(batch.perturbation_id.shape[0], 0, device=device)
            arrays["action_descriptor"].append(_cpu(descriptor))
            arrays["perturbation_id"].append(_cpu(batch.perturbation_id).reshape(-1, 1))
            arrays["cell_line_id"].append(_cpu(batch.cell_line_id).reshape(-1, 1))
            arrays["batch_id"].append(_cpu(batch.batch_id).reshape(-1, 1))
            labels["condition_key"].extend(batch.condition_key or [str(index) for index in range(batch.perturbation_id.numel())])
            labels["split"].extend(batch.split or ["unknown" for _ in range(batch.perturbation_id.numel())])
    return {
        **{key: np.concatenate(value, axis=0) if value else np.empty((0, 0), dtype=np.float32) for key, value in arrays.items()},
        **labels,
    }


def write_cache(prefix: Path, rows: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in rows.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    metadata = pd.DataFrame(
        {
            "condition_key": rows["condition_key"],
            "split": rows["split"],
            "perturbation_id": arrays["perturbation_id"].reshape(-1).astype(int),
            "cell_line_id": arrays["cell_line_id"].reshape(-1).astype(int),
            "batch_id": arrays["batch_id"].reshape(-1).astype(int),
        }
    )
    metadata.to_csv(prefix.with_suffix(".metadata.tsv"), sep="\t", index=False)


def _attach_synthetic_descriptors(
    batches: Iterable[BioActionConditionBatch],
    descriptor_matrix: np.ndarray,
) -> Iterable[BioActionConditionBatch]:
    for batch in batches:
        batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, descriptor_matrix)
        yield batch


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _config_from_payload(payload: dict[str, Any]) -> BioTechJEPAConfig:
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
