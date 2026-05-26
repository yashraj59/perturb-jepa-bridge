from __future__ import annotations

import argparse
from dataclasses import fields
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
from perturb_jepa.training.bioaction_batches import exact_train_key_fraction, iter_bioaction_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.evaluation.bioaction_metrics import evaluate_bioaction_batches
from scripts.train_bioaction_jepa import build_config, write_identity_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BioAction-JEPA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = _config_from_payload(checkpoint.get("config"), dataset)
    model = BioActionJEPA(config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    batches = iter_bioaction_condition_batches(
        dataset,
        split=args.eval_split,
        batch_size=8,
        steps=4,
        seed=args.seed + 2000,
        device=args.device,
    )
    metrics = evaluate_bioaction_batches(model, batches, device=args.device)
    metrics["exact_match_fraction"] = exact_train_key_fraction(dataset, eval_split=args.eval_split)
    metrics["source_as_target_transition_null_available"] = 1.0
    metrics["global_mean_transition_null_available"] = 1.0
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "retrieval_metrics.json", {key: value for key, value in metrics.items() if "recall" in key or "map" in key})
    write_json(args.output_dir / "transition_metrics.json", {key: value for key, value in metrics.items() if "transition" in key or "teacher/" in key})
    write_json(args.output_dir / "biology_metrics.json", {"exact_match_fraction": metrics["exact_match_fraction"]})
    write_json(args.output_dir / "morphology_metrics.json", {key: value for key, value in metrics.items() if "image" in key})
    write_json(args.output_dir / "collapse_diagnostics.json", {key: value for key, value in metrics.items() if key.startswith("latent/") or key.startswith("teacher/")})
    write_identity_report(args.output_dir / "jepa_identity_report.md", metrics)
    write_metrics_md(args.output_dir / "metrics.md", metrics)
    np.savez_compressed(args.output_dir / "prediction_arrays.npz", placeholder=np.asarray([0.0], dtype=np.float32))
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _config_from_payload(payload: Any, dataset) -> BioActionJEPAConfig:
    if not isinstance(payload, dict):
        return build_config(dataset, argparse.Namespace(shared_dim=64, predictor_dim=128, num_state_prototypes=4, disable_count_aux=False))
    def dataclass_kwargs(cls, data):
        return {field.name: data[field.name] for field in fields(cls) if field.name in data}
    return BioActionJEPAConfig(
        rna=RNAEncoderConfig(**dataclass_kwargs(RNAEncoderConfig, payload["rna"])),
        image=ImageEncoderConfig(**dataclass_kwargs(ImageEncoderConfig, payload["image"])),
        perturbation=PerturbationEncoderConfig(**dataclass_kwargs(PerturbationEncoderConfig, payload["perturbation"])),
        **{field.name: payload[field.name] for field in fields(BioActionJEPAConfig) if field.name in payload and field.name not in {"rna", "image", "perturbation"}},
    )


def write_metrics_md(path: Path, metrics: dict[str, float]) -> None:
    lines = ["# BioAction-JEPA Metrics", ""]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: `{value:.6g}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
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
