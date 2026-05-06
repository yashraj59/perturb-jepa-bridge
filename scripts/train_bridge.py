from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import forward_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the cross-modal Perturb-JEPA bridge.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic batches.")
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if not args.synthetic:
        print("No real-data bridge loader is configured yet; running synthetic scaffold. Pass --synthetic to make this explicit.")
    _run_synthetic_training(config, stage="bridge", checkpoint_out=args.checkpoint_out)
    return 0


def _run_synthetic_training(
    config: ExperimentConfig,
    *,
    stage: str,
    checkpoint_out: Path | None,
) -> None:
    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    model.to(device)
    for step in range(config.training.steps):
        batch = make_synthetic_bridge_batch(**asdict(config.synthetic), device=device)
        terms = _synthetic_stage_step(
            model,
            optimizer,
            batch,
            grad_clip_norm=config.training.grad_clip_norm,
            ema_decay=config.training.ema_decay,
        )
        if config.training.log_every > 0 and (step + 1) % config.training.log_every == 0:
            print(f"stage={stage} step={step} " + _format_terms(terms))
    if checkpoint_out is not None:
        save_checkpoint(
            checkpoint_out,
            model=model,
            optimizer=optimizer,
            trainer_state={"global_step": config.training.steps},
            experiment_config=config,
            metadata={"stage": stage},
        )


def _synthetic_stage_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    *,
    grad_clip_norm: float | None,
    ema_decay: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_batch(model, batch)
    terms: dict[str, torch.Tensor] = {}
    if "rna_reconstruction" in outputs:
        terms["rna_mask"] = torch.nn.functional.mse_loss(outputs["rna_reconstruction"], batch.expression_values)
    if "image_patch_reconstruction" in outputs:
        prediction = outputs["image_patch_reconstruction"]
        target = patchify(batch.images, model.config.image.patch_size)
        if prediction.ndim == target.ndim + 1 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)
        terms["image_mask"] = torch.nn.functional.mse_loss(prediction, target)
    if "rna_shared" in outputs and "image_shared" in outputs:
        terms["align"] = 1.0 - torch.nn.functional.cosine_similarity(
            outputs["rna_shared"],
            outputs["image_shared"],
            dim=-1,
        ).mean()
    if not terms:
        raise RuntimeError("model forward pass produced no synthetic training terms")
    total = sum(terms.values())
    total.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    if hasattr(model, "update_teachers"):
        model.update_teachers(decay=ema_decay)
    terms["total"] = total.detach()
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def _load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig.smoke()
    if path.suffix.lower() == ".json":
        return ExperimentConfig.load_json(path)
    data = _load_yaml(path)
    return ExperimentConfig.from_dict(data)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _format_terms(terms: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(terms.items()))


if __name__ == "__main__":
    raise SystemExit(main())
