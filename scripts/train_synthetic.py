from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import BridgeTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config-driven synthetic Perturb-JEPA training.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.load_json(args.config) if args.config is not None else ExperimentConfig.smoke()
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))

    seed_everything(config.training.seed)
    device = torch.device(config.training.device)
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        ema_decay=config.training.ema_decay,
        device=device,
        grad_clip_norm=config.training.grad_clip_norm,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume, map_location=device)

    for step in range(config.training.steps):
        batch = make_synthetic_bridge_batch(**asdict(config.synthetic), device=device)
        terms = trainer.step(batch)
        if config.training.log_every > 0 and (step + 1) % config.training.log_every == 0:
            print(
                f"step={trainer.global_step - 1} "
                + " ".join(f"{key}={value:.4f}" for key, value in sorted(terms.items()))
            )

    if args.checkpoint_out is not None:
        trainer.save_checkpoint(args.checkpoint_out, experiment_config=config)


if __name__ == "__main__":
    main()
