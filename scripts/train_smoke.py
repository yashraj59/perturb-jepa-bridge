from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import default_bridge_config
from perturb_jepa.models.bridge import PerturbJEPABridge
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import train_step


def build_smoke_model() -> PerturbJEPABridge:
    return PerturbJEPABridge(default_bridge_config())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic Perturb-JEPA smoke training.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
    device = torch.device(args.device)
    model = build_smoke_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(args.steps):
        batch = make_synthetic_bridge_batch(device=device)
        terms = train_step(model, optimizer, batch)
        print(f"step={step} " + " ".join(f"{key}={value:.4f}" for key, value in sorted(terms.items())))


if __name__ == "__main__":
    main()
