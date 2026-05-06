from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from scripts.train_bridge import _run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain the image encoder scaffold.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic image batches.")
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    if not args.synthetic:
        print("Image manifest loading is not wired in this scaffold; running synthetic pretrain.")
    _run_synthetic_training(config, stage="pretrain_image", checkpoint_out=args.checkpoint_out)
    return 0


def _load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig.smoke()
    if path.suffix.lower() == ".json":
        return ExperimentConfig.load_json(path)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_dict(yaml.safe_load(handle) or {})


if __name__ == "__main__":
    raise SystemExit(main())
