from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.training.real_data import load_yaml_or_json_config, raw_get
from scripts.train_bridge import _optional_path, _run_real_bridge_training, _run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Perturb-JEPA on downstream perturbation tasks.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated synthetic batches.")
    parser.add_argument("--rna-anndata", type=Path, default=None)
    parser.add_argument("--image-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args(argv)

    config, raw_config = load_yaml_or_json_config(args.config)
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.device is not None:
        config = replace(config, training=replace(config.training, device=args.device))
    rna_path = args.rna_anndata or _optional_path(config.data.rna_anndata or raw_get(raw_config, ("data", "rna_anndata")))
    manifest_path = args.image_manifest or _optional_path(
        config.data.image_manifest or raw_get(raw_config, ("data", "image_manifest"))
    )
    if args.synthetic or rna_path is None or manifest_path is None:
        if not args.synthetic:
            print("Real fine-tuning needs both --rna-anndata and --image-manifest; running synthetic fine-tune.")
        _run_synthetic_training(config, stage="finetune", checkpoint_out=args.checkpoint_out)
    else:
        _run_real_bridge_training(
            config,
            raw_config=raw_config,
            rna_path=rna_path,
            manifest_path=manifest_path,
            image_root=args.image_root or _optional_path(config.data.image_root or raw_get(raw_config, ("data", "image_root"))) or Path(""),
            max_cells=args.max_cells or raw_get(raw_config, ("data", "max_cells")),
            max_images=args.max_images or raw_get(raw_config, ("data", "max_images")),
            n_top_genes=args.n_top_genes or raw_get(raw_config, ("data", "n_top_genes")),
            batch_size=args.batch_size or config.training.batch_size,
            checkpoint_out=args.checkpoint_out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
