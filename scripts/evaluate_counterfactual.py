from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.image_counterfactual import image_counterfactual_metrics
from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RNA and image counterfactual predictions.")
    parser.add_argument("--predicted-rna", type=Path)
    parser.add_argument("--observed-rna", type=Path)
    parser.add_argument("--control-rna", type=Path)
    parser.add_argument("--predicted-image-embeddings", type=Path)
    parser.add_argument("--observed-image-embeddings", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Run on a deterministic synthetic dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=int, default=8)
    parser.add_argument("--genes", type=int, default=32)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        predicted_rna, observed_rna, control_rna, predicted_image, observed_image, metadata = _synthetic_data(
            seed=args.seed,
            conditions=args.conditions,
            genes=args.genes,
            dim=args.dim,
        )
    else:
        required = (
            args.predicted_rna,
            args.observed_rna,
            args.control_rna,
            args.predicted_image_embeddings,
            args.observed_image_embeddings,
            args.metadata,
        )
        if any(value is None for value in required):
            raise ValueError("provide prediction arrays and metadata, or pass --synthetic")
        predicted_rna = np.load(args.predicted_rna)
        observed_rna = np.load(args.observed_rna)
        control_rna = np.load(args.control_rna)
        predicted_image = np.load(args.predicted_image_embeddings)
        observed_image = np.load(args.observed_image_embeddings)
        metadata = pd.read_csv(args.metadata)

    metrics = {
        **rna_counterfactual_metrics(predicted_rna, observed_rna, control_rna, metadata),
        **image_counterfactual_metrics(predicted_image, observed_image, metadata),
    }
    report = pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _synthetic_data(*, seed: int, conditions: int, genes: int, dim: int):
    rng = np.random.default_rng(seed)
    control = rng.normal(size=(conditions, genes)).astype(np.float32)
    effect = rng.normal(scale=0.4, size=(conditions, genes)).astype(np.float32)
    observed = control + effect + rng.normal(scale=0.03, size=(conditions, genes)).astype(np.float32)
    predicted = control + effect + rng.normal(scale=0.05, size=(conditions, genes)).astype(np.float32)
    image_base = rng.normal(size=(conditions, dim)).astype(np.float32)
    image_observed = image_base + rng.normal(scale=0.03, size=(conditions, dim)).astype(np.float32)
    image_predicted = image_base + rng.normal(scale=0.05, size=(conditions, dim)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "condition_key": [f"drug{i}|{i % 3 + 1}uM|{24 + 24 * (i % 2)}h|CL{i % 2}" for i in range(conditions)],
            "perturbation": [f"drug{i}" for i in range(conditions)],
            "dose": [f"{i % 3 + 1}uM" for i in range(conditions)],
            "time": [f"{24 + 24 * (i % 2)}h" for i in range(conditions)],
            "cell_line": [f"CL{i % 2}" for i in range(conditions)],
            "moa": [f"moa{i % 3}" for i in range(conditions)],
        }
    )
    return predicted, observed, control, image_predicted, image_observed, metadata


if __name__ == "__main__":
    raise SystemExit(main())
