from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.baselines.batch_only_baseline import batch_only_retrieval_metrics
from perturb_jepa.baselines.metadata_only_retrieval import metadata_only_retrieval_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate learned cross-modal retrieval embeddings.")
    parser.add_argument("--rna-embeddings", type=Path)
    parser.add_argument("--image-embeddings", type=Path)
    parser.add_argument("--rna-metadata", type=Path)
    parser.add_argument("--image-metadata", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Run on a deterministic synthetic dataset.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        rna_embeddings, image_embeddings, rna_metadata, image_metadata = _synthetic_data(
            seed=args.seed,
            conditions=args.conditions,
            dim=args.dim,
        )
    else:
        required = (args.rna_embeddings, args.image_embeddings, args.rna_metadata, args.image_metadata)
        if any(value is None for value in required):
            raise ValueError("provide RNA/image embeddings and metadata, or pass --synthetic")
        rna_embeddings = np.load(args.rna_embeddings)
        image_embeddings = np.load(args.image_embeddings)
        rna_metadata = pd.read_csv(args.rna_metadata)
        image_metadata = pd.read_csv(args.image_metadata)

    rows = [
        {
            "method": "learned_embeddings",
            **cross_modal_retrieval_metrics(rna_embeddings, image_embeddings, rna_metadata, image_metadata),
        },
        {
            "method": "metadata_only",
            **metadata_only_retrieval_metrics(rna_metadata, image_metadata),
        },
        {
            "method": "batch_only",
            **batch_only_retrieval_metrics(rna_metadata, image_metadata),
        },
    ]
    report = pd.DataFrame.from_records(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _synthetic_data(*, seed: int, conditions: int, dim: int):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(conditions, dim)).astype(np.float32)
    rna = base + 0.03 * rng.normal(size=base.shape).astype(np.float32)
    image = base + 0.03 * rng.normal(size=base.shape).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "condition_key": [f"drug{i}|{i % 3 + 1}uM|{24 + 24 * (i % 2)}h|CL{i % 2}" for i in range(conditions)],
            "perturbation": [f"drug{i}" for i in range(conditions)],
            "dose": [f"{i % 3 + 1}uM" for i in range(conditions)],
            "time": [f"{24 + 24 * (i % 2)}h" for i in range(conditions)],
            "cell_line": [f"CL{i % 2}" for i in range(conditions)],
            "moa": [f"moa{i % 3}" for i in range(conditions)],
            "batch": [f"batch{i % 2}" for i in range(conditions)],
            "plate": [f"plate{i % 3}" for i in range(conditions)],
            "run": [f"run{i % 2}" for i in range(conditions)],
            "sequencing_lane": [f"L{i % 2}" for i in range(conditions)],
            "library_id": [f"lib{i % 4}" for i in range(conditions)],
        }
    )
    return rna, image, metadata.copy(), metadata.copy()


if __name__ == "__main__":
    raise SystemExit(main())
