from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.baselines import (
    centroid_retrieval_metrics,
    label_shuffle_centroid_retrieval_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate centroid retrieval baselines.")
    parser.add_argument("--gallery-embeddings", type=Path, required=True, help="Gallery embeddings .npy file.")
    parser.add_argument("--gallery-metadata", type=Path, required=True, help="Gallery metadata CSV.")
    parser.add_argument("--query-embeddings", type=Path, required=True, help="Query embeddings .npy file.")
    parser.add_argument("--query-metadata", type=Path, required=True, help="Query metadata CSV.")
    parser.add_argument("--label-col", default="condition_key", help="Metadata label column for retrieval.")
    parser.add_argument("--ks", default="1,5", help="Comma-separated recall@k values.")
    parser.add_argument("--random-state", type=int, default=0, help="Seed for label-shuffle control.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable row L2 normalization.")
    parser.add_argument("--output", type=Path, help="Optional output CSV path. Defaults to stdout.")
    return parser


def evaluate_retrieval_baselines(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_metadata: pd.DataFrame,
    gallery_metadata: pd.DataFrame,
    *,
    label_col: str = "condition_key",
    ks: Sequence[int] = (1, 5),
    random_state: int = 0,
    normalize: bool = True,
) -> pd.DataFrame:
    query_labels = _metadata_labels(query_metadata, label_col)
    gallery_labels = _metadata_labels(gallery_metadata, label_col)
    centroid_metrics = centroid_retrieval_metrics(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        ks=ks,
        normalize=normalize,
        prefix="retrieval",
    )
    shuffle_metrics = label_shuffle_centroid_retrieval_metrics(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        ks=ks,
        random_state=random_state,
        normalize=normalize,
        prefix="retrieval",
    )
    return pd.DataFrame.from_records(
        [
            {"baseline": "centroid_retrieval", **centroid_metrics},
            {"baseline": "label_shuffle_centroid", **shuffle_metrics},
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query_embeddings = np.load(args.query_embeddings)
    gallery_embeddings = np.load(args.gallery_embeddings)
    query_metadata = pd.read_csv(args.query_metadata)
    gallery_metadata = pd.read_csv(args.gallery_metadata)

    report = evaluate_retrieval_baselines(
        query_embeddings,
        gallery_embeddings,
        query_metadata,
        gallery_metadata,
        label_col=args.label_col,
        ks=_parse_ks(args.ks),
        random_state=args.random_state,
        normalize=not args.no_normalize,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _metadata_labels(metadata: pd.DataFrame, label_col: str) -> list[str]:
    if label_col not in metadata.columns:
        raise ValueError(f"metadata is missing label column: {label_col}")
    return metadata[label_col].astype(str).tolist()


def _parse_ks(value: str) -> tuple[int, ...]:
    ks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain positive integers")
    return ks


if __name__ == "__main__":
    raise SystemExit(main())
