from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.baselines.batch_only_baseline import batch_only_retrieval_metrics
from perturb_jepa.baselines.mean_prototype_alignment import MeanPrototypeAlignment
from perturb_jepa.baselines.metadata_only_retrieval import metadata_only_retrieval_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics, directional_retrieval_metrics


PAIRING_LABEL_BY_MODE = {
    "condition": "condition_key",
    "sample": "sample_id",
    "well": "well_id",
    "spot": "spot_id",
    "tile": "tile_id",
    "none": "condition_key",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate real-data Perturb-JEPA benchmark outputs.")
    parser.add_argument("--rna-embeddings", type=Path)
    parser.add_argument("--image-embeddings", type=Path)
    parser.add_argument("--rna-metadata", type=Path)
    parser.add_argument("--image-metadata", type=Path)
    parser.add_argument("--paired-manifest", type=Path)
    parser.add_argument("--train-rna-metadata", type=Path)
    parser.add_argument("--train-image-metadata", type=Path)
    parser.add_argument("--train-target-embeddings", type=Path)
    parser.add_argument("--train-target-metadata", type=Path)
    parser.add_argument(
        "--pairing-mode",
        choices=sorted(PAIRING_LABEL_BY_MODE),
        default="condition",
        help="Label granularity for paired/weakly paired retrieval metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--baselines-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--dry-run", action="store_true", help="Write an unavailable-metrics scaffold without inputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.baselines_dir.mkdir(parents=True, exist_ok=True)
    missing = _missing_required(args)
    if missing:
        if not args.dry_run:
            raise SystemExit(
                "error: missing required evaluation inputs: "
                + ", ".join(missing)
                + ". Use --dry-run to write a scaffold or provide embeddings/metadata."
            )
        payload = _unavailable_payload(missing)
        _write_outputs(payload, args.output_dir, args.baselines_dir)
        return 0

    rna_embeddings = np.load(args.rna_embeddings)
    image_embeddings = np.load(args.image_embeddings)
    rna_metadata = pd.read_csv(args.rna_metadata)
    image_metadata = pd.read_csv(args.image_metadata)
    learned = cross_modal_retrieval_metrics(rna_embeddings, image_embeddings, rna_metadata, image_metadata)
    rows = [{"result_type": "implementation result", "method": "learned", **learned}]

    baseline_rows = [
        {"result_type": "implementation result", "method": "metadata_only", **metadata_only_retrieval_metrics(rna_metadata, image_metadata)},
        {"result_type": "implementation result", "method": "batch_only", **batch_only_retrieval_metrics(rna_metadata, image_metadata)},
        {"result_type": "implementation result", "method": "random_embedding", **_random_embedding_metrics(rna_embeddings, image_embeddings, rna_metadata, image_metadata)},
    ]
    baseline_rows.append(
        {
            "result_type": "implementation result",
            "method": "mean_prototype_oracle",
            **MeanPrototypeAlignment().fit(image_embeddings, image_metadata).evaluate(
                rna_metadata,
                image_embeddings,
                image_metadata,
                prefix="mean_prototype_oracle",
            ),
        }
    )
    if args.train_target_embeddings is not None and args.train_target_metadata is not None:
        baseline_rows.append(
            {
                "result_type": "implementation result",
                "method": "mean_prototype_trainfit",
                **MeanPrototypeAlignment()
                .fit(np.load(args.train_target_embeddings), pd.read_csv(args.train_target_metadata))
                .evaluate(rna_metadata, image_embeddings, image_metadata, prefix="mean_prototype_trainfit"),
            }
        )
    else:
        baseline_rows.append(
            {
                "result_type": "not available",
                "method": "mean_prototype_trainfit",
                "reason": "train target embeddings and metadata were not provided",
            }
        )

    paired_rows = _paired_metrics(args, rna_embeddings, image_embeddings, rna_metadata, image_metadata)
    diagnostics = _leakage_diagnostics(args, rna_metadata, image_metadata)
    payload = {
        "metrics": rows,
        "baselines": baseline_rows,
        "paired_metrics": paired_rows,
        "leakage_diagnostics": diagnostics,
        "notes": [
            "Retrieval scores use embeddings only; metadata is used for relevance labels, baselines, and diagnostics.",
            "Metrics marked not available were not computed because required labels or files were absent.",
        ],
    }
    _write_outputs(payload, args.output_dir, args.baselines_dir)
    return 0


def _missing_required(args: argparse.Namespace) -> list[str]:
    required = {
        "--rna-embeddings": args.rna_embeddings,
        "--image-embeddings": args.image_embeddings,
        "--rna-metadata": args.rna_metadata,
        "--image-metadata": args.image_metadata,
    }
    return [name for name, path in required.items() if path is None or not path.exists()]


def _random_embedding_metrics(
    rna_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    rna_metadata: pd.DataFrame,
    image_metadata: pd.DataFrame,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    rna_random = rng.normal(size=rna_embeddings.shape)
    image_random = rng.normal(size=image_embeddings.shape)
    return cross_modal_retrieval_metrics(rna_random, image_random, rna_metadata, image_metadata)


def _paired_metrics(
    args: argparse.Namespace,
    rna_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    rna_metadata: pd.DataFrame,
    image_metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    if args.pairing_mode == "none":
        return [{"result_type": "not available", "method": "paired_retrieval", "reason": "pairing mode is none"}]
    label_col = PAIRING_LABEL_BY_MODE[args.pairing_mode]
    if label_col not in rna_metadata.columns or label_col not in image_metadata.columns:
        return [
            {
                "result_type": "not available",
                "method": "paired_retrieval",
                "pairing_mode": args.pairing_mode,
                "reason": f"label column {label_col!r} is absent from RNA or image metadata",
            }
        ]
    rows = [
        {
            "result_type": "implementation result",
            "method": "paired_retrieval",
            "pairing_mode": args.pairing_mode,
            **directional_retrieval_metrics(
                rna_embeddings,
                image_embeddings,
                rna_metadata,
                image_metadata,
                label_col=label_col,
                prefix=f"rna_to_image_{args.pairing_mode}",
            ),
        }
    ]
    shuffled = image_metadata.copy()
    shuffled[label_col] = shuffled[label_col].sample(frac=1.0, random_state=0).to_numpy()
    rows.append(
        {
            "result_type": "implementation result",
            "method": "shuffled_pairing",
            "pairing_mode": args.pairing_mode,
            **directional_retrieval_metrics(
                rna_embeddings,
                image_embeddings,
                rna_metadata,
                shuffled,
                label_col=label_col,
                prefix=f"rna_to_image_shuffled_{args.pairing_mode}",
            ),
        }
    )
    if args.paired_manifest is not None and args.paired_manifest.exists():
        paired = pd.read_csv(args.paired_manifest)
        rows.append(
            {
                "result_type": "verified fact",
                "method": "paired_manifest_summary",
                "pairing_mode": args.pairing_mode,
                "n_pairing_rows": int(len(paired)),
                "pairing_tiers": ",".join(sorted(paired.get("pairing_tier", pd.Series(dtype=str)).astype(str).unique())),
            }
        )
    return rows


def _leakage_diagnostics(args: argparse.Namespace, rna_metadata: pd.DataFrame, image_metadata: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ["condition_key", "perturbation", "dose", "time", "cell_line", "batch", "plate", "well", "site"]:
        if column in rna_metadata.columns and column in image_metadata.columns:
            rna_values = set(rna_metadata[column].astype(str))
            image_values = set(image_metadata[column].astype(str))
            rows.append(
                {
                    "result_type": "implementation result",
                    "diagnostic": f"rna_image_{column}_overlap",
                    "n_rna_values": len(rna_values),
                    "n_image_values": len(image_values),
                    "n_overlap": len(rna_values & image_values),
                }
            )
    if args.train_rna_metadata is not None and args.train_rna_metadata.exists():
        train = pd.read_csv(args.train_rna_metadata)
        rows.extend(_train_eval_overlap(train, rna_metadata, prefix="rna"))
    if args.train_image_metadata is not None and args.train_image_metadata.exists():
        train = pd.read_csv(args.train_image_metadata)
        rows.extend(_train_eval_overlap(train, image_metadata, prefix="image"))
    if not rows:
        rows.append({"result_type": "not available", "diagnostic": "leakage", "reason": "no shared metadata columns found"})
    return rows


def _train_eval_overlap(train: pd.DataFrame, eval_metadata: pd.DataFrame, *, prefix: str) -> list[dict[str, Any]]:
    rows = []
    for column in ["condition_key", "perturbation", "batch", "plate", "well", "site", "pairing_id"]:
        if column in train.columns and column in eval_metadata.columns:
            train_values = set(train[column].astype(str))
            eval_values = set(eval_metadata[column].astype(str))
            rows.append(
                {
                    "result_type": "implementation result",
                    "diagnostic": f"{prefix}_train_eval_{column}_overlap",
                    "n_train_values": len(train_values),
                    "n_eval_values": len(eval_values),
                    "n_overlap": len(train_values & eval_values),
                    "leakage_flag": bool(column in {"condition_key", "pairing_id"} and train_values & eval_values),
                }
            )
    return rows


def _unavailable_payload(missing: list[str]) -> dict[str, list[dict[str, Any]]]:
    reason = "missing inputs: " + ", ".join(missing)
    return {
        "metrics": [{"result_type": "not available", "method": "learned", "reason": reason}],
        "baselines": [{"result_type": "not available", "method": "all_baselines", "reason": reason}],
        "paired_metrics": [{"result_type": "not available", "method": "paired_retrieval", "reason": reason}],
        "leakage_diagnostics": [{"result_type": "not available", "diagnostic": "leakage", "reason": reason}],
    }


def _write_outputs(payload: dict[str, Any], output_dir: Path, baselines_dir: Path) -> None:
    metrics = pd.DataFrame(payload["metrics"])
    baselines = pd.DataFrame(payload["baselines"])
    paired = pd.DataFrame(payload["paired_metrics"])
    diagnostics = pd.DataFrame(payload["leakage_diagnostics"])
    metrics.to_csv(output_dir / "real_benchmark_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_metrics.csv", index=False)
    diagnostics.to_csv(output_dir / "leakage_diagnostics.csv", index=False)
    baselines.to_csv(baselines_dir / "real_benchmark_baselines.csv", index=False)
    (output_dir / "real_benchmark_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (baselines_dir / "real_benchmark_baselines.json").write_text(
        json.dumps({"baselines": payload["baselines"]}, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote real benchmark metrics to {output_dir}")
    print(f"Wrote baseline metrics to {baselines_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
