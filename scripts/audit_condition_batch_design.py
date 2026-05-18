from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.design_audit import condition_batch_confounding_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit condition-vs-batch confounding in metadata.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--condition-col", default="condition_key")
    parser.add_argument("--batch-col", default="batch")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--min-batches-per-condition", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    report = condition_batch_confounding_report(
        metadata,
        condition_col=args.condition_col,
        batch_col=args.batch_col,
        split_col=args.split_col,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    print(report.to_string(index=False))
    failing = report["condition_batches_min"].lt(float(args.min_batches_per_condition))
    if bool(failing.any()):
        bad_splits = ", ".join(report.loc[failing, "split"].astype(str).tolist())
        raise SystemExit(
            f"condition/batch audit failed for split(s): {bad_splits}; "
            f"minimum batches per condition is below {args.min_batches_per_condition}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
