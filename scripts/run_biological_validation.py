from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.biology import run_biology_validation, write_validation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pseudobulk biological validation.")
    parser.add_argument("--expression", type=Path, required=True, help="Observed cell-by-gene expression .npy file.")
    parser.add_argument("--metadata", type=Path, required=True, help="Cell metadata CSV aligned to expression rows.")
    parser.add_argument("--predicted-expression", type=Path, help="Optional predicted cell-by-gene expression .npy file.")
    parser.add_argument("--genes", type=Path, help="Optional one-column gene name file or CSV.")
    parser.add_argument("--gmt", type=Path, help="Optional GMT-like pathway file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for validation artifacts.")
    parser.add_argument("--groupby", default=None, help="Comma-separated pseudobulk columns. Defaults to available perturbation,dose,time,cell_line.")
    parser.add_argument("--condition-col", default="perturbation", help="Perturbation/condition metadata column.")
    parser.add_argument("--control-values", default="control,DMSO,dmso,vehicle,untreated", help="Comma-separated control labels.")
    parser.add_argument("--control-type-col", default="perturbation_type", help="Metadata column whose 'control' value marks controls.")
    parser.add_argument("--topk", type=int, default=50, help="Top-k genes for DE recovery metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expression = np.load(args.expression)
    metadata = pd.read_csv(args.metadata)
    predicted = np.load(args.predicted_expression) if args.predicted_expression is not None else None
    genes = _read_genes(args.genes) if args.genes is not None else None
    groupby = _split_csv(args.groupby) if args.groupby else None
    control_values = _split_csv(args.control_values)

    result = run_biology_validation(
        expression,
        metadata,
        predicted_expression=predicted,
        gene_names=genes,
        groupby=groupby,
        condition_col=args.condition_col,
        control_values=control_values,
        control_type_col=args.control_type_col,
        topk=args.topk,
        gmt_path=args.gmt,
    )
    paths = write_validation_outputs(result, args.output_dir)
    for label, path in sorted(paths.items()):
        print(f"{label}: {path}")
    return 0


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_genes(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if frame.empty:
            return []
        column = "gene" if "gene" in frame.columns else frame.columns[0]
        return frame[column].astype(str).tolist()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
