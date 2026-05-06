from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.baselines import ControlMeanBaseline, PerturbationMeanBaseline
from perturb_jepa.evaluation.reporting import grouped_metric_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate expression mean baselines.")
    parser.add_argument("--train-expression", type=Path, required=True, help="Training expression .npy file.")
    parser.add_argument("--train-metadata", type=Path, required=True, help="Training metadata CSV.")
    parser.add_argument("--eval-expression", type=Path, required=True, help="Evaluation expression .npy file.")
    parser.add_argument("--eval-metadata", type=Path, required=True, help="Evaluation metadata CSV.")
    parser.add_argument(
        "--perturbation-groupby",
        default="perturbation",
        help="Comma-separated metadata columns for the perturbation mean baseline.",
    )
    parser.add_argument(
        "--report-groupby",
        default="perturbation",
        help="Comma-separated metadata columns for grouped metric reporting. Use '' for overall only.",
    )
    parser.add_argument("--topk", type=int, default=50, help="Top-k used for delta Jaccard.")
    parser.add_argument(
        "--fallback",
        choices=("global", "control"),
        default="control",
        help="Fallback mean for unseen perturbation groups.",
    )
    parser.add_argument("--output", type=Path, help="Optional output CSV path. Defaults to stdout.")
    return parser


def evaluate_expression_baselines(
    train_expression: np.ndarray,
    train_metadata: pd.DataFrame,
    eval_expression: np.ndarray,
    eval_metadata: pd.DataFrame,
    *,
    perturbation_groupby: str | Sequence[str] = "perturbation",
    report_groupby: str | Sequence[str] | None = "perturbation",
    fallback: str = "control",
    topk: int = 50,
) -> pd.DataFrame:
    control_model = ControlMeanBaseline().fit(train_expression, train_metadata)
    control_prediction = control_model.predict(eval_metadata)

    perturbation_model = PerturbationMeanBaseline(
        groupby=perturbation_groupby,
        fallback=fallback,
    ).fit(train_expression, train_metadata)
    perturbation_prediction = perturbation_model.predict(eval_metadata)

    reports: list[pd.DataFrame] = []
    for baseline_name, prediction in (
        ("control_mean", control_prediction),
        ("perturbation_mean", perturbation_prediction),
    ):
        report = grouped_metric_report(
            prediction,
            eval_expression,
            control_prediction,
            eval_metadata,
            groupby=report_groupby,
            topk=topk,
        )
        report.insert(0, "baseline", baseline_name)
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_expression = np.load(args.train_expression)
    eval_expression = np.load(args.eval_expression)
    train_metadata = pd.read_csv(args.train_metadata)
    eval_metadata = pd.read_csv(args.eval_metadata)

    report = evaluate_expression_baselines(
        train_expression,
        train_metadata,
        eval_expression,
        eval_metadata,
        perturbation_groupby=_parse_columns(args.perturbation_groupby),
        report_groupby=_parse_optional_columns(args.report_groupby),
        fallback=args.fallback,
        topk=args.topk,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    else:
        print(report.to_csv(index=False), end="")
    return 0


def _parse_columns(value: str) -> tuple[str, ...]:
    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    if not columns:
        raise ValueError("at least one column is required")
    return columns


def _parse_optional_columns(value: str) -> tuple[str, ...] | None:
    columns = tuple(column.strip() for column in value.split(",") if column.strip())
    return columns or None


if __name__ == "__main__":
    raise SystemExit(main())
