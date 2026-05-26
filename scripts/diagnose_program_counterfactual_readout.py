from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_synthetic_lite_step0 import _jsonable, _program_effect_recovery


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PROGRAM_COUNTERFACTUAL_READOUT_AUDIT")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose closed-form synthetic program-delta counterfactual readouts.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=seed))
        train = _pair_table(dataset, split="train")
        test = _pair_table(dataset, split="test")
        if train["program_delta"].shape[0] == 0 or test["program_delta"].shape[0] == 0:
            raise RuntimeError(f"seed {seed} has no train/test counterfactual pairs")

        predictors = {
            "source_as_target": np.zeros_like(test["program_delta"]),
            "condition_train_mean_delta": _condition_lookup_predict(train, test),
            "metadata_ridge": _ridge_predict(
                _metadata_features(train["metadata"], dataset),
                train["program_delta"],
                _metadata_features(test["metadata"], dataset),
                ridge=args.ridge,
            ),
            "metadata_no_batch_ridge": _ridge_predict(
                _metadata_features(train["metadata"], dataset, include_batch=False),
                train["program_delta"],
                _metadata_features(test["metadata"], dataset, include_batch=False),
                ridge=args.ridge,
            ),
            "source_program_metadata_ridge": _ridge_predict(
                np.concatenate((_metadata_features(train["metadata"], dataset), train["control_program"]), axis=1),
                train["program_delta"],
                np.concatenate((_metadata_features(test["metadata"], dataset), test["control_program"]), axis=1),
                ridge=args.ridge,
            ),
            "source_program_no_batch_metadata_ridge": _ridge_predict(
                np.concatenate(
                    (_metadata_features(train["metadata"], dataset, include_batch=False), train["control_program"]),
                    axis=1,
                ),
                train["program_delta"],
                np.concatenate(
                    (_metadata_features(test["metadata"], dataset, include_batch=False), test["control_program"]),
                    axis=1,
                ),
                ridge=args.ridge,
            ),
            "oracle_observed_delta": test["program_delta"],
        }
        for name, predicted_program_delta in predictors.items():
            predicted = test["control_gene"] + _broadcast_program_delta(predicted_program_delta, dataset.gene_program_assignment)
            metrics = rna_counterfactual_metrics(
                predicted,
                test["target_gene"],
                test["control_gene"],
                test["metadata"],
                groupby=None,
                topk=(50,),
                gene_sets=_gene_sets(dataset),
            )
            rows.append(
                {
                    "seed": seed,
                    "predictor": name,
                    "program_level_effect_recovery": _program_effect_recovery(
                        predicted,
                        test["target_gene"],
                        test["control_gene"],
                        dataset.gene_program_assignment,
                    ),
                    "direction_accuracy": metrics["rna_counterfactual_direction_accuracy"],
                    "logfc_correlation": metrics["rna_counterfactual_logfc_correlation"],
                    "pseudobulk_correlation": metrics["rna_counterfactual_pseudobulk_correlation"],
                    "top50_de_overlap": metrics["rna_counterfactual_top50_de_overlap"],
                }
            )
        seed_dir = args.output_dir / f"{args.dataset}_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "generation_config.json").write_text(
            json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "PROGRAM_READOUT_RESULTS.tsv", sep="\t", index=False)
    _write_report(args.output_dir / "REPORT.md", frame, args=args)
    print(frame.to_json(orient="records"))
    return 0


def _pair_table(dataset, *, split: str) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    key_to_group = {
        (
            int(row.perturbation_id),
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        ): group
        for row, group in zip(metadata.itertuples(index=False), groups, strict=True)
    }
    rows = []
    control_gene = []
    target_gene = []
    control_program = []
    target_program = []
    for row, target_group in zip(metadata.itertuples(index=False), groups, strict=True):
        if int(row.perturbation_id) == dataset.config.control_perturbation_id:
            continue
        control_key = (
            dataset.config.control_perturbation_id,
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        )
        control_group = key_to_group.get(control_key)
        if control_group is None:
            continue
        control_mean = dataset.expression_values[control_group].mean(axis=0)
        target_mean = dataset.expression_values[target_group].mean(axis=0)
        control_gene.append(control_mean)
        target_gene.append(target_mean)
        control_program.append(_program_means(control_mean[None, :], dataset.gene_program_assignment)[0])
        target_program.append(_program_means(target_mean[None, :], dataset.gene_program_assignment)[0])
        rows.append(row._asdict())
    control_program_array = np.asarray(control_program, dtype=float)
    target_program_array = np.asarray(target_program, dtype=float)
    return {
        "metadata": pd.DataFrame(rows),
        "control_gene": np.asarray(control_gene, dtype=float),
        "target_gene": np.asarray(target_gene, dtype=float),
        "control_program": control_program_array,
        "target_program": target_program_array,
        "program_delta": target_program_array - control_program_array,
    }


def _program_means(values: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    return np.stack([values[:, assignment == program].mean(axis=1) for program in sorted(np.unique(assignment))], axis=1)


def _broadcast_program_delta(program_delta: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    return program_delta[:, assignment]


def _condition_lookup_predict(train: dict[str, Any], test: dict[str, Any]) -> np.ndarray:
    fallback = train["program_delta"].mean(axis=0)
    mapping = {
        _condition_key(row): train["program_delta"][idx]
        for idx, row in enumerate(train["metadata"].itertuples(index=False))
    }
    return np.stack([mapping.get(_condition_key(row), fallback) for row in test["metadata"].itertuples(index=False)])


def _condition_key(row: Any) -> tuple[int, int, float, int]:
    return (int(row.perturbation_id), int(row.cell_line_id), float(row.dose), int(row.batch_id))


def _metadata_features(metadata: pd.DataFrame, dataset, *, include_batch: bool = True) -> np.ndarray:
    parts = [
        _one_hot(metadata["perturbation_id"].to_numpy(dtype=int), dataset.config.num_perturbations),
        _one_hot(metadata["cell_line_id"].to_numpy(dtype=int), dataset.config.num_cell_lines),
        metadata["dose"].to_numpy(dtype=float)[:, None],
        metadata["time"].to_numpy(dtype=float)[:, None],
    ]
    if include_batch:
        parts.insert(2, _one_hot(metadata["batch_id"].to_numpy(dtype=int), dataset.config.num_batches))
    return np.concatenate(parts, axis=1)


def _one_hot(values: np.ndarray, width: int) -> np.ndarray:
    result = np.zeros((values.shape[0], width), dtype=float)
    result[np.arange(values.shape[0]), values] = 1.0
    return result


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, ridge: float) -> np.ndarray:
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(x_train.shape[1])
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ coef


def _gene_sets(dataset) -> dict[str, list[int]]:
    return {
        f"program_{int(program)}": np.flatnonzero(dataset.gene_program_assignment == program).astype(int).tolist()
        for program in sorted(np.unique(dataset.gene_program_assignment))
    }


def _write_report(path: Path, frame: pd.DataFrame, *, args: argparse.Namespace) -> None:
    summary = frame.groupby("predictor").agg(["mean", "std"])
    lines = [
        "# Program Counterfactual Readout Audit",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seeds: `{args.seeds}`",
        "- Real data used: `false`",
        "- External biological resources used: `false`",
        "",
        "## Summary",
        "",
    ]
    for predictor, group in frame.groupby("predictor", sort=False):
        lines.append(f"### {predictor}")
        for metric in (
            "program_level_effect_recovery",
            "direction_accuracy",
            "logfc_correlation",
            "pseudobulk_correlation",
            "top50_de_overlap",
        ):
            values = group[metric].astype(float)
            lines.append(f"- {metric}: `{values.mean():.4f} +/- {values.std(ddof=0):.4f}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The condition lookup is a technical-duplicate ceiling for this split: train and test contain cells from the same synthetic conditions.",
            "Metadata ridge and source-program metadata ridge test whether a simple closed-form readout can recover program deltas without neural counterfactual decoder training.",
            "",
            "## Artifacts",
            "",
            "- `PROGRAM_READOUT_RESULTS.tsv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
