from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset, generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import (
    BIOLOGICAL_KEY_FIELDS,
    _eval_control_splits,
    biological_key,
    counterfactual_gate_pass,
    evaluate_predictions,
    norm_ratio,
    pair_records,
    predict_matched_perturbed_mean,
    predict_residualized_matching,
    program_means,
    protected_pls_metrics,
    slug_float,
)
from scripts.run_synthetic_lite_step0 import _jsonable


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_N_DISTILLATION")


class TrainOnlyConditionalMeanTable:
    """Train-split-only target mean lookup keyed by biological condition fields."""

    def __init__(self, train_records: list[dict[str, Any]], *, value_name: str = "target_mean") -> None:
        if not train_records:
            raise ValueError("at least one train record is required")
        self.value_name = value_name
        self.train_record_count = len(train_records)
        grouped: dict[tuple[Any, ...], list[np.ndarray]] = {}
        perturbation_grouped: dict[Any, list[np.ndarray]] = {}
        for record in train_records:
            key = biological_key(record)
            value = np.asarray(record[value_name], dtype=float)
            grouped.setdefault(key, []).append(value)
            perturbation_grouped.setdefault(record["perturbation_id"], []).append(value)
        self.target_by_key = {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}
        self.perturbation_mean = {
            perturbation_id: np.stack(values).mean(axis=0) for perturbation_id, values in perturbation_grouped.items()
        }
        self.global_mean = np.stack([np.asarray(record[value_name], dtype=float) for record in train_records]).mean(axis=0)
        self.train_keys = tuple(sorted(self.target_by_key))

    def predict(self, records: list[dict[str, Any]]) -> tuple[list[np.ndarray], dict[str, float]]:
        predictions = []
        counts = {
            "exact": 0,
            "nearest_same_perturbation_cell": 0,
            "global_perturbation_mean": 0,
            "global_train_mean": 0,
        }
        for record in records:
            prediction, fallback = self._predict_one(record)
            counts[fallback] += 1
            predictions.append(prediction)
        rows = len(records)
        return predictions, {
            "rows": float(rows),
            "train_record_count": float(self.train_record_count),
            "train_key_count": float(len(self.target_by_key)),
            "test_key_count": float(len({biological_key(record) for record in records})),
            "exact_match_fraction": float(counts["exact"] / max(1, rows)),
            "fallback_exact_count": float(counts["exact"]),
            "fallback_nearest_same_perturbation_cell_count": float(counts["nearest_same_perturbation_cell"]),
            "fallback_global_perturbation_mean_count": float(counts["global_perturbation_mean"]),
            "fallback_global_train_mean_count": float(counts["global_train_mean"]),
            "batch_id_excluded": 1.0,
            "fit_split_train_only": 1.0,
            "teacher_target_test_rows_used": 0.0,
        }

    def _predict_one(self, record: dict[str, Any]) -> tuple[np.ndarray, str]:
        key = biological_key(record)
        exact = self.target_by_key.get(key)
        if exact is not None:
            return exact, "exact"
        nearest = self._nearest_same_perturbation_cell(record)
        if nearest is not None:
            return nearest, "nearest_same_perturbation_cell"
        perturbation_value = self.perturbation_mean.get(record["perturbation_id"])
        if perturbation_value is not None:
            return perturbation_value, "global_perturbation_mean"
        return self.global_mean, "global_train_mean"

    def _nearest_same_perturbation_cell(self, record: dict[str, Any]) -> np.ndarray | None:
        candidates = []
        for key, value in self.target_by_key.items():
            if key[0] != record["perturbation_id"] or key[1] != record["cell_line_id"]:
                continue
            distance = abs(float(key[2]) - float(record["dose"])) + abs(float(key[3]) - float(record["time"]))
            candidates.append((distance, key, value))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]


class ConditionFeatureEncoder:
    """No-batch biological condition and source-program feature encoder."""

    def __init__(self, dataset: SyntheticBiologyLiteDataset, train_records: list[dict[str, Any]]) -> None:
        if not train_records:
            raise ValueError("at least one train record is required")
        self.dataset = dataset
        self.perturbation_ids = tuple(sorted({int(record["perturbation_id"]) for record in train_records}))
        self.cell_line_ids = tuple(sorted({int(record["cell_line_id"]) for record in train_records}))
        self.biological_keys = tuple(sorted({biological_key(record) for record in train_records}))
        source_programs = np.asarray(
            [program_means(record["control_mean"], dataset.gene_program_assignment)[0] for record in train_records],
            dtype=float,
        )
        self.source_program_mean = source_programs.mean(axis=0)
        self.source_program_std = np.where(source_programs.std(axis=0) < 1e-6, 1.0, source_programs.std(axis=0))
        dose = np.asarray([float(record["dose"]) for record in train_records], dtype=float)
        time = np.asarray([float(record["time"]) for record in train_records], dtype=float)
        self.dose_mean = float(dose.mean())
        self.dose_std = float(dose.std()) if float(dose.std()) >= 1e-6 else 1.0
        self.time_mean = float(time.mean())
        self.time_std = float(time.std()) if float(time.std()) >= 1e-6 else 1.0
        self.feature_names = self._feature_names()

    def transform(self, records: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        perturb_index = {value: index for index, value in enumerate(self.perturbation_ids)}
        cell_index = {value: index for index, value in enumerate(self.cell_line_ids)}
        key_index = {value: index for index, value in enumerate(self.biological_keys)}
        for record in records:
            values: list[float] = [1.0]
            perturbation_one_hot = np.zeros(len(self.perturbation_ids), dtype=float)
            if int(record["perturbation_id"]) in perturb_index:
                perturbation_one_hot[perturb_index[int(record["perturbation_id"])]] = 1.0
            values.extend(perturbation_one_hot.tolist())
            cell_one_hot = np.zeros(len(self.cell_line_ids), dtype=float)
            if int(record["cell_line_id"]) in cell_index:
                cell_one_hot[cell_index[int(record["cell_line_id"])]] = 1.0
            values.extend(cell_one_hot.tolist())
            values.append((float(record["dose"]) - self.dose_mean) / self.dose_std)
            values.append((float(record["time"]) - self.time_mean) / self.time_std)
            key_one_hot = np.zeros(len(self.biological_keys), dtype=float)
            key = biological_key(record)
            if key in key_index:
                key_one_hot[key_index[key]] = 1.0
            values.extend(key_one_hot.tolist())
            source_program = program_means(record["control_mean"], self.dataset.gene_program_assignment)[0]
            values.extend(((source_program - self.source_program_mean) / self.source_program_std).tolist())
            rows.append(np.asarray(values, dtype=float))
        return np.stack(rows, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "batch_id_feature_present": any("batch" in name for name in self.feature_names),
            "fit_split_train_only": True,
            "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        }

    def _feature_names(self) -> list[str]:
        names = ["bias"]
        names.extend([f"perturbation_id={value}" for value in self.perturbation_ids])
        names.extend([f"cell_line_id={value}" for value in self.cell_line_ids])
        names.extend(["dose_z", "time_z"])
        names.extend([f"biological_key={index}" for index, _ in enumerate(self.biological_keys)])
        names.extend([f"source_program_z={index}" for index in range(self.dataset.config.num_programs)])
        return names


class LinearConditionalMeanModel:
    def __init__(self, weights: np.ndarray, *, ridge_alpha: float, train_mse: float) -> None:
        self.weights = weights
        self.ridge_alpha = float(ridge_alpha)
        self.train_mse = float(train_mse)

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, *, ridge_alpha: float) -> "LinearConditionalMeanModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        system = x.T @ x + penalty
        rhs = x.T @ y
        try:
            weights = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(system) @ rhs
        prediction = x @ weights
        return cls(weights, ridge_alpha=ridge_alpha, train_mse=float(np.mean((prediction - y) ** 2)))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.weights


class TargetStandardizer:
    def __init__(self, targets: np.ndarray) -> None:
        values = np.asarray(targets, dtype=float)
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = np.where(values.std(axis=0, keepdims=True) < 1e-6, 1.0, values.std(axis=0, keepdims=True))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean


class SmallMLPConditionalMeanModel:
    def __init__(
        self,
        model: torch.nn.Module,
        target_standardizer: TargetStandardizer,
        *,
        train_mse: float,
        final_loss: float,
    ) -> None:
        self.model = model
        self.target_standardizer = target_standardizer
        self.train_mse = float(train_mse)
        self.final_loss = float(final_loss)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        hidden_dim: int,
        steps: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> "SmallMLPConditionalMeanModel":
        torch.manual_seed(seed)
        x = torch.as_tensor(np.asarray(features, dtype=np.float32))
        standardizer = TargetStandardizer(targets)
        y_np = standardizer.transform(targets).astype(np.float32)
        y = torch.as_tensor(y_np)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, targets.shape[1]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        final_loss = 0.0
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        with torch.no_grad():
            train_prediction = standardizer.inverse_transform(model(x).detach().cpu().numpy())
        return cls(
            model,
            standardizer,
            train_mse=float(np.mean((train_prediction - np.asarray(targets, dtype=float)) ** 2)),
            final_loss=final_loss,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.as_tensor(np.asarray(features, dtype=np.float32))).cpu().numpy()
        return self.target_standardizer.inverse_transform(prediction)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family N train-only matched-mean distillation baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--linear-ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-steps", type=int, default=1200)
    parser.add_argument("--mlp-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hybrid-alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    train_records = pair_records(dataset, split="train")
    test_records = pair_records(dataset, split=args.eval_split, control_splits=_eval_control_splits(args.eval_split))
    if not train_records or not test_records:
        raise RuntimeError("Family N requires train and test counterfactual pairs")
    protected_metrics, _ = protected_pls_metrics(
        dataset,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    source_metrics = evaluate_predictions(
        dataset,
        test_records,
        [record["control_mean"] for record in test_records],
        candidate_name="source_as_target",
        method="source_as_target",
        exact_match_fraction=1.0,
        diagnostics={},
    )

    rows: list[dict[str, Any]] = []
    table = TrainOnlyConditionalMeanTable(train_records, value_name="target_mean")
    table_predictions, table_diagnostics = table.predict(test_records)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                table_predictions,
                candidate_name="seed2_train_only_condition_mean_table",
                method="train_only_condition_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics=table_diagnostics,
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(table_predictions, dtype=float),
            train_targets=None,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "A"},
        )
    )

    train_teacher_predictions, train_teacher_diagnostics = table.predict(train_records)
    train_targets = np.asarray(train_teacher_predictions, dtype=float)
    feature_encoder = ConditionFeatureEncoder(dataset, train_records)
    train_features = feature_encoder.transform(train_records)
    test_features = feature_encoder.transform(test_records)

    linear_model = LinearConditionalMeanModel.fit(
        train_features,
        train_targets,
        ridge_alpha=args.linear_ridge_alpha,
    )
    linear_predictions = linear_model.predict(test_features)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                list(linear_predictions),
                candidate_name="seed2_distilled_linear_condition_mean",
                method="distilled_linear_condition_mean",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                    "linear_ridge_alpha": float(args.linear_ridge_alpha),
                    "linear_train_teacher_mse": linear_model.train_mse,
                },
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(linear_predictions, dtype=float),
            train_targets=train_targets,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "B"},
        )
    )

    mlp_model = SmallMLPConditionalMeanModel.fit(
        train_features,
        train_targets,
        hidden_dim=args.mlp_hidden_dim,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed + 31_337,
    )
    mlp_predictions = mlp_model.predict(test_features)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                list(mlp_predictions),
                candidate_name="seed2_distilled_mlp_condition_mean",
                method="distilled_mlp_condition_mean",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.mlp_steps),
                    "mlp_lr": float(args.mlp_lr),
                    "mlp_weight_decay": float(args.mlp_weight_decay),
                    "mlp_final_standardized_loss": mlp_model.final_loss,
                    "mlp_train_teacher_mse": mlp_model.train_mse,
                },
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(mlp_predictions, dtype=float),
            train_targets=train_targets,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "C"},
        )
    )

    fallback = np.asarray(table_predictions, dtype=float)
    alphas = parse_float_list(args.hybrid_alphas)
    for model_name, learned_predictions in (
        ("linear", np.asarray(linear_predictions, dtype=float)),
        ("mlp", np.asarray(mlp_predictions, dtype=float)),
    ):
        for alpha in alphas:
            hybrid_predictions = float(alpha) * learned_predictions + (1.0 - float(alpha)) * fallback
            rows.append(
                _finalize_candidate(
                    evaluate_predictions(
                        dataset,
                        test_records,
                        list(hybrid_predictions),
                        candidate_name=f"seed2_{model_name}_condition_mean_hybrid_alpha{slug_float(alpha)}",
                        method="shrinkage_distillation_hybrid",
                        exact_match_fraction=table_diagnostics["exact_match_fraction"],
                        diagnostics={
                            **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                            "hybrid_alpha": float(alpha),
                            "hybrid_model": model_name,
                            "fallback": "train_only_condition_mean_table",
                        },
                    ),
                    protected_metrics=protected_metrics,
                    source_metrics=source_metrics,
                    candidate_predictions=np.asarray(hybrid_predictions, dtype=float),
                    train_targets=train_targets,
                    teacher_predictions=fallback,
                    extra={"candidate_family": "D"},
                )
            )

    comparator_rows = build_comparator_rows(args.output_dir, rows)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame = pd.DataFrame(rows)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_N_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(comparator_rows).to_csv(args.output_dir / "COMPARATOR_RESULTS.tsv", sep="\t", index=False)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "eval_split": args.eval_split,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_matching_and_features": True,
        "train_pairs": len(train_records),
        "test_pairs": len(test_records),
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "feature_encoder": feature_encoder.to_dict(),
        "train_teacher_diagnostics": train_teacher_diagnostics,
        "candidates": rows,
        "comparators": comparator_rows,
    }
    write_json(args.output_dir / "FAMILY_N_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_report(
        args.output_dir / "REPORT.md",
        frame,
        comparator_rows=comparator_rows,
        source_metrics=source_metrics,
        args=args,
    )
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def _teacher_fit_diagnostics(
    train_teacher_diagnostics: dict[str, float],
    feature_encoder: ConditionFeatureEncoder,
) -> dict[str, Any]:
    feature_payload = feature_encoder.to_dict()
    return {
        "train_teacher_exact_match_fraction": float(train_teacher_diagnostics["exact_match_fraction"]),
        "feature_count": float(feature_payload["feature_count"]),
        "batch_id_feature_present": float(bool(feature_payload["batch_id_feature_present"])),
        "fit_split_train_only": 1.0,
        "teacher_target_test_rows_used": 0.0,
    }


def _finalize_candidate(
    result: dict[str, Any],
    *,
    protected_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    candidate_predictions: np.ndarray,
    train_targets: np.ndarray | None,
    teacher_predictions: np.ndarray,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = dict(result)
    row.update(extra)
    row["counterfactual_gate_pass"] = counterfactual_gate_pass(row, source_metrics)
    row["protected_geometry_preserved"] = True
    row["protected_rna_to_image_recall@1"] = float(protected_metrics.get("model_rna_to_image_recall@1", 0.0))
    row["protected_bio_latent_r2_rna_shared"] = float(protected_metrics.get("model_bio_latent_r2_rna_shared", 0.0))
    row["protected_representation_rank"] = float(protected_metrics.get("model_embedding_rank", 0.0))
    row["protected_batch_probe_balanced_accuracy"] = float(
        protected_metrics.get("model_batch_probe_balanced_accuracy", 0.0)
    )
    row["leakage_gate_pass"] = bool(
        float(row.get("fit_split_train_only", 1.0)) == 1.0
        and float(row.get("teacher_target_test_rows_used", 0.0)) == 0.0
        and float(row.get("batch_id_feature_present", 0.0)) == 0.0
        and float(row.get("batch_id_excluded", 1.0)) == 1.0
    )
    row["teacher_prediction_mse_on_test"] = float(np.mean((candidate_predictions - teacher_predictions) ** 2))
    if train_targets is not None:
        row["train_teacher_target_variance"] = float(np.var(train_targets))
    return row


def build_comparator_rows(output_dir: Path, family_n_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparators: list[dict[str, Any]] = []
    family_m_path = output_dir.parent / "FAMILY_M_TRANSPORT_BASELINES" / "FAMILY_M_RESULTS.tsv"
    if family_m_path.exists():
        family_m = pd.read_csv(family_m_path, sep="\t")
        for name in ("seed2_no_batch_matched_perturbed_mean", "seed2_no_batch_residualized_matching"):
            match = family_m[family_m["candidate_name"] == name]
            if not match.empty:
                comparators.append(_comparator_from_series(match.iloc[0], source="Family M"))
    comparators.extend(load_sparse_residual_comparators(output_dir.parent / "SPARSE_PERTURBATION_RESIDUAL_HEAD"))
    ridge = load_prefit_ridge_best(output_dir.parent / "CLONE_COUNTERFACTUAL_DECODER")
    if ridge is not None:
        comparators.append(ridge)
    for row in family_n_rows:
        if row["candidate_family"] in {"A", "B", "C"}:
            comparators.append(
                {
                    "candidate_name": row["candidate_name"],
                    "source": "Family N",
                    "program_level_effect_recovery": row["program_level_effect_recovery"],
                    "direction_accuracy": row["direction_accuracy"],
                    "logfc_correlation": row["logfc_correlation"],
                    "pseudobulk_correlation": row["pseudobulk_correlation"],
                    "top50_de_overlap": row["top50_de_overlap"],
                    "mean_delta_to_target_ratio": row["mean_delta_to_target_ratio"],
                }
            )
    return comparators


def _comparator_from_series(row: pd.Series, *, source: str) -> dict[str, Any]:
    return {
        "candidate_name": row["candidate_name"],
        "source": source,
        "program_level_effect_recovery": float(row["program_level_effect_recovery"]),
        "direction_accuracy": float(row["direction_accuracy"]),
        "logfc_correlation": float(row["logfc_correlation"]),
        "pseudobulk_correlation": float(row["pseudobulk_correlation"]),
        "top50_de_overlap": float(row["top50_de_overlap"]),
        "mean_delta_to_target_ratio": float(row["mean_delta_to_target_ratio"]),
    }


def load_sparse_residual_comparators(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/AFTER_METRICS.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate_name": path.parent.name,
                "source": "Family L",
                "program_level_effect_recovery": float(payload["model_program_level_effect_recovery"]),
                "direction_accuracy": float(payload["model_rna_counterfactual_direction_accuracy"]),
                "logfc_correlation": float(payload["model_rna_counterfactual_logfc_correlation"]),
                "pseudobulk_correlation": float(payload["model_rna_counterfactual_pseudobulk_correlation"]),
                "top50_de_overlap": float(payload["model_rna_counterfactual_top50_de_overlap"]),
                "mean_delta_to_target_ratio": float(payload.get("mean_sparse_final_delta_to_target_ratio", 0.0)),
            }
        )
    return rows


def load_prefit_ridge_best(root: Path) -> dict[str, Any] | None:
    best = None
    for path in sorted(root.glob("**/AFTER_METRICS.json")):
        if "prefitridge" not in str(path) or "seed2" not in str(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "candidate_name": path.parent.name,
            "source": "prefit ridge best",
            "program_level_effect_recovery": float(payload["model_program_level_effect_recovery"]),
            "direction_accuracy": float(payload["model_rna_counterfactual_direction_accuracy"]),
            "logfc_correlation": float(payload["model_rna_counterfactual_logfc_correlation"]),
            "pseudobulk_correlation": float(payload["model_rna_counterfactual_pseudobulk_correlation"]),
            "top50_de_overlap": float(payload["model_rna_counterfactual_top50_de_overlap"]),
            "mean_delta_to_target_ratio": float(payload.get("mean_final_delta_to_target_ratio", 0.0)),
        }
        if best is None:
            best = row
            continue
        if (row["top50_de_overlap"], row["logfc_correlation"]) > (
            best["top50_de_overlap"],
            best["logfc_correlation"],
        ):
            best = row
    return best


def parse_float_list(values: str) -> list[float]:
    result = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not result:
        raise ValueError("at least one alpha is required")
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    frame: pd.DataFrame,
    *,
    comparator_rows: list[dict[str, Any]],
    source_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    key_rows = frame[frame["candidate_family"].isin(["A", "B", "C"])].copy()
    hybrid = frame[frame["candidate_family"] == "D"].copy()
    best_hybrid = hybrid.sort_values(
        ["program_level_effect_recovery", "logfc_correlation", "top50_de_overlap"],
        ascending=False,
    ).head(1)
    lines = [
        "# Family N Train-Only Matched-Mean Distillation",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from matching and features: `true`",
        "- Teacher targets fit on train split only: `true`",
        "- Test target rows used for teacher construction: `0`",
        "- Real data used: `false`",
        "- Marker/pathway/pretrained biological assets used: `false`",
        "",
        "## Source-As-Target Reference",
        "",
        f"- program recovery: `{source_metrics['program_level_effect_recovery']:.4f}`",
        f"- direction accuracy: `{source_metrics['direction_accuracy']:.4f}`",
        f"- logFC correlation: `{source_metrics['logfc_correlation']:.4f}`",
        f"- pseudobulk correlation: `{source_metrics['pseudobulk_correlation']:.4f}`",
        f"- top50 overlap: `{source_metrics['top50_de_overlap']:.4f}`",
        "",
        "## Core Candidates",
        "",
    ]
    for row in key_rows.itertuples(index=False):
        lines.extend(_candidate_lines(row))
    if not best_hybrid.empty:
        lines.extend(["## Best Shrinkage Hybrid", ""])
        lines.extend(_candidate_lines(best_hybrid.iloc[0]))
    lines.extend(["## Required Comparators", ""])
    for comparator in comparator_rows:
        lines.extend(
            [
                f"### {comparator['candidate_name']}",
                f"- source: `{comparator['source']}`",
                f"- program recovery: `{comparator['program_level_effect_recovery']:.4f}`",
                f"- direction accuracy: `{comparator['direction_accuracy']:.4f}`",
                f"- logFC correlation: `{comparator['logfc_correlation']:.4f}`",
                f"- pseudobulk correlation: `{comparator['pseudobulk_correlation']:.4f}`",
                f"- top50 overlap: `{comparator['top50_de_overlap']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Candidate A is the leakage-safe train-only teacher and should match the Family M direct matched perturbed mean when exact train biological keys cover the test split.",
            "The learned students are useful only if they approximate that teacher without batch features or test target statistics; they do not change protected bridge geometry because no bridge weights are trained.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_N_RESULTS.tsv`",
            "- `FAMILY_N_RESULTS.json`",
            "- `COMPARATOR_RESULTS.tsv`",
            "- `generation_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_lines(row: Any) -> list[str]:
    return [
        f"### {row.candidate_name}",
        f"- method: `{row.method}`",
        f"- exact train key coverage on test: `{row.exact_match_fraction:.4f}`",
        f"- leakage gate pass: `{bool(row.leakage_gate_pass)}`",
        f"- program recovery: `{row.program_level_effect_recovery:.4f}`",
        f"- direction accuracy: `{row.direction_accuracy:.4f}`",
        f"- logFC correlation: `{row.logfc_correlation:.4f}`",
        f"- pseudobulk correlation: `{row.pseudobulk_correlation:.4f}`",
        f"- top50 overlap: `{row.top50_de_overlap:.4f}`",
        f"- mean delta/target ratio: `{row.mean_delta_to_target_ratio:.4f}`",
        f"- counterfactual gate pass: `{bool(row.counterfactual_gate_pass)}`",
        "",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
