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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.training.prefit_readout import fit_pls_readout, install_prefit_pls_distillation_head, install_prefit_pls_readout
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset, generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _gene_sets, _jsonable, _program_effect_recovery, evaluate_step0
from scripts.train_clone_counterfactual_decoder import _counterfactual_pairs


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_M_TRANSPORT_BASELINES")
BIOLOGICAL_KEY_FIELDS = ("perturbation_id", "cell_line_id", "dose", "time")


class FeatureProjector:
    def __init__(
        self,
        *,
        dataset: SyntheticBiologyLiteDataset,
        feature_space: str,
        fit_values: np.ndarray,
        readout: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.feature_space = feature_space
        self.readout = readout
        fitted = self._raw_project(fit_values)
        self.mean = fitted.mean(axis=0, keepdims=True)
        self.std = np.where(fitted.std(axis=0, keepdims=True) < 1e-6, 1.0, fitted.std(axis=0, keepdims=True))

    def project(self, values: np.ndarray) -> np.ndarray:
        raw = self._raw_project(values)
        return (raw - self.mean) / self.std

    def _raw_project(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if self.feature_space == "rna":
            return values
        if self.feature_space == "program":
            return program_means(values, self.dataset.gene_program_assignment)
        if self.feature_space == "pls":
            if self.readout is None:
                raise ValueError("feature_space='pls' requires a prefit readout")
            return np.asarray(self.readout.rna.transform(values), dtype=float)
        raise ValueError(f"unsupported feature space: {self.feature_space!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family M no-batch matching and conditional transport baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--knn-feature-space", choices=("program", "pls", "rna"), default="program")
    parser.add_argument("--sinkhorn-feature-space", choices=("program", "pls", "rna"), default="program")
    parser.add_argument("--k-values", default="1,3,5,8")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn-query-epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn-iterations", type=int, default=200)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    protected_metrics, readout = protected_pls_metrics(
        dataset,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    train_records = pair_records(dataset, split="train")
    test_records = pair_records(dataset, split=args.eval_split, control_splits=_eval_control_splits(args.eval_split))
    if not train_records or not test_records:
        raise RuntimeError("Family M transport baselines require train and test counterfactual pairs")

    all_train_controls = np.concatenate([record["control_cells"] for record in train_records], axis=0)
    knn_projector = FeatureProjector(
        dataset=dataset,
        feature_space=args.knn_feature_space,
        fit_values=all_train_controls,
        readout=readout,
    )
    sinkhorn_projector = FeatureProjector(
        dataset=dataset,
        feature_space=args.sinkhorn_feature_space,
        fit_values=all_train_controls,
        readout=readout,
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

    candidates: list[dict[str, Any]] = []
    prediction, diagnostics = predict_matched_perturbed_mean(train_records, test_records)
    candidates.append(
        evaluate_predictions(
            dataset,
            test_records,
            prediction,
            candidate_name="seed2_no_batch_matched_perturbed_mean",
            method="matched_perturbed_mean",
            exact_match_fraction=diagnostics["exact_match_fraction"],
            diagnostics=diagnostics,
        )
    )

    prediction, diagnostics = predict_residualized_matching(train_records, test_records)
    residual_matching = evaluate_predictions(
        dataset,
        test_records,
        prediction,
        candidate_name="seed2_no_batch_residualized_matching",
        method="residualized_matching",
        exact_match_fraction=diagnostics["exact_match_fraction"],
        diagnostics=diagnostics,
    )
    candidates.append(residual_matching)

    for k in parse_int_list(args.k_values):
        prediction, diagnostics = predict_knn_residual_transport(
            train_records,
            test_records,
            projector=knn_projector,
            k=k,
        )
        candidates.append(
            evaluate_predictions(
                dataset,
                test_records,
                prediction,
                candidate_name=f"seed2_knn_residual_transport_{args.knn_feature_space}_k{k}",
                method="knn_residual_transport",
                exact_match_fraction=diagnostics["exact_match_fraction"],
                diagnostics={**diagnostics, "k": float(k), "feature_space": args.knn_feature_space},
            )
        )

    prediction, diagnostics = predict_sinkhorn_residual_transport(
        train_records,
        test_records,
        projector=sinkhorn_projector,
        epsilon=args.sinkhorn_epsilon,
        query_epsilon=args.sinkhorn_query_epsilon,
        iterations=args.sinkhorn_iterations,
    )
    candidates.append(
        evaluate_predictions(
            dataset,
            test_records,
            prediction,
            candidate_name=(
                f"seed2_sinkhorn_residual_transport_{args.sinkhorn_feature_space}"
                f"_eps{slug_float(args.sinkhorn_epsilon)}"
            ),
            method="sinkhorn_residual_transport",
            exact_match_fraction=diagnostics["exact_match_fraction"],
            diagnostics={**diagnostics, "feature_space": args.sinkhorn_feature_space},
        )
    )

    matching_reference = residual_matching
    rows = []
    for candidate in candidates:
        candidate["beats_residualized_matching"] = beats_matching_reference(candidate, matching_reference)
        candidate["counterfactual_gate_pass"] = counterfactual_gate_pass(candidate, source_metrics)
        candidate["protected_geometry_preserved"] = True
        candidate["protected_rna_to_image_recall@1"] = float(protected_metrics.get("model_rna_to_image_recall@1", 0.0))
        candidate["protected_bio_latent_r2_rna_shared"] = float(
            protected_metrics.get("model_bio_latent_r2_rna_shared", 0.0)
        )
        candidate["protected_representation_rank"] = float(protected_metrics.get("model_embedding_rank", 0.0))
        candidate["protected_batch_probe_balanced_accuracy"] = float(
            protected_metrics.get("model_batch_probe_balanced_accuracy", 0.0)
        )
        rows.append(candidate)

    frame = pd.DataFrame(rows)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_M_RESULTS.tsv", sep="\t", index=False)

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "eval_split": args.eval_split,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_matching": True,
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "candidates": rows,
    }
    write_json(args.output_dir / "FAMILY_M_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_json(args.output_dir / "prefit_pls_readout.json", readout.to_dict())
    write_report(args.output_dir / "REPORT.md", frame, source_metrics=source_metrics, args=args)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def protected_pls_metrics(
    dataset: SyntheticBiologyLiteDataset,
    *,
    rank: int,
    device: str,
    seed: int,
    eval_split: str = "test",
) -> tuple[dict[str, Any], Any]:
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=rank)
    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=0,
        device=device,
        dropout=0.0,
        model_dim=max(4, rank),
        shared_dim=rank,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
    )
    model = experiment_config.build_model().to(device)
    install_prefit_pls_readout(model, readout, freeze=True, device=device)
    install_prefit_pls_distillation_head(model, readout, freeze=True, device=device)
    metrics = evaluate_step0(
        dataset,
        model,
        split=eval_split,
        train_split="train",
        device=device,
        bag_size=bag_size,
        seed=seed,
        label_shuffle_repeats=20,
    )
    return metrics, readout


def pair_records(
    dataset: SyntheticBiologyLiteDataset,
    *,
    split: str,
    control_splits: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split, control_splits=control_splits):
        control_cells = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_cells = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_cells": control_cells,
                "target_cells": target_cells,
                "control_mean": control_cells.mean(axis=0),
                "target_mean": target_cells.mean(axis=0),
            }
        )
    return records


def _eval_control_splits(eval_split: str) -> tuple[str, ...] | None:
    if eval_split == "test":
        return None
    return tuple(dict.fromkeys((eval_split, "train", "test", "val")))


def biological_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in BIOLOGICAL_KEY_FIELDS)


def predict_matched_perturbed_mean(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    target_by_key = group_mean_by_key(train_records, value_name="target_mean")
    global_target = np.stack([record["target_mean"] for record in train_records]).mean(axis=0)
    predictions = []
    exact_hits = 0
    for record in test_records:
        value = target_by_key.get(record["biological_key"])
        if value is None:
            value = global_target
        else:
            exact_hits += 1
        predictions.append(value)
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "batch_id_excluded": 1.0,
    }


def predict_residualized_matching(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    deltas_by_key: dict[tuple[Any, ...], list[np.ndarray]] = {}
    for record in train_records:
        deltas_by_key.setdefault(record["biological_key"], []).append(record["target_mean"] - record["control_mean"])
    mean_delta_by_key = {key: np.stack(values).mean(axis=0) for key, values in deltas_by_key.items()}
    global_delta = np.stack([record["target_mean"] - record["control_mean"] for record in train_records]).mean(axis=0)
    predictions = []
    exact_hits = 0
    for record in test_records:
        delta = mean_delta_by_key.get(record["biological_key"])
        if delta is None:
            delta = global_delta
        else:
            exact_hits += 1
        predictions.append(record["control_mean"] + delta)
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "batch_id_excluded": 1.0,
    }


def predict_knn_residual_transport(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    projector: FeatureProjector,
    k: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    samples_by_key = residual_samples_by_key(train_records)
    predictions = []
    exact_hits = 0
    neighbor_counts = []
    neighbor_distances = []
    for record in test_records:
        samples = samples_by_key.get(record["biological_key"])
        if not samples:
            samples = [sample for values in samples_by_key.values() for sample in values]
        else:
            exact_hits += 1
        sources = np.stack([sample["source"] for sample in samples])
        residuals = np.stack([sample["residual"] for sample in samples])
        distances = squared_distances(projector.project(record["control_mean"])[0:1], projector.project(sources))[0]
        take = np.argsort(distances)[: min(int(k), len(samples))]
        predictions.append(record["control_mean"] + residuals[take].mean(axis=0))
        neighbor_counts.append(float(len(take)))
        neighbor_distances.append(float(distances[take].mean()))
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "mean_neighbor_count": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "mean_neighbor_distance": float(np.mean(neighbor_distances)) if neighbor_distances else 0.0,
        "batch_id_excluded": 1.0,
    }


def predict_sinkhorn_residual_transport(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    projector: FeatureProjector,
    epsilon: float,
    query_epsilon: float,
    iterations: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    transports_by_key = {}
    for key in sorted({record["biological_key"] for record in train_records}):
        key_records = [record for record in train_records if record["biological_key"] == key]
        controls = np.concatenate([record["control_cells"] for record in key_records], axis=0)
        targets = np.concatenate([record["target_cells"] for record in key_records], axis=0)
        control_features = projector.project(controls)
        target_features = projector.project(targets)
        plan = sinkhorn_plan(control_features, target_features, epsilon=epsilon, iterations=iterations)
        row_mass = plan.sum(axis=1, keepdims=True)
        row_mass = np.where(row_mass <= 1e-12, 1.0, row_mass)
        barycentric_target = (plan / row_mass) @ targets
        transports_by_key[key] = {
            "controls": controls,
            "control_features": control_features,
            "residuals": barycentric_target - controls,
            "plan_entropy": plan_entropy(plan),
        }

    predictions = []
    exact_hits = 0
    entropies = []
    support_sizes = []
    for record in test_records:
        transport = transports_by_key.get(record["biological_key"])
        if transport is None:
            key, transport = next(iter(transports_by_key.items()))
        else:
            exact_hits += 1
        query = projector.project(record["control_mean"])[0:1]
        distances = squared_distances(query, transport["control_features"])[0]
        weights = softmax_negative_distance(distances, epsilon=query_epsilon)
        predictions.append(record["control_mean"] + weights @ transport["residuals"])
        entropies.append(float(transport["plan_entropy"]))
        support_sizes.append(float(transport["controls"].shape[0]))
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "sinkhorn_epsilon": float(epsilon),
        "sinkhorn_query_epsilon": float(query_epsilon),
        "sinkhorn_iterations": float(iterations),
        "mean_plan_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
        "batch_id_excluded": 1.0,
    }


def residual_samples_by_key(train_records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, np.ndarray]]]:
    samples: dict[tuple[Any, ...], list[dict[str, np.ndarray]]] = {}
    for record in train_records:
        target_mean = record["target_mean"]
        key_samples = samples.setdefault(record["biological_key"], [])
        for source in record["control_cells"]:
            key_samples.append({"source": source, "residual": target_mean - source})
    return samples


def group_mean_by_key(records: list[dict[str, Any]], *, value_name: str) -> dict[tuple[Any, ...], np.ndarray]:
    grouped: dict[tuple[Any, ...], list[np.ndarray]] = {}
    for record in records:
        grouped.setdefault(record["biological_key"], []).append(np.asarray(record[value_name], dtype=float))
    return {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}


def evaluate_predictions(
    dataset: SyntheticBiologyLiteDataset,
    test_records: list[dict[str, Any]],
    predictions: list[np.ndarray],
    *,
    candidate_name: str,
    method: str,
    exact_match_fraction: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=float)
    observed = np.stack([record["target_mean"] for record in test_records])
    control = np.stack([record["control_mean"] for record in test_records])
    metadata = pd.DataFrame([{key: record[key] for key in BIOLOGICAL_KEY_FIELDS + ("batch_id",)} for record in test_records])
    metrics = rna_counterfactual_metrics(
        predicted,
        observed,
        control,
        metadata,
        groupby=None,
        topk=(50,),
        gene_sets=_gene_sets(dataset),
    )
    predicted_delta = predicted - control
    observed_delta = observed - control
    result: dict[str, Any] = {
        "candidate_name": candidate_name,
        "method": method,
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_match_fraction),
        "program_level_effect_recovery": _program_effect_recovery(
            predicted,
            observed,
            control,
            dataset.gene_program_assignment,
        ),
        "direction_accuracy": float(metrics["rna_counterfactual_direction_accuracy"]),
        "logfc_correlation": float(metrics["rna_counterfactual_logfc_correlation"]),
        "pseudobulk_correlation": float(metrics["rna_counterfactual_pseudobulk_correlation"]),
        "top50_de_overlap": float(metrics["rna_counterfactual_top50_de_overlap"]),
        "pathway_score_correlation": float(metrics.get("rna_counterfactual_pathway_score_correlation", 0.0)),
        "mean_delta_to_target_ratio": norm_ratio(predicted_delta, observed_delta),
    }
    for key, value in diagnostics.items():
        result[key] = value
    return result


def counterfactual_gate_pass(candidate: dict[str, Any], source: dict[str, Any]) -> bool:
    return bool(
        float(candidate["pseudobulk_correlation"]) >= float(source["pseudobulk_correlation"]) - 0.02
        and float(candidate["direction_accuracy"]) >= float(source["direction_accuracy"]) + 0.10
        and float(candidate["logfc_correlation"]) >= float(source["logfc_correlation"]) + 0.05
        and float(candidate["program_level_effect_recovery"]) >= float(source["program_level_effect_recovery"]) + 0.05
        and float(candidate["top50_de_overlap"]) >= float(source["top50_de_overlap"]) + 0.03
    )


def beats_matching_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return bool(
        float(candidate["program_level_effect_recovery"]) > float(reference["program_level_effect_recovery"]) + 1e-9
        and float(candidate["logfc_correlation"]) > float(reference["logfc_correlation"]) + 1e-9
        and float(candidate["top50_de_overlap"]) > float(reference["top50_de_overlap"]) + 1e-9
    )


def program_means(values: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    return np.stack([values[:, assignment == program].mean(axis=1) for program in sorted(np.unique(assignment))], axis=1)


def sinkhorn_plan(x: np.ndarray, y: np.ndarray, *, epsilon: float, iterations: int) -> np.ndarray:
    cost = squared_distances(x, y)
    positive = cost[cost > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = cost / max(scale, 1e-12)
    kernel = np.exp(-cost / max(float(epsilon), 1e-6))
    a = np.full(x.shape[0], 1.0 / max(1, x.shape[0]), dtype=float)
    b = np.full(y.shape[0], 1.0 / max(1, y.shape[0]), dtype=float)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max(1, int(iterations))):
        u = a / np.maximum(kernel @ v, 1e-12)
        v = b / np.maximum(kernel.T @ u, 1e-12)
    return (u[:, None] * kernel) * v[None, :]


def plan_entropy(plan: np.ndarray) -> float:
    values = plan[plan > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = np.asarray(x, dtype=float)[:, None, :] - np.asarray(y, dtype=float)[None, :, :]
    return np.sum(diff * diff, axis=-1)


def softmax_negative_distance(distances: np.ndarray, *, epsilon: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    positive = distances[distances > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    logits = -distances / max(float(epsilon) * scale, 1e-12)
    logits = logits - logits.max()
    weights = np.exp(logits)
    total = weights.sum()
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / max(1, weights.size), dtype=float)
    return weights / total


def norm_ratio(numerator: np.ndarray, denominator: np.ndarray, *, eps: float = 1e-8) -> float:
    num = np.linalg.norm(np.asarray(numerator, dtype=float), axis=-1).mean()
    den = max(float(np.linalg.norm(np.asarray(denominator, dtype=float), axis=-1).mean()), eps)
    return float(num / den)


def parse_int_list(values: str) -> list[int]:
    result = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not result:
        raise ValueError("at least one k value is required")
    return result


def slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, frame: pd.DataFrame, *, source_metrics: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Family M Transport Baselines",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological matching key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from matching/features: `true`",
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
        "## Candidate Results",
        "",
    ]
    for row in frame.itertuples(index=False):
        lines.extend(
            [
                f"### {row.candidate_name}",
                f"- method: `{row.method}`",
                f"- exact no-batch key coverage: `{row.exact_match_fraction:.4f}`",
                f"- program recovery: `{row.program_level_effect_recovery:.4f}`",
                f"- direction accuracy: `{row.direction_accuracy:.4f}`",
                f"- logFC correlation: `{row.logfc_correlation:.4f}`",
                f"- pseudobulk correlation: `{row.pseudobulk_correlation:.4f}`",
                f"- top50 overlap: `{row.top50_de_overlap:.4f}`",
                f"- mean delta/target ratio: `{row.mean_delta_to_target_ratio:.4f}`",
                f"- beats residualized matching: `{bool(row.beats_residualized_matching)}`",
                f"- counterfactual gate pass: `{bool(row.counterfactual_gate_pass)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Family M tests the CellOT/CINEMA-OT/scDRP/Conditional-Monge intuition on the synthetic seed-2 task without using batch features.",
            "Systema-style discipline is enforced by treating residualized matching as the baseline that transport must beat before neural transport is justified.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_M_RESULTS.tsv`",
            "- `FAMILY_M_RESULTS.json`",
            "- `generation_config.json`",
            "- `prefit_pls_readout.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
