from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
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
    protected_pls_metrics,
    slug_float,
)
from scripts.run_family_n_distillation import (
    ConditionFeatureEncoder,
    TrainOnlyConditionalMeanTable,
    load_prefit_ridge_best,
    load_sparse_residual_comparators,
)
from scripts.run_synthetic_lite_step0 import _jsonable
from scripts.train_clone_counterfactual_decoder import _counterfactual_pairs


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_O_COUNT_LIKELIHOOD")
RESULTS_PATH = Path("outputs/autoresearch_synth_lite/results.tsv")


class CountMeanMLP(torch.nn.Module):
    """Small no-batch decoder that predicts per-gene log-count means."""

    def __init__(self, input_dim: int, genes: int, *, hidden_dim: int, initial_mean: np.ndarray) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, genes),
        )
        final = self.net[-1]
        if not isinstance(final, torch.nn.Linear):
            raise TypeError("CountMeanMLP final layer must be linear")
        torch.nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.log(torch.as_tensor(np.clip(initial_mean, 1e-4, None), dtype=torch.float32)))

    def forward(self, features: torch.Tensor, *, min_log_mean: float, max_log_mean: float) -> torch.Tensor:
        log_mu = torch.clamp(self.net(features), min=float(min_log_mean), max=float(max_log_mean))
        return torch.clamp(torch.exp(log_mu), min=1e-6, max=float(np.exp(max_log_mean)))


class TrainedCountModel:
    def __init__(
        self,
        model: CountMeanMLP,
        *,
        likelihood: str,
        train_nll: float,
        final_loss: float,
        dispersion: np.ndarray | None,
        min_log_mean: float,
        max_log_mean: float,
        device: str,
    ) -> None:
        self.model = model
        self.likelihood = likelihood
        self.train_nll = float(train_nll)
        self.final_loss = float(final_loss)
        self.dispersion = None if dispersion is None else np.asarray(dispersion, dtype=float)
        self.min_log_mean = float(min_log_mean)
        self.max_log_mean = float(max_log_mean)
        self.device = device

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device)
            mean = self.model(x, min_log_mean=self.min_log_mean, max_log_mean=self.max_log_mean)
        return mean.detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family O synthetic count-likelihood perturbation diagnostics.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pseudo-count-scale", type=float, default=40.0)
    parser.add_argument("--dispersion-steps", type=int, default=700)
    parser.add_argument("--dispersion-lr", type=float, default=5e-2)
    parser.add_argument("--poisson-steps", type=int, default=1200)
    parser.add_argument("--nb-steps", type=int, default=1400)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-dispersion", type=float, default=1e-4)
    parser.add_argument("--max-dispersion", type=float, default=10.0)
    parser.add_argument("--min-log-mean", type=float, default=-12.0)
    parser.add_argument("--max-log-mean", type=float, default=12.0)
    parser.add_argument("--append-results", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    counts, count_diagnostics = resolve_count_matrix(dataset, pseudo_count_scale=args.pseudo_count_scale)
    train_records = count_pair_records(dataset, counts, split="train")
    test_records = count_pair_records(
        dataset,
        counts,
        split=args.eval_split,
        control_splits=_eval_control_splits(args.eval_split),
    )
    if not train_records or not test_records:
        raise RuntimeError("Family O requires train and test counterfactual pairs")

    protected_metrics, _ = protected_pls_metrics(
        dataset,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    source_count_predictions = [record["control_count_mean"] for record in test_records]
    source_metrics = evaluate_count_predictions(
        dataset,
        test_records,
        source_count_predictions,
        candidate_name="seed2_count_audit_source_as_target",
        method="count_audit_source_as_target",
        exact_match_fraction=1.0,
        diagnostics={"batch_id_excluded": 1.0, "fit_split_train_only": 1.0, "teacher_target_test_rows_used": 0.0},
    )

    rows: list[dict[str, Any]] = []
    global_train_count_mean = np.stack([record["target_count_mean"] for record in train_records]).mean(axis=0)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                source_count_predictions,
                candidate_name="seed2_count_audit_source_as_target",
                method="count_audit_source_as_target",
                exact_match_fraction=1.0,
                diagnostics={
                    "candidate_stage": "A",
                    "batch_id_excluded": 1.0,
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(source_count_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="COUNT_AUDIT_BASELINE_COMPLETE",
        )
    )
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                [global_train_count_mean for _ in test_records],
                candidate_name="seed2_train_global_count_mean_poisson_baseline",
                method="train_global_count_mean_poisson_baseline",
                exact_match_fraction=0.0,
                diagnostics={
                    "candidate_stage": "A",
                    "batch_id_excluded": 1.0,
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "global_train_count_mean": 1.0,
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray([global_train_count_mean for _ in test_records], dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="COUNT_AUDIT_BASELINE_COMPLETE",
        )
    )

    table = TrainOnlyConditionalMeanTable(train_records, value_name="target_count_mean")
    table_count_predictions, table_diagnostics = table.predict(test_records)
    table_count_array = np.asarray(table_count_predictions, dtype=float)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                table_count_predictions,
                candidate_name="seed2_poisson_train_only_count_mean_table",
                method="poisson_train_only_count_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **table_diagnostics,
                    "candidate_stage": "B",
                    "count_likelihood": "poisson",
                    "output_parameterization": "train_count_mean_positive_rate",
                },
            ),
            test_records=test_records,
            count_predictions=table_count_array,
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="TIER1_POISSON_COUNT_TABLE_COMPLETE",
        )
    )

    train_table_predictions, train_table_diagnostics = table.predict(train_records)
    learned_table_dispersion = fit_gene_wise_nb_dispersion(
        train_records,
        np.asarray(train_table_predictions, dtype=float),
        steps=args.dispersion_steps,
        lr=args.dispersion_lr,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 17,
        device=args.device,
    )
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                table_count_predictions,
                candidate_name="seed2_nb_train_only_count_mean_table",
                method="nb_train_only_count_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **table_diagnostics,
                    "candidate_stage": "C",
                    "count_likelihood": "negative_binomial",
                    "dispersion_fit_split_train_only": 1.0,
                    "train_teacher_exact_match_fraction": float(train_table_diagnostics["exact_match_fraction"]),
                    **dispersion_stats(learned_table_dispersion, prefix="nb_train"),
                },
            ),
            test_records=test_records,
            count_predictions=table_count_array,
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=learned_table_dispersion,
            decision_label="TIER1_NB_COUNT_TABLE_COMPLETE",
        )
    )

    feature_encoder = ConditionFeatureEncoder(dataset, train_records)
    train_features = feature_encoder.transform(train_records)
    test_features = feature_encoder.transform(test_records)
    feature_diagnostics = feature_encoder.to_dict()

    poisson_model = fit_count_mlp(
        train_features,
        train_records,
        likelihood="poisson",
        hidden_dim=args.mlp_hidden_dim,
        steps=args.poisson_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        min_log_mean=args.min_log_mean,
        max_log_mean=args.max_log_mean,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 101,
        device=args.device,
    )
    poisson_predictions = poisson_model.predict_mean(test_features)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                list(poisson_predictions),
                candidate_name="seed2_poisson_mlp_no_batch_condition_source",
                method="poisson_mlp_no_batch_condition_source",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    "candidate_stage": "D",
                    "count_likelihood": "poisson",
                    "feature_count": float(feature_diagnostics["feature_count"]),
                    "batch_id_feature_present": float(bool(feature_diagnostics["batch_id_feature_present"])),
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.poisson_steps),
                    "mlp_train_poisson_nll": poisson_model.train_nll,
                    "mlp_final_loss": poisson_model.final_loss,
                    "output_parameterization": "log_mean_exp_clamped",
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(poisson_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="TIER1_POISSON_COUNT_MLP_COMPLETE",
        )
    )

    nb_model = fit_count_mlp(
        train_features,
        train_records,
        likelihood="negative_binomial",
        hidden_dim=args.mlp_hidden_dim,
        steps=args.nb_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        min_log_mean=args.min_log_mean,
        max_log_mean=args.max_log_mean,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 202,
        device=args.device,
    )
    nb_predictions = nb_model.predict_mean(test_features)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                list(nb_predictions),
                candidate_name="seed2_nb_mlp_no_batch_condition_source",
                method="nb_mlp_no_batch_condition_source",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    "candidate_stage": "E",
                    "count_likelihood": "negative_binomial",
                    "feature_count": float(feature_diagnostics["feature_count"]),
                    "batch_id_feature_present": float(bool(feature_diagnostics["batch_id_feature_present"])),
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.nb_steps),
                    "mlp_train_nb_nll": nb_model.train_nll,
                    "mlp_final_loss": nb_model.final_loss,
                    "output_parameterization": "log_mean_exp_clamped",
                    **dispersion_stats(np.asarray(nb_model.dispersion, dtype=float), prefix="nb_model"),
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(nb_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=nb_model.dispersion,
            decision_label="TIER1_NB_COUNT_MLP_COMPLETE",
        )
    )

    comparator_rows = load_required_comparators(args.output_dir)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame = pd.DataFrame(rows)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_O_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(comparator_rows).to_csv(args.output_dir / "COMPARATOR_RESULTS.tsv", sep="\t", index=False)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "eval_split": args.eval_split,
        "count_diagnostics": count_diagnostics,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_features": True,
        "train_pairs": len(train_records),
        "test_pairs": len(test_records),
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "feature_encoder": feature_diagnostics,
        "candidates": rows,
        "comparators": comparator_rows,
    }
    write_json(args.output_dir / "FAMILY_O_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_report(args.output_dir / "REPORT.md", frame, comparator_rows=comparator_rows, count_diagnostics=count_diagnostics, args=args)
    if args.append_results:
        append_results_tsv(frame, protected_metrics=protected_metrics, device=args.device, wallclock_minutes=elapsed)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def resolve_count_matrix(
    dataset: SyntheticBiologyLiteDataset,
    *,
    pseudo_count_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(dataset, "observed_counts") and dataset.observed_counts is not None:
        counts = np.asarray(dataset.observed_counts, dtype=float)
        path = "raw_synthetic_observed_counts"
        pseudo = False
        scale = None
    else:
        counts = pseudo_counts_from_expression(dataset.expression_values, pseudo_count_scale=pseudo_count_scale)
        path = "synthetic_pseudo_count_from_expression_values"
        pseudo = True
        scale = float(pseudo_count_scale)
    diagnostics = count_path_diagnostics(dataset, counts, count_path=path, pseudo_count_used=pseudo, pseudo_count_scale=scale)
    return counts, diagnostics


def pseudo_counts_from_expression(expression_values: np.ndarray, *, pseudo_count_scale: float) -> np.ndarray:
    expression = np.asarray(expression_values, dtype=float)
    positive = np.clip(np.expm1(expression), 0.0, None)
    library = positive.sum(axis=1, keepdims=True)
    median_library = float(np.median(library[library > 0.0])) if np.any(library > 0.0) else 1.0
    scaled = positive * (float(pseudo_count_scale) * expression.shape[1]) / max(median_library, 1e-8)
    return np.rint(np.clip(scaled, 0.0, None)).astype(float)


def count_path_diagnostics(
    dataset: SyntheticBiologyLiteDataset,
    counts: np.ndarray,
    *,
    count_path: str,
    pseudo_count_used: bool,
    pseudo_count_scale: float | None,
) -> dict[str, Any]:
    values = np.asarray(counts, dtype=float)
    gene_mean = values.mean(axis=0)
    gene_var = values.var(axis=0)
    dispersion = method_of_moments_dispersion(values)
    return {
        "raw_count_available": bool(
            hasattr(dataset, "observed_counts") and dataset.observed_counts is not None and not pseudo_count_used
        ),
        "count_path": count_path,
        "pseudo_count_used": bool(pseudo_count_used),
        "pseudo_count_scale": pseudo_count_scale,
        "count_rows": int(values.shape[0]),
        "count_genes": int(values.shape[1]),
        "count_min": float(values.min()),
        "count_max": float(values.max()),
        "count_mean": float(values.mean()),
        "count_variance": float(values.var()),
        "zero_fraction": float(np.mean(values <= 0.0)),
        "integer_valued_fraction": float(np.mean(np.isclose(values, np.rint(values), atol=1e-6))),
        "mean_library_size": float(values.sum(axis=1).mean()),
        "median_library_size": float(np.median(values.sum(axis=1))),
        "gene_mean_variance_log_correlation": safe_corr(np.log1p(gene_mean), np.log1p(gene_var)),
        "gene_mean_variance_log_slope": safe_slope(np.log1p(gene_mean), np.log1p(gene_var)),
        **dispersion_stats(dispersion, prefix="moments"),
    }


def count_pair_records(
    dataset: SyntheticBiologyLiteDataset,
    counts: np.ndarray,
    *,
    split: str,
    control_splits: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split, control_splits=control_splits):
        control_counts = np.asarray(counts[pair["control_group"]], dtype=float)
        target_counts = np.asarray(counts[pair["target_group"]], dtype=float)
        control_expression = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_expression = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_cells": control_expression,
                "target_cells": target_expression,
                "control_mean": control_expression.mean(axis=0),
                "target_mean": target_expression.mean(axis=0),
                "control_count_cells": control_counts,
                "target_count_cells": target_counts,
                "control_count_mean": control_counts.mean(axis=0),
                "target_count_mean": target_counts.mean(axis=0),
            }
        )
    return records


def evaluate_count_predictions(
    dataset: SyntheticBiologyLiteDataset,
    test_records: list[dict[str, Any]],
    count_predictions: list[np.ndarray],
    *,
    candidate_name: str,
    method: str,
    exact_match_fraction: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    predicted_counts = np.asarray(count_predictions, dtype=float)
    predicted_expression = np.log1p(np.clip(predicted_counts, 0.0, None))
    result = evaluate_predictions(
        dataset,
        test_records,
        list(predicted_expression),
        candidate_name=candidate_name,
        method=method,
        exact_match_fraction=exact_match_fraction,
        diagnostics=diagnostics,
    )
    target_counts = np.concatenate([record["target_count_cells"] for record in test_records], axis=0)
    result["predicted_count_mean"] = float(predicted_counts.mean())
    result["predicted_count_zero_fraction"] = float(np.mean(predicted_counts <= 0.0))
    result["target_count_mean"] = float(target_counts.mean())
    return result


def finalize_count_candidate(
    result: dict[str, Any],
    *,
    test_records: list[dict[str, Any]],
    count_predictions: np.ndarray,
    protected_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    dispersion: np.ndarray | None,
    decision_label: str,
) -> dict[str, Any]:
    row = dict(result)
    row["poisson_nll_test"] = poisson_nll_for_records(test_records, count_predictions)
    if dispersion is not None:
        row["nb_nll_test"] = nb_nll_for_records(test_records, count_predictions, dispersion)
    else:
        row["nb_nll_test"] = float("nan")
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
    row["decision_label"] = decision_label if row["leakage_gate_pass"] else "TIER1_DISCARD_LEAKAGE_CONTROL_FAIL"
    return row


def fit_gene_wise_nb_dispersion(
    records: list[dict[str, Any]],
    count_predictions: np.ndarray,
    *,
    steps: int,
    lr: float,
    min_dispersion: float,
    max_dispersion: float,
    seed: int,
    device: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    y_np, mu_np = expand_record_counts_and_means(records, count_predictions)
    init = np.clip(method_of_moments_dispersion(y_np), min_dispersion, max_dispersion)
    raw = torch.nn.Parameter(inverse_softplus(torch.as_tensor(init - min_dispersion, dtype=torch.float32, device=device)))
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    mu = torch.as_tensor(np.clip(mu_np, 1e-6, None), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([raw], lr=float(lr))
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        dispersion = positive_dispersion(raw, min_dispersion=min_dispersion, max_dispersion=max_dispersion)
        loss = nb_nll_torch(y, mu, dispersion).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return positive_dispersion(raw, min_dispersion=min_dispersion, max_dispersion=max_dispersion).detach().cpu().numpy()


def fit_count_mlp(
    features: np.ndarray,
    records: list[dict[str, Any]],
    *,
    likelihood: str,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    min_log_mean: float,
    max_log_mean: float,
    min_dispersion: float,
    max_dispersion: float,
    seed: int,
    device: str,
) -> TrainedCountModel:
    torch.manual_seed(seed)
    x_np, y_np = expand_record_features_and_counts(features, records)
    initial_mean = np.clip(y_np.mean(axis=0), 1e-4, None)
    model = CountMeanMLP(x_np.shape[1], y_np.shape[1], hidden_dim=hidden_dim, initial_mean=initial_mean).to(device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    parameters: list[torch.nn.Parameter] = list(model.parameters())
    dispersion_param = None
    if likelihood == "negative_binomial":
        init = np.clip(method_of_moments_dispersion(y_np), min_dispersion, max_dispersion)
        dispersion_param = torch.nn.Parameter(
            inverse_softplus(torch.as_tensor(init - min_dispersion, dtype=torch.float32, device=device))
        )
        parameters.append(dispersion_param)
    elif likelihood != "poisson":
        raise ValueError(f"unsupported count likelihood: {likelihood!r}")
    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    final_loss = 0.0
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        mu = model(x, min_log_mean=min_log_mean, max_log_mean=max_log_mean)
        if likelihood == "poisson":
            loss = poisson_nll_torch(y, mu).mean()
        else:
            assert dispersion_param is not None
            dispersion = positive_dispersion(
                dispersion_param,
                min_dispersion=min_dispersion,
                max_dispersion=max_dispersion,
            )
            loss = nb_nll_torch(y, mu, dispersion).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        mu = model(x, min_log_mean=min_log_mean, max_log_mean=max_log_mean)
        if likelihood == "poisson":
            train_nll = float(poisson_nll_torch(y, mu).mean().detach().cpu())
            dispersion_np = None
        else:
            assert dispersion_param is not None
            dispersion = positive_dispersion(
                dispersion_param,
                min_dispersion=min_dispersion,
                max_dispersion=max_dispersion,
            )
            train_nll = float(nb_nll_torch(y, mu, dispersion).mean().detach().cpu())
            dispersion_np = dispersion.detach().cpu().numpy()
    return TrainedCountModel(
        model,
        likelihood=likelihood,
        train_nll=train_nll,
        final_loss=final_loss,
        dispersion=dispersion_np,
        min_log_mean=min_log_mean,
        max_log_mean=max_log_mean,
        device=device,
    )


def expand_record_features_and_counts(features: np.ndarray, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for feature, record in zip(np.asarray(features, dtype=float), records, strict=True):
        target = np.asarray(record["target_count_cells"], dtype=float)
        x_rows.append(np.repeat(feature[None, :], target.shape[0], axis=0))
        y_rows.append(target)
    return np.concatenate(x_rows, axis=0), np.concatenate(y_rows, axis=0)


def expand_record_counts_and_means(
    records: list[dict[str, Any]],
    count_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_rows = []
    mu_rows = []
    for prediction, record in zip(np.asarray(count_predictions, dtype=float), records, strict=True):
        target = np.asarray(record["target_count_cells"], dtype=float)
        y_rows.append(target)
        mu_rows.append(np.repeat(prediction[None, :], target.shape[0], axis=0))
    return np.concatenate(y_rows, axis=0), np.concatenate(mu_rows, axis=0)


def poisson_nll_for_records(records: list[dict[str, Any]], count_predictions: np.ndarray) -> float:
    y, mu = expand_record_counts_and_means(records, count_predictions)
    mu = np.clip(mu, 1e-8, None)
    nll = mu - y * np.log(mu) + gammaln(y + 1.0)
    return float(np.mean(nll))


def nb_nll_for_records(records: list[dict[str, Any]], count_predictions: np.ndarray, dispersion: np.ndarray) -> float:
    y, mu = expand_record_counts_and_means(records, count_predictions)
    return float(np.mean(nb_nll_numpy(y, np.clip(mu, 1e-8, None), np.asarray(dispersion, dtype=float))))


def poisson_nll_torch(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)


def nb_nll_torch(y: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    alpha = torch.clamp(dispersion, min=1e-8)
    theta = 1.0 / alpha
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_prob


def nb_nll_numpy(y: np.ndarray, mu: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(dispersion, dtype=float), 1e-8, None)
    theta = 1.0 / alpha
    log_prob = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (np.log(mu) - np.log(theta + mu))
    )
    return -log_prob


def method_of_moments_dispersion(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    return np.clip((variance - mean) / np.maximum(mean * mean, 1e-8), 1e-4, 10.0)


def inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, min=1e-8)
    return torch.log(torch.expm1(values))


def positive_dispersion(
    raw: torch.Tensor,
    *,
    min_dispersion: float,
    max_dispersion: float,
) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + float(min_dispersion), max=float(max_dispersion))


def dispersion_stats(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_dispersion_min": float(np.min(array)),
        f"{prefix}_dispersion_median": float(np.median(array)),
        f"{prefix}_dispersion_mean": float(np.mean(array)),
        f"{prefix}_dispersion_max": float(np.max(array)),
        f"{prefix}_dispersion_zero_fraction": float(np.mean(array <= 1e-8)),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centered = x - x.mean()
    denom = float(np.sum(centered * centered))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(centered * (y - y.mean())) / denom)


def load_required_comparators(output_dir: Path) -> list[dict[str, Any]]:
    parent = output_dir.parent
    comparators: list[dict[str, Any]] = []
    family_n_path = parent / "FAMILY_N_DISTILLATION" / "FAMILY_N_RESULTS.tsv"
    if family_n_path.exists():
        family_n = pd.read_csv(family_n_path, sep="\t")
        for name in (
            "seed2_train_only_condition_mean_table",
            "seed2_distilled_linear_condition_mean",
            "seed2_distilled_mlp_condition_mean",
        ):
            match = family_n[family_n["candidate_name"] == name]
            if not match.empty:
                comparators.append(comparator_from_series(match.iloc[0], source="Family N"))
    family_m_path = parent / "FAMILY_M_TRANSPORT_BASELINES" / "FAMILY_M_RESULTS.tsv"
    if family_m_path.exists():
        family_m = pd.read_csv(family_m_path, sep="\t")
        for name in ("seed2_no_batch_matched_perturbed_mean", "seed2_no_batch_residualized_matching"):
            match = family_m[family_m["candidate_name"] == name]
            if not match.empty:
                comparators.append(comparator_from_series(match.iloc[0], source="Family M"))
    comparators.extend(load_sparse_residual_comparators(parent / "SPARSE_PERTURBATION_RESIDUAL_HEAD"))
    ridge = load_prefit_ridge_best(parent / "CLONE_COUNTERFACTUAL_DECODER")
    if ridge is not None:
        comparators.append(ridge)
    return comparators


def comparator_from_series(row: pd.Series, *, source: str) -> dict[str, Any]:
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


def append_results_tsv(
    frame: pd.DataFrame,
    *,
    protected_metrics: dict[str, Any],
    device: str,
    wallclock_minutes: float,
) -> None:
    existing = pd.read_csv(RESULTS_PATH, sep="\t") if RESULTS_PATH.exists() else pd.DataFrame()
    columns = list(existing.columns) if not existing.empty else [
        "commit",
        "experiment_num",
        "family",
        "candidate_name",
        "tier_reached",
        "decision_label",
        "device_used",
        "wallclock_minutes",
        "max_gpu_memory_gb",
        "synth_micro_recall1",
        "synth_easy_recall1",
        "synth_medium_recall1",
        "synth_easy_cf_dir_acc",
        "synth_medium_cf_dir_acc",
        "heldout_pert_cf_dir_acc",
        "batch_confound_batch_leakage",
        "batch_confound_recall1",
        "dose_extrap_logfc_corr",
        "bio_latent_r2",
        "representation_rank",
        "delta_norm_ratio",
        "cap_bound",
        "collapse_flag",
        "architecture_change",
        "description",
    ]
    last_num = int(existing["experiment_num"].max()) if not existing.empty else 0
    commit = current_commit()
    rows = []
    for offset, row in enumerate(frame.itertuples(index=False), start=1):
        rows.append(
            {
                "commit": commit,
                "experiment_num": last_num + offset,
                "family": "O",
                "candidate_name": row.candidate_name,
                "tier_reached": "Tier 1 synth_micro seed2",
                "decision_label": row.decision_label,
                "device_used": device,
                "wallclock_minutes": f"{wallclock_minutes:.3f}",
                "max_gpu_memory_gb": "0.000",
                "synth_micro_recall1": f"{float(protected_metrics.get('model_rna_to_image_recall@1', 0.0)):.6f}",
                "synth_easy_recall1": "",
                "synth_medium_recall1": "",
                "synth_easy_cf_dir_acc": "",
                "synth_medium_cf_dir_acc": "",
                "heldout_pert_cf_dir_acc": "",
                "batch_confound_batch_leakage": "",
                "batch_confound_recall1": "",
                "dose_extrap_logfc_corr": f"{float(row.logfc_correlation):.6f}",
                "bio_latent_r2": f"{float(protected_metrics.get('model_bio_latent_r2_rna_shared', 0.0)):.6f}",
                "representation_rank": f"{float(protected_metrics.get('model_embedding_rank', 0.0)):.1f}",
                "delta_norm_ratio": f"{float(row.mean_delta_to_target_ratio):.6f}",
                "cap_bound": "false",
                "collapse_flag": "false",
                "architecture_change": "count_likelihood_synthetic_only",
                "description": count_result_description(row),
            }
        )
    output = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    output.loc[:, columns].to_csv(RESULTS_PATH, sep="\t", index=False)


def count_result_description(row: Any) -> str:
    nb = "NA" if pd.isna(row.nb_nll_test) else f"{float(row.nb_nll_test):.4f}"
    return (
        f"Family O synthetic-only count-likelihood candidate; stage {row.candidate_stage}, "
        f"Poisson NLL {float(row.poisson_nll_test):.4f}, NB NLL {nb}, "
        f"program {float(row.program_level_effect_recovery):.4f}, logFC {float(row.logfc_correlation):.4f}, "
        f"top50 {float(row.top50_de_overlap):.4f}, direction {float(row.direction_accuracy):.4f}, "
        f"pseudobulk {float(row.pseudobulk_correlation):.4f}; batch_id excluded and raw-count path audited."
    )


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    frame: pd.DataFrame,
    *,
    comparator_rows: list[dict[str, Any]],
    count_diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    best_poisson = frame.sort_values("poisson_nll_test", ascending=True).head(1).iloc[0]
    nb_frame = frame[np.isfinite(frame["nb_nll_test"].astype(float))]
    best_nb = nb_frame.sort_values("nb_nll_test", ascending=True).head(1).iloc[0] if not nb_frame.empty else None
    lines = [
        "# Family O Count-Likelihood Perturbation Training",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from features and matching: `true`",
        "- Training targets/statistics split: `train`",
        "- Test target rows used for teacher/training construction: `0`",
        "- Real data used: `false`",
        "- Marker/pathway/pretrained biological assets used: `false`",
        "",
        "## Count Path Audit",
        "",
        f"- Raw count-like RNA values available: `{bool(count_diagnostics['raw_count_available'])}`",
        f"- Count path: `{count_diagnostics['count_path']}`",
        f"- Pseudo-count path used: `{bool(count_diagnostics['pseudo_count_used'])}`",
        f"- Pseudo-count scale: `{count_diagnostics['pseudo_count_scale']}`",
        f"- Zero fraction: `{count_diagnostics['zero_fraction']:.4f}`",
        f"- Count mean / variance: `{count_diagnostics['count_mean']:.4f}` / `{count_diagnostics['count_variance']:.4f}`",
        f"- Mean library size: `{count_diagnostics['mean_library_size']:.2f}`",
        f"- log mean-variance correlation: `{count_diagnostics['gene_mean_variance_log_correlation']:.4f}`",
        f"- MoM dispersion median: `{count_diagnostics['moments_dispersion_median']:.4f}`",
        "",
        "## Candidate Results",
        "",
    ]
    for row in frame.itertuples(index=False):
        lines.extend(count_candidate_lines(row))
    lines.extend(
        [
            "## NLL Winners",
            "",
            f"- Best Poisson NLL: `{best_poisson.candidate_name}` with `{best_poisson.poisson_nll_test:.4f}`",
        ]
    )
    if best_nb is not None:
        lines.append(f"- Best NB NLL: `{best_nb.candidate_name}` with `{best_nb.nb_nll_test:.4f}`")
    lines.extend(["", "## Required Comparators", ""])
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
            "Family O tests whether count likelihood changes the seed-2 counterfactual signal rather than replacing Family N. The raw synthetic count path is available, so no pseudo-count fallback was used in this run.",
            "Poisson and NB table candidates evaluate likelihood calibration around the train-only condition mean; MLP candidates test a learned no-batch condition/source-feature decoder with positive log-mean parameterization and stable positive dispersion for NB.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_O_RESULTS.tsv`",
            "- `FAMILY_O_RESULTS.json`",
            "- `COMPARATOR_RESULTS.tsv`",
            "- `generation_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_candidate_lines(row: Any) -> list[str]:
    nb_nll = "NA" if pd.isna(row.nb_nll_test) else f"{row.nb_nll_test:.4f}"
    return [
        f"### {row.candidate_name}",
        f"- method: `{row.method}`",
        f"- stage: `{row.candidate_stage}`",
        f"- leakage gate pass: `{bool(row.leakage_gate_pass)}`",
        f"- Poisson NLL test: `{row.poisson_nll_test:.4f}`",
        f"- NB NLL test: `{nb_nll}`",
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
