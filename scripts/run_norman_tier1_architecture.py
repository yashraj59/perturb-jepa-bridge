from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import (  # noqa: E402
    DEFAULT_NORMAN_H5AD,
    add_gears_simulation_split,
    is_combo,
    is_single,
    load_norman2019_condition_data,
    perturbation_genes,
)
from perturb_jepa.models.norman_compositional import (  # noqa: E402
    NormanAdditiveInteraction,
    NormanFeatureBridge,
    NormanFeatureBridgeConfig,
    NormanInteractionConfig,
    interaction_ratio_loss,
    norm_ratio,
)
from scripts.run_norman_step0 import evaluate_predictions  # noqa: E402


RUN_DIR = Path("outputs/autoresearch_norman_v1")
ARCH_DIRNAME = "tier1_architecture"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Norman v1 Tier 1 architecture smoke experiments.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_NORMAN_H5AD)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _select_device(args.device)
    output_dir = args.output_dir
    arch_dir = output_dir / ARCH_DIRNAME
    arch_dir.mkdir(parents=True, exist_ok=True)

    dataset = add_gears_simulation_split(load_norman2019_condition_data(args.h5ad), seed=args.split_seed)
    if dataset.split is None:
        raise RuntimeError("Norman split was not attached")
    split = dataset.split
    train_conditions = [condition for condition in split.train_conditions if condition != dataset.ctrl_condition]
    train_subset = _subsample_train_conditions(train_conditions, fraction=args.train_fraction, seed=args.seed)
    features = NormanFeatureFactory(dataset, train_reference_conditions=train_conditions)
    standardizer = FeatureStandardizer(np.stack([features.condition_feature(condition) for condition in train_subset]))
    additive = TrainSingleAdditive(dataset)
    baselines = _load_step0_gates(output_dir)
    primary_gate = max(baselines["family_n_condition_mean_table"], baselines["single_perturbation_additive"]) + 0.02

    experiments: list[dict[str, Any]] = [
        {
            "name": "A1_feature_bridge_mlp",
            "family": "Family A",
            "title": "Feature-conditioned perturbation bridge",
            "runner": "feature_bridge",
            "operator_rank": 0,
        },
        {
            "name": "A2_feature_bridge_rank2_operator",
            "family": "Family A",
            "title": "Feature-conditioned low-rank operator bridge",
            "runner": "feature_bridge",
            "operator_rank": 2,
        },
        {
            "name": "B1_pure_additive_architecture",
            "family": "Family B",
            "title": "Pure additive composition architecture",
            "runner": "pure_additive",
        },
        {
            "name": "B2_additive_bounded_interaction_mlp",
            "family": "Family B",
            "title": "Additive plus bounded MLP interaction",
            "runner": "interaction",
            "rank": 0,
        },
        {
            "name": "B3_additive_low_rank_interaction",
            "family": "Family B",
            "title": "Additive plus low-rank interaction",
            "runner": "interaction",
            "rank": 4,
        },
    ]

    all_aggregate_rows: list[dict[str, Any]] = []
    all_condition_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    journal_entries: list[str] = []
    architecture_entries: list[str] = []
    consecutive_discards = 0
    stop_fired = False
    commit = _git_commit()
    start_experiment_num = _next_experiment_num(output_dir / "results.tsv")

    for offset, experiment in enumerate(experiments):
        experiment_num = start_experiment_num + offset
        experiment_started = time.perf_counter()
        if experiment["runner"] == "feature_bridge":
            payload = _run_feature_bridge(
                dataset,
                train_subset,
                features,
                standardizer,
                experiment,
                args,
                device=device,
            )
        elif experiment["runner"] == "pure_additive":
            payload = _predict_pure_additive(dataset, additive)
        elif experiment["runner"] == "interaction":
            payload = _run_interaction(
                dataset,
                train_subset,
                features,
                standardizer,
                additive,
                experiment,
                args,
                device=device,
            )
        else:
            raise ValueError(f"unknown runner: {experiment['runner']}")

        condition_rows, aggregate_rows = evaluate_predictions(dataset, experiment["name"], args.split_seed, payload)
        exact = _aggregate_row(aggregate_rows, "exact_train_combo")
        unseen = _aggregate_row(aggregate_rows, "unseen_single")
        status, hard_fail = _decision(exact, payload, primary_gate)
        if status.startswith("TIER1_DISCARD"):
            consecutive_discards += 1
        else:
            consecutive_discards = 0
        elapsed = time.perf_counter() - experiment_started
        result_rows.append(
            _result_row(
                commit,
                experiment_num,
                experiment,
                status,
                exact,
                unseen,
                payload,
            )
        )
        journal_entries.append(
            _journal_entry(
                experiment_num,
                experiment,
                status,
                exact,
                unseen,
                payload,
                primary_gate=primary_gate,
                elapsed=elapsed,
                hard_fail=hard_fail,
            )
        )
        architecture_entries.append(_architecture_entry(experiment_num, experiment, payload, status))
        all_condition_rows.extend(condition_rows)
        all_aggregate_rows.extend(aggregate_rows)
        if consecutive_discards >= 5:
            stop_fired = True
            break

    aggregate_df = pd.DataFrame(all_aggregate_rows)
    condition_df = pd.DataFrame(all_condition_rows)
    result_df = pd.DataFrame(result_rows)
    aggregate_df.to_csv(arch_dir / "tier1_metrics.tsv", sep="\t", index=False)
    condition_df.to_csv(arch_dir / "tier1_per_condition_metrics.tsv", sep="\t", index=False)
    result_df.to_csv(arch_dir / "tier1_results.tsv", sep="\t", index=False)
    _append_results(output_dir / "results.tsv", result_df)
    _append_text(output_dir / "research_journal.md", "\n\n".join(journal_entries) + "\n")
    _append_text(output_dir / "architectural_changes_log.md", "\n\n".join(architecture_entries) + "\n")
    _write_family_allocation(output_dir, result_rows, stop_fired)
    _write_tier1_summary(
        arch_dir,
        args,
        device=device,
        baselines=baselines,
        primary_gate=primary_gate,
        result_df=result_df,
        aggregate_df=aggregate_df,
        elapsed=time.perf_counter() - started,
        stop_fired=stop_fired,
    )
    if stop_fired:
        _write_final_report(output_dir, result_df, aggregate_df, primary_gate)
    print(f"Wrote Norman Tier 1 architecture results to {arch_dir}")
    if stop_fired:
        print("Stop condition fired: 5 consecutive Tier 1 discards.")


class FeatureStandardizer:
    def __init__(self, train_features: np.ndarray) -> None:
        values = np.asarray(train_features, dtype=np.float32)
        self.mean = values.mean(axis=0)
        self.std = np.where(values.std(axis=0) < 1e-6, 1.0, values.std(axis=0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std


class NormanFeatureFactory:
    def __init__(self, dataset: Any, *, train_reference_conditions: list[str]) -> None:
        self.dataset = dataset
        self.reference_conditions = tuple(train_reference_conditions)
        self.gene_name_to_index = {name: idx for idx, name in enumerate(dataset.gene_names)}
        ref_indices = [dataset.condition_to_index[condition] for condition in self.reference_conditions]
        profiles = dataset.condition_means[ref_indices].T.astype(np.float32)
        self.profile_mean = profiles.mean(axis=0, keepdims=True)
        self.profile_std = np.where(profiles.std(axis=0, keepdims=True) < 1e-6, 1.0, profiles.std(axis=0, keepdims=True))
        self.standardized_profiles = (profiles - self.profile_mean) / self.profile_std
        self.feature_dim = int(self.standardized_profiles.shape[1] + 3)

    def condition_feature(self, condition: str) -> np.ndarray:
        genes = perturbation_genes(condition)
        if not genes:
            profile = np.zeros(self.standardized_profiles.shape[1], dtype=np.float32)
            missing = 0
        else:
            rows = []
            missing = 0
            for gene in genes:
                index = self.gene_name_to_index.get(gene)
                if index is None:
                    rows.append(np.zeros(self.standardized_profiles.shape[1], dtype=np.float32))
                    missing += 1
                else:
                    rows.append(self.standardized_profiles[index])
            profile = np.stack(rows).mean(axis=0)
        extras = np.asarray(
            [
                float(len(genes)),
                float(missing / max(1, len(genes))),
                float(len(genes) == 2),
            ],
            dtype=np.float32,
        )
        return np.concatenate((profile.astype(np.float32), extras), axis=0)

    def component_features(self, condition: str) -> tuple[np.ndarray, np.ndarray]:
        genes = perturbation_genes(condition)
        if not genes:
            zero = self.condition_feature("ctrl")
            return zero, zero
        first = self.condition_feature(genes[0])
        second = self.condition_feature(genes[1]) if len(genes) > 1 else np.zeros_like(first)
        return first, second


class TrainSingleAdditive:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        if dataset.split is None:
            raise RuntimeError("NormanDataset requires a split")
        self.delta_by_gene: dict[str, np.ndarray] = {}
        for condition in dataset.split.train_conditions:
            if is_single(condition):
                self.delta_by_gene[perturbation_genes(condition)[0]] = dataset.delta_for(condition)

    def delta_for(self, condition: str) -> np.ndarray:
        genes = perturbation_genes(condition)
        if not genes:
            return np.zeros_like(self.dataset.ctrl_mean)
        deltas = [self.delta_by_gene.get(gene) for gene in genes]
        if any(delta is None for delta in deltas):
            return np.zeros_like(self.dataset.ctrl_mean)
        return np.stack([delta for delta in deltas if delta is not None]).sum(axis=0).astype(np.float32)


def _run_feature_bridge(
    dataset: Any,
    train_subset: list[str],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
    experiment: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    config = NormanFeatureBridgeConfig(
        gene_dim=len(dataset.gene_ids),
        feature_dim=features.feature_dim,
        hidden_dim=args.hidden_dim,
        operator_rank=int(experiment.get("operator_rank", 0)),
        dropout=0.0,
    )
    model = NormanFeatureBridge(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_arrays = _condition_arrays(dataset, train_subset, features, standardizer)
    history = _train_feature_bridge(model, optimizer, train_arrays, args, device=device)
    predictions, diagnostics = _predict_feature_bridge(model, dataset, dataset.split.test_conditions, features, standardizer, device=device)
    diagnostics.update(
        {
            "model_config": asdict(config),
            "train_conditions": len(train_subset),
            "train_loss_final": history[-1]["loss"],
            "train_loss_initial": history[0]["loss"],
            "device": str(device),
            "description": experiment["title"],
        }
    )
    return {"predictions": predictions, "fit_metadata": diagnostics}


def _train_feature_bridge(
    model: NormanFeatureBridge,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    control = torch.as_tensor(arrays["control"], dtype=torch.float32, device=device)
    target = torch.as_tensor(arrays["target"], dtype=torch.float32, device=device)
    feature = torch.as_tensor(arrays["feature"], dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)
    history = []
    for _ in range(args.steps):
        indices = rng.choice(control.shape[0], size=min(args.batch_size, control.shape[0]), replace=False)
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        output = model(control[idx], feature[idx])
        loss = F.mse_loss(output["prediction"], target[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            ratio = norm_ratio(output["operator_delta"], output["additive_delta"]).mean()
        history.append({"loss": float(loss.detach().cpu()), "operator_ratio": float(ratio.detach().cpu())})
    return history


@torch.no_grad()
def _predict_feature_bridge(
    model: NormanFeatureBridge,
    dataset: Any,
    conditions: list[str] | tuple[str, ...],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    arrays = _condition_arrays(dataset, list(conditions), features, standardizer)
    control = torch.as_tensor(arrays["control"], dtype=torch.float32, device=device)
    feature = torch.as_tensor(arrays["feature"], dtype=torch.float32, device=device)
    output = model(control, feature)
    predictions = {
        condition: output["prediction"][idx].detach().cpu().numpy().astype(np.float32)
        for idx, condition in enumerate(conditions)
    }
    additive_ratio = float(norm_ratio(output["operator_delta"], output["additive_delta"]).mean().detach().cpu())
    raw_ratio = additive_ratio
    gate_mean = float(output["operator_gate"].mean().detach().cpu())
    return predictions, {
        "operator_to_additive_ratio_raw": raw_ratio,
        "operator_to_additive_ratio_post_gate": additive_ratio,
        "operator_to_additive_ratio_final": additive_ratio,
        "operator_gate_mean": gate_mean,
    }


def _predict_pure_additive(dataset: Any, additive: TrainSingleAdditive) -> dict[str, Any]:
    predictions = {
        condition: dataset.ctrl_mean + additive.delta_for(condition)
        for condition in dataset.split.test_conditions
    }
    return {
        "predictions": predictions,
        "fit_metadata": {
            "description": "Pure additive architecture using train single perturbation deltas.",
            "train_single_genes": len(additive.delta_by_gene),
            "interaction_to_additive_ratio": 0.0,
        },
    }


def _run_interaction(
    dataset: Any,
    train_subset: list[str],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
    additive: TrainSingleAdditive,
    experiment: dict[str, Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    config = NormanInteractionConfig(
        gene_dim=len(dataset.gene_ids),
        feature_dim=features.feature_dim,
        hidden_dim=args.hidden_dim,
        rank=int(experiment.get("rank", 0)),
        dropout=0.0,
    )
    model = NormanAdditiveInteraction(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    arrays = _interaction_arrays(dataset, train_subset, features, standardizer, additive)
    history = _train_interaction(model, optimizer, arrays, args, device=device)
    predictions, diagnostics = _predict_interaction(
        model,
        dataset,
        dataset.split.test_conditions,
        features,
        standardizer,
        additive,
        device=device,
    )
    diagnostics.update(
        {
            "model_config": asdict(config),
            "train_conditions": len(train_subset),
            "train_loss_final": history[-1]["loss"],
            "train_loss_initial": history[0]["loss"],
            "train_interaction_ratio_final": history[-1]["interaction_ratio"],
            "interaction_dominating": bool(history[-1]["interaction_ratio"] > 0.5),
            "device": str(device),
            "description": experiment["title"],
        }
    )
    return {"predictions": predictions, "fit_metadata": diagnostics}


def _train_interaction(
    model: NormanAdditiveInteraction,
    optimizer: torch.optim.Optimizer,
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    additive = torch.as_tensor(arrays["additive_delta"], dtype=torch.float32, device=device)
    target_delta = torch.as_tensor(arrays["target_delta"], dtype=torch.float32, device=device)
    feature = torch.as_tensor(arrays["feature"], dtype=torch.float32, device=device)
    left = torch.as_tensor(arrays["left_feature"], dtype=torch.float32, device=device)
    right = torch.as_tensor(arrays["right_feature"], dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed + 17)
    history = []
    for _ in range(args.steps):
        indices = rng.choice(additive.shape[0], size=min(args.batch_size, additive.shape[0]), replace=False)
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        output = model(additive[idx], feature[idx], left[idx], right[idx])
        interaction = output["interaction_delta"]
        loss = F.mse_loss(output["delta"], target_delta[idx])
        loss = loss + 0.05 * interaction.pow(2).mean()
        loss = loss + 0.5 * interaction_ratio_loss(interaction, additive[idx], max_ratio=0.5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            ratio = norm_ratio(interaction, additive[idx]).mean()
        history.append({"loss": float(loss.detach().cpu()), "interaction_ratio": float(ratio.detach().cpu())})
    return history


@torch.no_grad()
def _predict_interaction(
    model: NormanAdditiveInteraction,
    dataset: Any,
    conditions: list[str] | tuple[str, ...],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
    additive: TrainSingleAdditive,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    arrays = _interaction_arrays(dataset, list(conditions), features, standardizer, additive)
    additive_tensor = torch.as_tensor(arrays["additive_delta"], dtype=torch.float32, device=device)
    feature = torch.as_tensor(arrays["feature"], dtype=torch.float32, device=device)
    left = torch.as_tensor(arrays["left_feature"], dtype=torch.float32, device=device)
    right = torch.as_tensor(arrays["right_feature"], dtype=torch.float32, device=device)
    output = model(additive_tensor, feature, left, right)
    delta = output["delta"].detach().cpu().numpy().astype(np.float32)
    predictions = {
        condition: dataset.ctrl_mean + delta[idx]
        for idx, condition in enumerate(conditions)
    }
    ratio_values = norm_ratio(output["interaction_delta"], output["additive_delta"]).detach().cpu().numpy()
    return predictions, {
        "eval_all_interaction_to_additive_ratio_mean": float(np.mean(ratio_values)),
        "eval_all_interaction_to_additive_ratio_max": float(np.max(ratio_values)),
        "interaction_scale": float(output["interaction_scale"].detach().cpu()),
    }


def _condition_arrays(
    dataset: Any,
    conditions: list[str],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
) -> dict[str, np.ndarray]:
    control = np.repeat(dataset.ctrl_mean[None, :], len(conditions), axis=0)
    target = np.stack([dataset.mean_for(condition) for condition in conditions]).astype(np.float32)
    feature = np.stack([standardizer.transform(features.condition_feature(condition)) for condition in conditions])
    return {"control": control.astype(np.float32), "target": target, "feature": feature.astype(np.float32)}


def _interaction_arrays(
    dataset: Any,
    conditions: list[str],
    features: NormanFeatureFactory,
    standardizer: FeatureStandardizer,
    additive: TrainSingleAdditive,
) -> dict[str, np.ndarray]:
    condition_features = []
    left_features = []
    right_features = []
    for condition in conditions:
        left, right = features.component_features(condition)
        condition_features.append(standardizer.transform(features.condition_feature(condition)))
        left_features.append(standardizer.transform(left))
        right_features.append(standardizer.transform(right))
    return {
        "additive_delta": np.stack([additive.delta_for(condition) for condition in conditions]).astype(np.float32),
        "target_delta": np.stack([dataset.delta_for(condition) for condition in conditions]).astype(np.float32),
        "feature": np.stack(condition_features).astype(np.float32),
        "left_feature": np.stack(left_features).astype(np.float32),
        "right_feature": np.stack(right_features).astype(np.float32),
    }


def _subsample_train_conditions(conditions: list[str], *, fraction: float, seed: int) -> list[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    singles = [condition for condition in conditions if is_single(condition)]
    combos = [condition for condition in conditions if is_combo(condition)]
    target_total = max(1, int(round(len(conditions) * fraction)))
    single_count = min(len(singles), max(1, int(round(len(singles) * fraction))))
    combo_count = min(len(combos), max(0, target_total - single_count))
    selected = []
    if single_count:
        selected.extend(rng.choice(singles, size=single_count, replace=False).tolist())
    if combo_count:
        selected.extend(rng.choice(combos, size=combo_count, replace=False).tolist())
    selected = sorted(set(selected))
    if len(selected) < target_total:
        remaining = [condition for condition in conditions if condition not in selected]
        needed = min(len(remaining), target_total - len(selected))
        selected.extend(rng.choice(remaining, size=needed, replace=False).tolist())
    return sorted(selected)


def _load_step0_gates(output_dir: Path) -> dict[str, float]:
    path = output_dir / "step0_baselines" / "baseline_metrics.tsv"
    df = pd.read_csv(path, sep="\t")
    values = {}
    for baseline in ("family_n_condition_mean_table", "single_perturbation_additive"):
        row = df[df["baseline"].eq(baseline) & df["subset"].eq("exact_train_combo")]
        if row.empty:
            raise FileNotFoundError(f"missing Step 0 baseline gate row for {baseline}")
        values[baseline] = float(row.iloc[0]["pearson_delta_all_mean"])
    return values


def _decision(exact: dict[str, Any], payload: dict[str, Any], primary_gate: float) -> tuple[str, str]:
    primary = float(exact.get("pearson_delta_all_mean", math.nan))
    variance_ratio = float(exact.get("prediction_delta_variance_ratio", math.nan))
    metadata = payload.get("fit_metadata", {})
    if not math.isnan(variance_ratio) and variance_ratio < 0.1:
        return "TIER1_DISCARD_MODE_COLLAPSE", "prediction_delta_variance_ratio < 0.1"
    if metadata.get("interaction_dominating"):
        return "TIER1_DISCARD_INTERACTION_DOMINATING", "interaction_to_additive_ratio_mean > 0.5"
    if primary > primary_gate:
        return "TIER1_KEEP_CONTROLLED_SIGNAL", "none"
    return "TIER1_DISCARD_NO_SIGNAL", f"primary {primary:.4f} <= gate {primary_gate:.4f}"


def _aggregate_row(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    for row in rows:
        if row["subset"] == subset:
            return row
    return {}


def _result_row(
    commit: str,
    experiment_num: int,
    experiment: dict[str, Any],
    status: str,
    exact: dict[str, Any],
    unseen: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "commit": commit,
        "experiment_num": experiment_num,
        "family": experiment["family"],
        "tier_reached": "Tier1",
        "status": status,
        "primary_metric": _metric_text(exact, "exact_train_combo", ["pearson_delta_all_mean", "pearson_delta_de20_mean"]),
        "secondary_metric": _metric_text(exact, "exact_train_combo", ["top20_de_overlap_mean", "mse_delta_all_mean"]),
        "protected_metric_summary": _metric_text(
            unseen,
            "unseen_single",
            ["direction_accuracy_de20_mean", "pearson_delta_all_mean", "prediction_delta_variance_ratio"],
        ),
        "architectural_change": experiment["title"],
        "description": payload.get("fit_metadata", {}).get("description", experiment["name"]),
    }


def _metric_text(row: dict[str, Any], prefix: str, names: list[str]) -> str:
    if not row:
        return f"{prefix}=unavailable"
    values = []
    for name in names:
        value = float(row.get(name, math.nan))
        values.append(f"{name}={'nan' if math.isnan(value) else f'{value:.4f}'}")
    return f"{prefix} " + "; ".join(values)


def _journal_entry(
    experiment_num: int,
    experiment: dict[str, Any],
    status: str,
    exact: dict[str, Any],
    unseen: dict[str, Any],
    payload: dict[str, Any],
    *,
    primary_gate: float,
    elapsed: float,
    hard_fail: str,
) -> str:
    metadata = payload.get("fit_metadata", {})
    return "\n".join(
        [
            f"## Experiment {experiment_num}: {experiment['title']}",
            "",
            f"**Hypothesis**: {experiment['family']} can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.",
            "",
            f"**Family**: {experiment['family']}.",
            "",
            f"**Implementation**: `{experiment['name']}` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.",
            "",
            "**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.",
            "",
            f"**Tier result**: {status}. Exact-train-combo Pearson delta all genes = `{float(exact.get('pearson_delta_all_mean', math.nan)):.4f}`; Tier 1 gate = `{primary_gate:.4f}`.",
            "",
            f"**Diagnostics**: exact top20 overlap = `{float(exact.get('top20_de_overlap_mean', math.nan)):.4f}`; unseen-single direction accuracy = `{float(unseen.get('direction_accuracy_de20_mean', math.nan)):.4f}`; variance ratio = `{float(exact.get('prediction_delta_variance_ratio', math.nan)):.4f}`; hard fail = `{hard_fail}`; metadata = `{json.dumps(_jsonable(metadata), sort_keys=True)}`.",
            "",
            f"**Learning**: Runtime `{elapsed:.2f}` seconds. No model-of-record change.",
        ]
    )


def _architecture_entry(experiment_num: int, experiment: dict[str, Any], payload: dict[str, Any], status: str) -> str:
    metadata = payload.get("fit_metadata", {})
    return "\n".join(
        [
            f"## Experiment {experiment_num}: {experiment['name']}",
            "",
            f"- Family: `{experiment['family']}`",
            f"- Status: `{status}`",
            f"- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`",
            f"- Config/diagnostics: `{json.dumps(_jsonable(metadata), sort_keys=True)}`",
        ]
    )


def _write_family_allocation(output_dir: Path, result_rows: list[dict[str, Any]], stop_fired: bool) -> None:
    df = pd.DataFrame(result_rows)
    a_count = int(df["family"].eq("Family A").sum()) if not df.empty else 0
    b_count = int(df["family"].eq("Family B").sum()) if not df.empty else 0
    keeps = int(df["status"].eq("TIER1_KEEP_CONTROLLED_SIGNAL").sum()) if not df.empty else 0
    lines = [
        "# Family Allocation",
        "",
        "| stage/family | experiments used | status | notes |",
        "|---|---:|---|---|",
        "| Step 0 / Stage A | 6 baseline registrations | complete | Four recomputed baselines plus GEARS and CPA published comparators. |",
        f"| Family A | {a_count} | {'closed' if stop_fired else 'tier1_smoke'} | Feature perturbation and low-rank operator smoke tests. |",
        f"| Family B | {b_count} | {'closed' if stop_fired else 'tier1_smoke'} | Additive and bounded-interaction smoke tests. |",
        "",
        f"Tier 1 keeps: `{keeps}`.",
        f"Architecture experiment cap remains `20`; used `{a_count + b_count}`.",
    ]
    if stop_fired:
        lines.append("")
        lines.append("Stop condition fired: `5 consecutive Tier 1 discards across families`.")
    (output_dir / "family_allocation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tier1_summary(
    arch_dir: Path,
    args: argparse.Namespace,
    *,
    device: torch.device,
    baselines: dict[str, float],
    primary_gate: float,
    result_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    elapsed: float,
    stop_fired: bool,
) -> None:
    exact = aggregate_df[aggregate_df["subset"].eq("exact_train_combo")].copy()
    columns = [
        "baseline",
        "pearson_delta_all_mean",
        "pearson_delta_de20_mean",
        "top20_de_overlap_mean",
        "mse_delta_all_mean",
        "direction_accuracy_de20_mean",
        "prediction_delta_variance_ratio",
        "mean_collapse_flag",
    ]
    lines = [
        "# Norman Tier 1 Architecture Summary",
        "",
        f"- Device: `{device}`",
        f"- Seed: `{args.seed}`",
        f"- Train fraction: `{args.train_fraction}`",
        f"- Steps per trained candidate: `{args.steps}`",
        f"- Family N exact Pearson gate input: `{baselines['family_n_condition_mean_table']:.4f}`",
        f"- Single-additive exact Pearson gate input: `{baselines['single_perturbation_additive']:.4f}`",
        f"- Tier 1 primary pass gate: `>{primary_gate:.4f}`",
        f"- Runtime seconds: `{elapsed:.2f}`",
        "",
        "## Exact-Train-Combo Metrics",
        "",
        _df_markdown(exact[columns]),
        "",
        "## Decisions",
        "",
        _df_markdown(result_df[["experiment_num", "family", "status", "architectural_change"]]),
    ]
    if stop_fired:
        lines.extend(["", "Stop condition fired: 5 consecutive Tier 1 discards across families."])
    (arch_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(output_dir: Path, result_df: pd.DataFrame, aggregate_df: pd.DataFrame, primary_gate: float) -> None:
    exact = aggregate_df[aggregate_df["subset"].eq("exact_train_combo")].copy()
    best = exact.sort_values("pearson_delta_all_mean", ascending=False).iloc[0].to_dict()
    lines = [
        "# Norman v1 Final Report",
        "",
        "Status: SEARCH_CLOSED_NO_NEW_BASELINE.",
        "",
        "## Stop Condition",
        "",
        "Five consecutive Tier 1 discards fired across Family A and Family B. The autonomous architecture loop is stopped pending explicit user instruction.",
        "",
        "## Best Tier 1 Candidate",
        "",
        f"- Candidate: `{best['baseline']}`",
        f"- Exact-train-combo Pearson delta all genes: `{float(best['pearson_delta_all_mean']):.4f}`",
        f"- Tier 1 pass gate: `>{primary_gate:.4f}`",
        f"- Exact top20 DE overlap: `{float(best['top20_de_overlap_mean']):.4f}`",
        f"- Exact MSE delta all genes: `{float(best['mse_delta_all_mean']):.4f}`",
        "",
        "## Model Of Record",
        "",
        "- Published GEARS remains the active model of record.",
        "- Family N condition-mean table remains the carried reference.",
        "- The Step 0 single-additive baseline remains the strongest recomputed exact-combo comparator.",
        "- No Tier 3 pass occurred.",
        "",
        "## Result Rows",
        "",
        _df_markdown(result_df[["experiment_num", "family", "status", "architectural_change"]]),
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_results(path: Path, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    rows.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text)


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _next_experiment_num(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return 0
    return int(df["experiment_num"].max()) + 1


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _df_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "nan" if pd.isna(value) else f"{float(value):.4f}")
    rows = [[str(column) for column in view.columns]]
    rows.extend([[str(row[column]) for column in view.columns] for _, row in view.iterrows()])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    header = "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(rows[0])) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
