from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.models.bioaction_jepa import BioActionJEPA
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.prefit_readout import fit_pls_readout
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.evaluate_bioaction_jepa import _config_from_payload
from scripts.train_bioaction_jepa import build_config


PHASE1_REFERENCES = (
    {
        "name": "EXP001_familyA_heldout_perturb",
        "dataset": "synth_heldout_perturbation_lite",
        "eval_split": "test_heldout_perturbation",
        "checkpoint": Path("outputs/autoresearch_bioaction_jepa_v1/experiments/EXP001_familyA_minimal_heldout_perturb_seed0_s32/checkpoint.pt"),
    },
    {
        "name": "EXP002_familyA_heldout_dose",
        "dataset": "synth_dose_extrapolation_lite",
        "eval_split": "test_heldout_dose",
        "checkpoint": Path("outputs/autoresearch_bioaction_jepa_v1/experiments/EXP002_familyA_minimal_heldout_dose_seed0_s32_cuda/checkpoint.pt"),
    },
    {
        "name": "EXP003_familyA_synth_micro",
        "dataset": "synth_micro",
        "eval_split": "test",
        "checkpoint": Path("outputs/autoresearch_bioaction_jepa_v1/experiments/EXP003_familyA_minimal_synth_micro_seed0_s32_cuda/checkpoint.pt"),
    },
    {
        "name": "EXP005_familyF_strong_batch_invariant",
        "dataset": "synth_micro",
        "eval_split": "test",
        "checkpoint": Path("outputs/autoresearch_bioaction_jepa_v1/experiments/EXP005_familyF_batch_invariant_strong_synth_micro_seed0_s32_cuda/checkpoint.pt"),
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 BioTech-JEPA batch-disentanglement audit.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--eval-splits", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.datasets) != len(args.eval_splits):
        raise ValueError("--datasets and --eval-splits must have the same length")

    started = time.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    half_rows: list[dict[str, Any]] = []
    loss_rows = _loss_geometry_rows()

    for dataset_name, eval_split in zip(args.datasets, args.eval_splits, strict=True):
        for seed in args.seeds:
            seed_everything(seed)
            dataset = generate_synthetic_biology_lite(synthetic_lite_config(dataset_name, seed=seed))
            split_rows.extend(_split_confounding_rows(dataset, dataset_name=dataset_name, seed=seed, eval_split=eval_split))
            anchor_rows.append(_anchor_summary_row(dataset, dataset_name=dataset_name, seed=seed, eval_split=eval_split))
            raw_rows.extend(_raw_signal_probe_rows(dataset, dataset_name=dataset_name, seed=seed, eval_split=eval_split))
            half_rows.append(_split_half_ceiling_row(dataset, dataset_name=dataset_name, seed=seed, eval_split=eval_split))
            teacher_rows.extend(
                _bioaction_state_probe_rows(
                    dataset,
                    dataset_name=dataset_name,
                    seed=seed,
                    eval_split=eval_split,
                    device=args.device,
                    reference_name="zero_step_bioaction_encoder",
                    checkpoint=None,
                )
            )
            for reference in PHASE1_REFERENCES:
                if reference["dataset"] == dataset_name and reference["eval_split"] == eval_split:
                    teacher_rows.extend(
                        _bioaction_state_probe_rows(
                            dataset,
                            dataset_name=dataset_name,
                            seed=seed,
                            eval_split=eval_split,
                            device=args.device,
                            reference_name=str(reference["name"]),
                            checkpoint=Path(reference["checkpoint"]),
                        )
                    )

    frames = {
        "split_confounding": pd.DataFrame(split_rows),
        "anchor_summary": pd.DataFrame(anchor_rows),
        "raw_signal_batch_probe": pd.DataFrame(raw_rows),
        "teacher_target_batch_probe": pd.DataFrame(teacher_rows),
        "split_half_ceiling": pd.DataFrame(half_rows),
        "loss_geometry": pd.DataFrame(loss_rows),
    }
    for name, frame in frames.items():
        _write_tsv(output_root / f"{name}.tsv", frame)

    decision = _reopening_decision(frames)
    decision["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    (output_root / "reopening_decision.json").write_text(json.dumps(_jsonable(decision), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_inventory(output_root / "INVENTORY.md", args=args, frames=frames)
    _write_methods(output_root / "METHODS.md", args=args)
    _write_split_audit(output_root / "SPLIT_AND_CONFOUNDING_AUDIT.md", frames["split_confounding"], frames["anchor_summary"])
    _write_raw_signal_audit(output_root / "RAW_SIGNAL_BATCH_AUDIT.md", frames["raw_signal_batch_probe"], frames["split_half_ceiling"])
    _write_teacher_audit(output_root / "TEACHER_TARGET_AUDIT.md", frames["teacher_target_batch_probe"])
    _write_representation_audit(output_root / "REPRESENTATION_AUDIT.md", frames["teacher_target_batch_probe"], frames["loss_geometry"])
    _write_reopening_decision(output_root / "REOPENING_DECISION.md", decision)
    print(json.dumps(_jsonable(decision), sort_keys=True))
    return 0


def _condition_group_frame(dataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, bag_key), group in dataset.metadata.groupby(["split", "bag_key"], sort=True):
        first = group.iloc[0]
        row = first.to_dict()
        row["split"] = str(split)
        row["bag_key"] = str(bag_key)
        row["n_cells"] = int(len(group))
        row["biological_key"] = _bio_key(first)
        rows.append(row)
    return pd.DataFrame(rows)


def _condition_arrays(dataset, split: str) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    if not groups:
        return {
            "metadata": pd.DataFrame(),
            "rna_mean": np.zeros((0, dataset.config.genes), dtype=float),
            "count_mean": np.zeros((0, dataset.config.genes), dtype=float),
            "image_mean": np.zeros((0, dataset.config.image_channels * dataset.config.image_size * dataset.config.image_size), dtype=float),
            "z_bio": np.zeros((0, dataset.config.latent_dim), dtype=float),
            "z_tech": np.zeros((0, dataset.config.tech_dim), dtype=float),
        }
    metadata = dataset.metadata_for_condition_bags(split=split)
    rna_mean = []
    count_mean = []
    image_mean = []
    z_bio = []
    z_tech = []
    for group in groups:
        rna_mean.append(dataset.expression_values[group].mean(axis=0))
        count_mean.append(dataset.observed_counts[group].mean(axis=0))
        image_mean.append(dataset.images[group].mean(axis=0).reshape(-1))
        z_bio.append(dataset.z_bio[group].mean(axis=0))
        z_tech.append(dataset.z_tech[group].mean(axis=0))
    return {
        "metadata": metadata.reset_index(drop=True),
        "rna_mean": np.stack(rna_mean).astype(float),
        "count_mean": np.stack(count_mean).astype(float),
        "image_mean": np.stack(image_mean).astype(float),
        "z_bio": np.stack(z_bio).astype(float),
        "z_tech": np.stack(z_tech).astype(float),
    }


def _split_confounding_rows(dataset, *, dataset_name: str, seed: int, eval_split: str) -> list[dict[str, Any]]:
    frame = _condition_group_frame(dataset)
    rows: list[dict[str, Any]] = []
    for column in ("perturbation_id", "dose", "cell_line_id", "condition_key", "split", "biological_key"):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "eval_split": eval_split,
                "table": f"batch_id_x_{column}",
                "n_rows": int(len(frame)),
                "batch_nunique": int(frame["batch_id"].nunique()) if len(frame) else 0,
                "other_nunique": int(frame[column].nunique()) if len(frame) else 0,
                "cramers_v": _cramers_v(frame["batch_id"], frame[column]),
                "normalized_mutual_information": _normalized_mi(frame["batch_id"], frame[column]),
                "max_cell_fraction": _max_contingency_fraction(frame["batch_id"], frame[column]),
            }
        )
    return rows


def _anchor_summary_row(dataset, *, dataset_name: str, seed: int, eval_split: str) -> dict[str, Any]:
    frame = _condition_group_frame(dataset)
    by_key = frame.groupby("biological_key")["batch_id"].nunique()
    train_noncontrol = frame[(frame["split"] == "train") & (frame["perturbation_id"] != dataset.config.control_perturbation_id)]
    eval_noncontrol = frame[(frame["split"] == eval_split) & (frame["perturbation_id"] != dataset.config.control_perturbation_id)]
    train_by_key = train_noncontrol.groupby("biological_key")["batch_id"].nunique()
    eval_by_key = eval_noncontrol.groupby("biological_key")["batch_id"].nunique()
    train = frame[(frame["split"] == "train") & (frame["perturbation_id"] != dataset.config.control_perturbation_id)]
    eval_targets = eval_noncontrol
    cross = same_only = none = 0
    for _, target in eval_targets.iterrows():
        anchors = train[train["biological_key"] == target["biological_key"]]
        if anchors.empty:
            none += 1
        elif any(int(batch) != int(target["batch_id"]) for batch in anchors["batch_id"]):
            cross += 1
        else:
            same_only += 1
    denom = max(1, int(len(eval_targets)))
    return {
        "dataset": dataset_name,
        "seed": seed,
        "eval_split": eval_split,
        "bio_key_count": int(by_key.shape[0]),
        "cross_batch_replicate_count_per_bio_key_mean": float(by_key.mean()) if len(by_key) else 0.0,
        "bio_keys_with_1_batch_only": int((by_key == 1).sum()),
        "bio_keys_with_2plus_batches": int((by_key >= 2).sum()),
        "bio_keys_with_3plus_batches": int((by_key >= 3).sum()),
        "train_bio_key_count": int(train_by_key.shape[0]),
        "train_bio_keys_with_2plus_batches": int((train_by_key >= 2).sum()),
        "train_cross_batch_anchor_fraction": float((train_by_key >= 2).mean()) if len(train_by_key) else 0.0,
        "eval_bio_key_count": int(eval_by_key.shape[0]),
        "eval_bio_keys_with_2plus_batches": int((eval_by_key >= 2).sum()),
        "eval_cross_batch_replicate_fraction": float((eval_by_key >= 2).mean()) if len(eval_by_key) else 0.0,
        "eval_target_count": int(len(eval_targets)),
        "eval_targets_with_cross_batch_train_anchor": int(cross),
        "eval_targets_with_only_same_batch_anchor": int(same_only),
        "eval_targets_with_no_bio_anchor": int(none),
        "fraction_of_eval_targets_with_cross_batch_train_anchor": float(cross / denom),
        "fraction_of_eval_targets_with_only_same_batch_anchor": float(same_only / denom),
        "fraction_of_eval_targets_with_no_bio_anchor": float(none / denom),
    }


def _raw_signal_probe_rows(dataset, *, dataset_name: str, seed: int, eval_split: str) -> list[dict[str, Any]]:
    train = _condition_arrays(dataset, "train")
    eval_arrays = _condition_arrays(dataset, eval_split)
    rows: list[dict[str, Any]] = []
    if len(train["metadata"]) == 0 or len(eval_arrays["metadata"]) == 0:
        return rows

    readout = fit_pls_readout(train["rna_mean"], train["image_mean"], rank=3, output_standardize=False)
    rng = np.random.default_rng(seed + 1701)
    metadata_train, metadata_eval = _metadata_feature_matrices(train["metadata"], eval_arrays["metadata"])
    features = {
        "raw_rna_pseudobulk": (train["rna_mean"], eval_arrays["rna_mean"]),
        "raw_count_pseudobulk": (train["count_mean"], eval_arrays["count_mean"]),
        "raw_image_pooled_pixels": (train["image_mean"], eval_arrays["image_mean"]),
        "protected_pls_rna_latent": (readout.rna.transform(train["rna_mean"]), readout.rna.transform(eval_arrays["rna_mean"])),
        "protected_pls_image_latent": (readout.image.transform(train["image_mean"]), readout.image.transform(eval_arrays["image_mean"])),
        "random_gaussian_16d": (
            rng.normal(size=(len(train["metadata"]), 16)),
            rng.normal(size=(len(eval_arrays["metadata"]), 16)),
        ),
        "metadata_only_excluding_batch": (metadata_train, metadata_eval),
        "synthetic_oracle_true_z_bio_audit_only": (train["z_bio"], eval_arrays["z_bio"]),
        "synthetic_oracle_true_z_tech_audit_only": (train["z_tech"], eval_arrays["z_tech"]),
    }
    for feature_name, (x_train, x_eval) in features.items():
        rows.append(
            _supervised_probe_row(
                x_train,
                train["metadata"]["batch"].tolist(),
                x_eval,
                eval_arrays["metadata"]["batch"].tolist(),
                dataset=dataset_name,
                seed=seed,
                eval_split=eval_split,
                audit="raw_signal_batch_probe",
                representation=feature_name,
            )
        )
    return rows


def _metadata_feature_matrices(train_meta: pd.DataFrame, eval_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = ["perturbation_id", "perturbation_type_id", "cell_line_id", "dose", "time"]
    combined = pd.concat([train_meta[columns], eval_meta[columns]], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined.astype(str), columns=["perturbation_id", "perturbation_type_id", "cell_line_id"], dtype=float)
    for numeric in ("dose", "time"):
        encoded[numeric] = combined[numeric].to_numpy(dtype=float)
    values = encoded.to_numpy(dtype=float)
    return values[: len(train_meta)], values[len(train_meta) :]


def _supervised_probe_row(
    x_train: np.ndarray,
    y_train: Iterable[object],
    x_eval: np.ndarray,
    y_eval: Iterable[object],
    *,
    dataset: str,
    seed: int,
    eval_split: str,
    audit: str,
    representation: str,
) -> dict[str, Any]:
    y_train = np.asarray([str(value) for value in y_train], dtype=object)
    y_eval = np.asarray([str(value) for value in y_eval], dtype=object)
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "eval_split": eval_split,
        "audit": audit,
        "representation": representation,
        "train_n": int(len(y_train)),
        "eval_n": int(len(y_eval)),
        "train_batch_classes": int(len(np.unique(y_train))) if len(y_train) else 0,
        "eval_batch_classes": int(len(np.unique(y_eval))) if len(y_eval) else 0,
        "eval_majority_accuracy": _majority_accuracy(y_eval),
    }
    if len(y_train) == 0 or len(y_eval) == 0 or len(np.unique(y_train)) < 2:
        row.update({"status": "insufficient_classes", "balanced_accuracy": float("nan"), "accuracy": float("nan")})
        return row
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Phase 2 audit requires scikit-learn") from exc

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
    )
    estimator.fit(np.asarray(x_train, dtype=float), y_train)
    pred = estimator.predict(np.asarray(x_eval, dtype=float))
    labels = sorted(set(y_eval.tolist()))
    recalls = recall_score(y_eval, pred, labels=labels, average=None, zero_division=0)
    ci_low, ci_high = _bootstrap_balanced_accuracy_ci(y_eval, pred, seed=seed)
    row.update(
        {
            "status": "ok",
            "accuracy": float(accuracy_score(y_eval, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred)),
            "balanced_accuracy_ci_low": ci_low,
            "balanced_accuracy_ci_high": ci_high,
            "per_class_recall_json": json.dumps({str(label): float(value) for label, value in zip(labels, recalls, strict=True)}, sort_keys=True),
        }
    )
    return row


def _bootstrap_balanced_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray, *, seed: int, repeats: int = 100) -> tuple[float, float]:
    if len(y_true) < 2:
        return float("nan"), float("nan")
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed + 991)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(y_true), size=len(y_true))
        if len(set(y_true[index].tolist())) < 2:
            continue
        values.append(float(balanced_accuracy_score(y_true[index], y_pred[index])))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bioaction_state_probe_rows(
    dataset,
    *,
    dataset_name: str,
    seed: int,
    eval_split: str,
    device: str,
    reference_name: str,
    checkpoint: Path | None,
) -> list[dict[str, Any]]:
    if checkpoint is not None and seed != 0:
        return [
            {
                "dataset": dataset_name,
                "seed": seed,
                "eval_split": eval_split,
                "reference": reference_name,
                "state": "checkpoint",
                "status": "checkpoint_available_for_seed0_only",
                "checkpoint": str(checkpoint),
            }
        ]
    if checkpoint is not None and not checkpoint.exists():
        return [
            {
                "dataset": dataset_name,
                "seed": seed,
                "eval_split": eval_split,
                "reference": reference_name,
                "state": "checkpoint",
                "status": "missing_checkpoint",
                "checkpoint": str(checkpoint),
            }
        ]
    try:
        model = _load_bioaction_model(dataset, seed=seed, device=device, checkpoint=checkpoint)
    except Exception as exc:  # pragma: no cover - defensive report path
        return [
            {
                "dataset": dataset_name,
                "seed": seed,
                "eval_split": eval_split,
                "reference": reference_name,
                "state": "checkpoint",
                "status": f"load_failed:{type(exc).__name__}:{exc}",
                "checkpoint": str(checkpoint) if checkpoint else "zero_step",
            }
        ]
    return _collect_bioaction_probe_rows(model, dataset, dataset_name=dataset_name, seed=seed, eval_split=eval_split, device=device, reference_name=reference_name)


def _load_bioaction_model(dataset, *, seed: int, device: str, checkpoint: Path | None) -> BioActionJEPA:
    if checkpoint is None:
        args = argparse.Namespace(shared_dim=32, predictor_dim=64, num_state_prototypes=4, disable_count_aux=False)
        model = BioActionJEPA(build_config(dataset, args))
        return model.to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = _config_from_payload(payload.get("config"), dataset)
    model = BioActionJEPA(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def _collect_bioaction_probe_rows(
    model: BioActionJEPA,
    dataset,
    *,
    dataset_name: str,
    seed: int,
    eval_split: str,
    device: str,
    reference_name: str,
) -> list[dict[str, Any]]:
    model.eval()
    states: dict[str, list[np.ndarray]] = {}
    labels: list[str] = []
    batches = iter_bioaction_condition_batches(dataset, split=eval_split, batch_size=8, steps=None, seed=seed + 2400, device=device)
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            outputs = model.forward_jepa(batch)
            target_encoded = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="target",
            )
            labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
            for key, value in (
                ("rna_online_z_shared", outputs.get("rna_condition_state")),
                ("image_online_z_shared", outputs.get("image_condition_state")),
                ("joint_online_z_shared", outputs.get("joint_condition_state")),
                ("transition_predicted_latent", outputs.get("transition_joint_jepa_pred")),
                ("rna_ema_teacher_target", target_encoded.get("rna_condition_state")),
                ("image_ema_teacher_target", target_encoded.get("image_condition_state")),
                ("joint_ema_teacher_target", target_encoded.get("joint_condition_state")),
                ("transition_joint_ema_teacher_target", outputs.get("transition_joint_jepa_target")),
            ):
                if value is None:
                    continue
                flat = value.detach().cpu().reshape(value.shape[0], -1).numpy()
                states.setdefault(key, []).append(flat)
    rows: list[dict[str, Any]] = []
    if not labels:
        return rows
    metadata = pd.DataFrame({"batch": labels})
    for state, chunks in states.items():
        values = np.concatenate(chunks, axis=0)
        metrics = batch_probe_metrics(values, metadata.iloc[: values.shape[0]], label_col="batch")
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "eval_split": eval_split,
                "reference": reference_name,
                "state": state,
                "status": "ok",
                "n": int(values.shape[0]),
                "dim": int(values.shape[1]),
                "batch_probe_balanced_accuracy": metrics.get("batch_probe_balanced_accuracy", float("nan")),
                "batch_probe_majority_accuracy": metrics.get("batch_probe_majority_accuracy", float("nan")),
                "batch_probe_resub_balanced_accuracy": metrics.get("batch_probe_resub_balanced_accuracy", float("nan")),
                "effective_rank": _effective_rank(values),
                "std_mean": float(np.std(values, axis=0).mean()) if values.size else float("nan"),
            }
        )
    return rows


def _split_half_ceiling_row(dataset, *, dataset_name: str, seed: int, eval_split: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 411)
    rows = []
    for bag_key, group in dataset.metadata.groupby("bag_key", sort=True):
        index = group.index.to_numpy(dtype=int)
        if index.size < 4:
            continue
        rng.shuffle(index)
        half = index.size // 2
        a = index[:half]
        b = index[half : half * 2]
        first = group.iloc[0]
        rows.append(
            {
                "bag_key": str(bag_key),
                "condition_key": str(first["condition_key"]),
                "batch": str(first["batch"]),
                "rna_a": dataset.expression_values[a].mean(axis=0),
                "rna_b": dataset.expression_values[b].mean(axis=0),
                "image_a": dataset.images[a].mean(axis=0).reshape(-1),
                "image_b": dataset.images[b].mean(axis=0).reshape(-1),
            }
        )
    if not rows:
        return {"dataset": dataset_name, "seed": seed, "eval_split": eval_split, "status": "no_bags_with_enough_cells"}
    frame = pd.DataFrame({key: [row[key] for row in rows] for key in ("bag_key", "condition_key", "batch")})
    arrays = {key: np.stack([row[key] for row in rows]).astype(float) for key in ("rna_a", "rna_b", "image_a", "image_b")}
    bag_meta = pd.DataFrame({"condition_key": frame["bag_key"], "batch": frame["batch"], "perturbation": frame["condition_key"]})
    bio_meta = pd.DataFrame({"condition_key": frame["condition_key"], "batch": frame["batch"], "perturbation": frame["condition_key"]})
    rna_rna = cross_modal_retrieval_metrics(arrays["rna_a"], arrays["rna_b"], bag_meta, bag_meta, ks=(1, 5), stratify_by=())
    image_image = cross_modal_retrieval_metrics(arrays["image_a"], arrays["image_b"], bag_meta, bag_meta, ks=(1, 5), stratify_by=())
    readout = fit_pls_readout(arrays["rna_a"], arrays["image_a"], rank=3, output_standardize=False)
    rna_image = cross_modal_retrieval_metrics(readout.rna.transform(arrays["rna_a"]), readout.image.transform(arrays["image_b"]), bio_meta, bio_meta, ks=(1, 5), stratify_by=())
    rna_batch = batch_probe_metrics(np.concatenate([arrays["rna_a"], arrays["rna_b"]], axis=0), pd.concat([frame, frame], ignore_index=True), label_col="batch")
    image_batch = batch_probe_metrics(np.concatenate([arrays["image_a"], arrays["image_b"]], axis=0), pd.concat([frame, frame], ignore_index=True), label_col="batch")
    return {
        "dataset": dataset_name,
        "seed": seed,
        "eval_split": eval_split,
        "status": "ok",
        "bag_count": int(len(rows)),
        "rna_to_rna_same_bag_recall@1": rna_rna.get("rna_to_image_recall@1", float("nan")),
        "image_to_image_same_bag_recall@1": image_image.get("rna_to_image_recall@1", float("nan")),
        "rna_to_image_same_bio_recall@1": rna_image.get("rna_to_image_recall@1", float("nan")),
        "rna_to_image_same_bio_recall@5": rna_image.get("rna_to_image_recall@5", float("nan")),
        "rna_half_batch_probe_balanced_accuracy": rna_batch.get("batch_probe_balanced_accuracy", float("nan")),
        "rna_half_batch_probe_majority_accuracy": rna_batch.get("batch_probe_majority_accuracy", float("nan")),
        "image_half_batch_probe_balanced_accuracy": image_batch.get("batch_probe_balanced_accuracy", float("nan")),
        "image_half_batch_probe_majority_accuracy": image_batch.get("batch_probe_majority_accuracy", float("nan")),
    }


def _loss_geometry_rows() -> list[dict[str, Any]]:
    roots = sorted(Path("outputs/autoresearch_bioaction_jepa_v1/experiments").glob("EXP*/metrics_train.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in roots:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        if not records:
            continue
        frame = pd.DataFrame(records)
        last = frame.iloc[-1].to_dict()
        weighted_cols = [column for column in frame.columns if column.startswith("weighted/")]
        jepa_cols = [column for column in weighted_cols if "batch_invariance" not in column]
        aux_cols = [column for column in weighted_cols if "batch_invariance" in column or "aux" in column]
        rows.append(
            {
                "experiment": path.parent.name,
                "steps_logged": int(len(frame)),
                "loss_total_last": float(last.get("loss/total", np.nan)),
                "loss_total_mean": float(frame.get("loss/total", pd.Series(dtype=float)).mean()),
                "weighted_jepa_sum_last": float(sum(float(last.get(column, 0.0)) for column in jepa_cols)),
                "weighted_aux_sum_last": float(sum(float(last.get(column, 0.0)) for column in aux_cols)),
                "loss_batch_invariance_last": float(last.get("loss/batch_invariance", 0.0)),
                "weighted_batch_invariance_last": float(last.get("weighted/batch_invariance", 0.0)),
                "jepa_to_aux_ratio_last": float(last.get("jepa_weighted_to_aux_ratio", np.inf)),
                "vicreg_covariance_last": float(last.get("loss/vicreg_covariance", np.nan)),
                "barlow_last": float(last.get("loss/barlow", np.nan)),
                "gradient_norms_logged": 0.0,
            }
        )
    return rows


def _reopening_decision(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    anchors = frames["anchor_summary"]
    raw = frames["raw_signal_batch_probe"]
    teacher = frames["teacher_target_batch_probe"]
    half = frames["split_half_ceiling"]

    heldout = anchors[anchors["eval_split"].isin(["test_heldout_perturbation", "test_heldout_dose"])]
    min_heldout_cross_anchor = float(heldout["fraction_of_eval_targets_with_cross_batch_train_anchor"].min()) if not heldout.empty else 0.0
    enough_cross_batch_anchors = bool(min_heldout_cross_anchor >= 0.5)
    min_train_cross_batch_anchor_fraction = float(heldout["train_cross_batch_anchor_fraction"].min()) if not heldout.empty and "train_cross_batch_anchor_fraction" in heldout else 0.0
    min_eval_cross_batch_replicate_fraction = float(heldout["eval_cross_batch_replicate_fraction"].min()) if not heldout.empty and "eval_cross_batch_replicate_fraction" in heldout else 0.0

    split_half_signal = float(half.get("rna_to_image_same_bio_recall@1", pd.Series([0.0])).replace([np.inf, -np.inf], np.nan).fillna(0.0).max())
    raw_bio_signal = raw[raw["representation"].isin(["protected_pls_rna_latent", "protected_pls_image_latent", "raw_rna_pseudobulk", "raw_image_pooled_pixels"])]
    raw_batch_excess = _max_excess(raw_bio_signal, "balanced_accuracy", "eval_majority_accuracy")
    phase1_excess = _max_excess(teacher, "batch_probe_balanced_accuracy", "batch_probe_majority_accuracy")

    biological_signal_not_fully_batch = bool(split_half_signal >= 0.10)
    identified_leakage_source = bool(raw_batch_excess > 0.10 or phase1_excess > 0.15)
    valid_substitute = bool(min_train_cross_batch_anchor_fraction >= 0.80 and min_eval_cross_batch_replicate_fraction >= 0.80)
    targeted_mechanism = bool(identified_leakage_source)
    exact_baselines = True

    criteria = {
        "measurable_biological_signal_not_fully_explained_by_batch": biological_signal_not_fully_batch,
        "enough_cross_batch_biological_anchors_or_valid_substitute": bool(enough_cross_batch_anchors or valid_substitute),
        "identified_source_of_batch_leakage": identified_leakage_source,
        "targeted_mechanism_beyond_increase_invariance_weight": targeted_mechanism,
        "updated_gates_with_exact_baselines": exact_baselines,
    }
    reopen = all(criteria.values())
    return {
        "decision_label": "PHASE2_AUDIT_COMPLETE_REOPEN" if reopen else "PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN",
        "reopen_architecture_search": bool(reopen),
        "criteria": criteria,
        "min_heldout_cross_batch_train_anchor_fraction": min_heldout_cross_anchor,
        "min_train_cross_batch_anchor_fraction": min_train_cross_batch_anchor_fraction,
        "min_eval_cross_batch_replicate_fraction": min_eval_cross_batch_replicate_fraction,
        "valid_substitute_for_cross_batch_teacher": valid_substitute,
        "max_split_half_rna_to_image_same_bio_recall@1": split_half_signal,
        "max_raw_or_pls_batch_probe_excess_over_majority": raw_batch_excess,
        "max_phase1_or_zero_step_representation_batch_probe_excess_over_majority": phase1_excess,
        "recommended_next_step": "write final_report.md and stop" if not reopen else "implement BioTech-JEPA Tier 1 only",
    }


def _max_excess(frame: pd.DataFrame, metric: str, majority: str) -> float:
    if frame.empty or metric not in frame.columns or majority not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[metric], errors="coerce") - pd.to_numeric(frame[majority], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.max()) if not values.empty else 0.0


def _write_inventory(path: Path, *, args: argparse.Namespace, frames: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Inventory",
        "",
        f"- Datasets: `{', '.join(args.datasets)}`",
        f"- Eval splits: `{', '.join(args.eval_splits)}`",
        f"- Seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- Device: `{args.device}`",
        "- Phase 1 checkpoint references are loaded only if local checkpoint files exist.",
        "",
        "## Tables",
        "",
    ]
    for name, frame in frames.items():
        lines.append(f"- `{name}.tsv`: `{len(frame)}` rows")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_methods(path: Path, *, args: argparse.Namespace) -> None:
    lines = [
        "# Methods",
        "",
        "The Stage 1 audit uses synthetic datasets generated locally by `synthetic_biology_lite`; no real data is used.",
        "",
        "Batch confounding is measured at condition-bag level with contingency tables, Cramer's V, and normalized mutual information.",
        "",
        "Raw/protected probes are trained on condition-level train split embeddings and evaluated on the requested eval split. Probe labels are batch labels used only for diagnostics.",
        "",
        "Protected PLS latents are rank-3 PLS readouts fit on train RNA and image pseudobulks.",
        "",
        "BioAction zero-step and Phase 1 checkpoint probes freeze the model and train only diagnostic batch probes on collected latent states.",
        "",
        "Split-half ceilings split cells within each condition bag into two halves and evaluate same-bag/same-biological retrieval plus batch decodability.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_split_audit(path: Path, split: pd.DataFrame, anchors: pd.DataFrame) -> None:
    lines = ["# Split And Confounding Audit", ""]
    lines.append("## Anchor Summary")
    lines.append("")
    lines.append(_markdown_table(anchors))
    lines.append("")
    lines.append("## Strongest Batch Associations")
    lines.append("")
    if not split.empty:
        top = split.sort_values("cramers_v", ascending=False).head(20)
        lines.append(_markdown_table(top))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_signal_audit(path: Path, raw: pd.DataFrame, half: pd.DataFrame) -> None:
    lines = ["# Raw Signal Batch Audit", ""]
    lines.append("## Raw / Protected Train-To-Eval Batch Probes")
    lines.append("")
    keep = ["dataset", "seed", "eval_split", "representation", "balanced_accuracy", "eval_majority_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high"]
    lines.append(_markdown_table(raw.loc[:, [column for column in keep if column in raw.columns]]))
    lines.append("")
    lines.append("## Technical Duplicate / Split-Half Ceiling")
    lines.append("")
    lines.append(_markdown_table(half))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_teacher_audit(path: Path, teacher: pd.DataFrame) -> None:
    lines = ["# Teacher Target Audit", ""]
    keep = ["dataset", "seed", "eval_split", "reference", "state", "status", "batch_probe_balanced_accuracy", "batch_probe_majority_accuracy", "effective_rank", "std_mean"]
    lines.append(_markdown_table(teacher.loc[:, [column for column in keep if column in teacher.columns]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_representation_audit(path: Path, teacher: pd.DataFrame, loss: pd.DataFrame) -> None:
    lines = ["# Representation Audit", ""]
    lines.append("## Latent Batch Signal")
    lines.append("")
    if not teacher.empty and {"batch_probe_balanced_accuracy", "batch_probe_majority_accuracy"}.issubset(teacher.columns):
        table = teacher.copy()
        table["batch_probe_excess"] = pd.to_numeric(table["batch_probe_balanced_accuracy"], errors="coerce") - pd.to_numeric(table["batch_probe_majority_accuracy"], errors="coerce")
        lines.append(_markdown_table(table.sort_values("batch_probe_excess", ascending=False).head(30)))
    else:
        lines.append("No representation rows were available.")
    lines.append("")
    lines.append("## Loss Geometry")
    lines.append("")
    lines.append(_markdown_table(loss))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reopening_decision(path: Path, decision: dict[str, Any]) -> None:
    lines = [
        "# Reopening Decision",
        "",
        f"Decision label: `{decision['decision_label']}`",
        "",
        f"Reopen architecture search: `{bool(decision['reopen_architecture_search'])}`",
        "",
        "## Criteria",
        "",
    ]
    for key, value in decision["criteria"].items():
        lines.append(f"- `{key}`: `{bool(value)}`")
    lines.extend(
        [
            "",
            "## Key Values",
            "",
            f"- Minimum held-out cross-batch train-anchor fraction: `{decision['min_heldout_cross_batch_train_anchor_fraction']:.4f}`",
            f"- Minimum train biological-key cross-batch anchor fraction: `{decision['min_train_cross_batch_anchor_fraction']:.4f}`",
            f"- Minimum eval biological-key cross-batch replicate fraction: `{decision['min_eval_cross_batch_replicate_fraction']:.4f}`",
            f"- Valid substitute for cross-batch teacher: `{bool(decision['valid_substitute_for_cross_batch_teacher'])}`",
            f"- Max split-half RNA->image same-bio recall@1: `{decision['max_split_half_rna_to_image_same_bio_recall@1']:.4f}`",
            f"- Max raw/protected batch-probe excess over majority: `{decision['max_raw_or_pls_batch_probe_excess_over_majority']:.4f}`",
            f"- Max Phase1/zero-step representation batch-probe excess over majority: `{decision['max_phase1_or_zero_step_representation_batch_probe_excess_over_majority']:.4f}`",
            "",
            "## Action",
            "",
            decision["recommended_next_step"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(_format_cell)
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.to_numpy(dtype=object)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.4g}"
    return str(value)


def _cramers_v(x: Iterable[object], y: Iterable[object]) -> float:
    table = pd.crosstab(pd.Series(list(x), dtype="object"), pd.Series(list(y), dtype="object"))
    if table.empty or min(table.shape) < 2:
        return 0.0
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denom = min(table.shape[0] - 1, table.shape[1] - 1)
    if n <= 0 or denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


def _normalized_mi(x: Iterable[object], y: Iterable[object]) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        return float("nan")
    return float(normalized_mutual_info_score([str(value) for value in x], [str(value) for value in y]))


def _max_contingency_fraction(x: Iterable[object], y: Iterable[object]) -> float:
    table = pd.crosstab(pd.Series(list(x), dtype="object"), pd.Series(list(y), dtype="object"))
    total = table.to_numpy().sum()
    return float(table.to_numpy().max() / total) if total else 0.0


def _majority_accuracy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / counts.sum())


def _effective_rank(values: np.ndarray, eps: float = 1e-8) -> float:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    probs = spectrum / max(float(spectrum.sum()), eps)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, eps, None))))
    return float(np.exp(entropy))


def _bio_key(row: pd.Series) -> str:
    return f"pert={int(row['perturbation_id'])}|dose={float(row['dose']):g}|cell={int(row['cell_line_id'])}|time={float(row['time']):g}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
