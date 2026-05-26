from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_synthetic_lite_step0 import _jsonable, _latent_r2


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PSEUDOBULK_WHITENING_PROBE")
SPLITS = ("train", "val", "test")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe direct RNA pseudobulk-to-shared whitening geometry.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arrays = {split: _condition_arrays(dataset, split) for split in SPLITS}
    probe_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    for probe in _fit_probes(arrays["train"], dim=args.dim, ridge=args.ridge):
        for split in SPLITS:
            rna_embedding, image_embedding = probe["transform"](arrays[split])
            probe_rows.append(_probe_row(probe["name"], split, arrays, rna_embedding, image_embedding))
            embedding_rows.extend(_embedding_rows(probe["name"], split, "rna", rna_embedding))
            embedding_rows.extend(_embedding_rows(probe["name"], split, "image", image_embedding))

    _write_tsv(args.output_dir / "PROBE_RESULTS.tsv", probe_rows)
    _write_tsv(args.output_dir / "EMBEDDING_VALUES.tsv", embedding_rows)
    _write_report(args.output_dir / "REPORT.md", probe_rows, wallclock_minutes=(time.perf_counter() - started) / 60.0)
    return 0


def _condition_arrays(dataset, split: str) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    rna_mean = []
    image_mean = []
    z_bio_mean = []
    for group in groups:
        rna_mean.append(dataset.expression_values[group].mean(axis=0))
        image_mean.append(dataset.images[group].mean(axis=0).reshape(-1))
        z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
    return {
        "metadata": metadata,
        "rna_mean": np.stack(rna_mean).astype(float),
        "image_mean": np.stack(image_mean).astype(float),
        "z_bio": np.stack(z_bio_mean).astype(float),
    }


def _fit_probes(train: dict[str, Any], *, dim: int, ridge: float) -> list[dict[str, Any]]:
    x = train["rna_mean"]
    y = train["image_mean"]
    z = train["z_bio"]
    probes: list[dict[str, Any]] = []

    probes.append(
        {
            "name": "oracle_true_z_bio",
            "transform": lambda arrays: (arrays["z_bio"], arrays["z_bio"]),
        }
    )

    image_pca = _fit_pca(y, dim=dim, whiten=True)
    image_train_shared = image_pca(y)
    probes.append(
        {
            "name": "ridge_rna_to_image_pca",
            "transform": _ridge_probe(
                x,
                image_train_shared,
                image_pca,
                ridge=ridge,
                x_key="rna_mean",
                y_key="image_mean",
            ),
        }
    )

    probes.append(
        {
            "name": "ridge_supervised_z_bio",
            "transform": _dual_ridge_probe(
                x,
                z,
                y,
                z,
                ridge=ridge,
                x_key="rna_mean",
                y_key="image_mean",
            ),
        }
    )

    probes.append(
        {
            "name": "pls_cross_covariance",
            "transform": _pls_probe(x, y, dim=dim),
        }
    )

    probes.append(
        {
            "name": "regularized_cca",
            "transform": _cca_probe(x, y, dim=dim, ridge=ridge),
        }
    )
    return probes


def _ridge_probe(
    x_train: np.ndarray,
    y_shared_train: np.ndarray,
    y_project: Callable[[np.ndarray], np.ndarray],
    *,
    ridge: float,
    x_key: str,
    y_key: str,
) -> Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    x_standardize = _fit_standardizer(x_train)
    coef = _ridge_coef(x_standardize(x_train), y_shared_train, ridge=ridge)

    def transform(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        x_shared = _apply_linear(x_standardize(arrays[x_key]), coef)
        y_shared = y_project(arrays[y_key])
        return x_shared, y_shared

    return transform


def _dual_ridge_probe(
    x_train: np.ndarray,
    x_target_train: np.ndarray,
    y_train: np.ndarray,
    y_target_train: np.ndarray,
    *,
    ridge: float,
    x_key: str,
    y_key: str,
) -> Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    x_standardize = _fit_standardizer(x_train)
    y_standardize = _fit_standardizer(y_train)
    x_coef = _ridge_coef(x_standardize(x_train), x_target_train, ridge=ridge)
    y_coef = _ridge_coef(y_standardize(y_train), y_target_train, ridge=ridge)

    def transform(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return _apply_linear(x_standardize(arrays[x_key]), x_coef), _apply_linear(y_standardize(arrays[y_key]), y_coef)

    return transform


def _pls_probe(x_train: np.ndarray, y_train: np.ndarray, *, dim: int) -> Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    x_standardize = _fit_standardizer(x_train)
    y_standardize = _fit_standardizer(y_train)
    xz = x_standardize(x_train)
    yz = y_standardize(y_train)
    cross = xz.T @ yz / max(1, xz.shape[0] - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    keep = min(dim, u.shape[1], vt.shape[0])
    x_projection = u[:, :keep] * np.sqrt(singular[:keep])[None, :]
    y_projection = vt.T[:, :keep] * np.sqrt(singular[:keep])[None, :]

    def transform(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return x_standardize(arrays["rna_mean"]) @ x_projection, y_standardize(arrays["image_mean"]) @ y_projection

    return transform


def _cca_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    dim: int,
    ridge: float,
) -> Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
    x_whiten = _fit_pca(x_train, dim=min(dim * 2, x_train.shape[0] - 1), whiten=True, ridge=ridge)
    y_whiten = _fit_pca(y_train, dim=min(dim * 2, y_train.shape[0] - 1), whiten=True, ridge=ridge)
    xw = x_whiten(x_train)
    yw = y_whiten(y_train)
    cross = xw.T @ yw / max(1, xw.shape[0] - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    keep = min(dim, u.shape[1], vt.shape[0])
    x_projection = u[:, :keep] * np.sqrt(singular[:keep])[None, :]
    y_projection = vt.T[:, :keep] * np.sqrt(singular[:keep])[None, :]

    def transform(arrays: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return x_whiten(arrays["rna_mean"]) @ x_projection, y_whiten(arrays["image_mean"]) @ y_projection

    return transform


def _fit_standardizer(values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return lambda value: (value - mean) / std


def _fit_pca(
    values: np.ndarray,
    *,
    dim: int,
    whiten: bool,
    ridge: float = 1e-6,
) -> Callable[[np.ndarray], np.ndarray]:
    standardize = _fit_standardizer(values)
    centered = standardize(values)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(dim, vt.shape[0])
    components = vt[:keep].T
    if whiten:
        scale = np.sqrt(max(1, values.shape[0] - 1)) / np.sqrt(singular[:keep] ** 2 + ridge)
    else:
        scale = np.ones(keep, dtype=float)

    def transform(next_values: np.ndarray) -> np.ndarray:
        return (standardize(next_values) @ components) * scale[None, :]

    return transform


def _ridge_coef(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(x_aug.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)


def _apply_linear(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return x_aug @ coef


def _probe_row(
    name: str,
    split: str,
    arrays: dict[str, dict[str, Any]],
    rna_embedding: np.ndarray,
    image_embedding: np.ndarray,
) -> dict[str, Any]:
    if name not in _probe_cache:
        _probe_cache[name] = {}
    metadata = arrays[split]["metadata"]
    retrieval = cross_modal_retrieval_metrics(
        rna_embedding,
        image_embedding,
        metadata,
        metadata,
        ks=(1, 5),
        stratify_by=(),
    )
    rna_stats = _embedding_stats(rna_embedding)
    image_stats = _embedding_stats(image_embedding)
    batch_probe = batch_probe_metrics(rna_embedding, metadata, label_col="batch")
    if split == "train":
        rna_r2 = _latent_r2(rna_embedding, arrays[split]["z_bio"], rna_embedding, arrays[split]["z_bio"])
        image_r2 = _latent_r2(image_embedding, arrays[split]["z_bio"], image_embedding, arrays[split]["z_bio"])
    else:
        rna_r2 = _latent_r2(
            _probe_cache[name]["train_rna"],
            arrays["train"]["z_bio"],
            rna_embedding,
            arrays[split]["z_bio"],
        )
        image_r2 = _latent_r2(
            _probe_cache[name]["train_image"],
            arrays["train"]["z_bio"],
            image_embedding,
            arrays[split]["z_bio"],
        )
    if split == "train":
        _probe_cache[name]["train_rna"] = rna_embedding
        _probe_cache[name]["train_image"] = image_embedding
    batch_balanced = float(batch_probe.get("batch_probe_balanced_accuracy", np.nan))
    majority = float(batch_probe.get("batch_probe_majority_accuracy", 0.5))
    collapse = bool(rna_stats["min_std"] < 0.01 or image_stats["min_std"] < 0.01)
    pass_gate = bool(
        split == "test"
        and not collapse
        and retrieval["rna_to_image_recall@1"] >= RANDOM_RECALL1 + 0.05
        and rna_r2 > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= majority + 0.10)
    )
    return {
        "probe": name,
        "split": split,
        "collapse_flag": collapse,
        "pass_gate": pass_gate,
        "rna_min_std": rna_stats["min_std"],
        "image_min_std": image_stats["min_std"],
        "rna_mean_std": rna_stats["mean_std"],
        "image_mean_std": image_stats["mean_std"],
        "rna_rank": rna_stats["rank"],
        "image_rank": image_stats["rank"],
        "rna_bio_latent_r2": rna_r2,
        "image_bio_latent_r2": image_r2,
        "batch_balanced_accuracy": batch_balanced,
        "batch_majority_accuracy": majority,
        "rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "image_to_rna_recall@5": retrieval["image_to_rna_recall@5"],
        "rna_to_image_median_rank": retrieval["rna_to_image_median_rank"],
    }


_probe_cache: dict[str, dict[str, np.ndarray]] = {}


def _embedding_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    std = values.std(axis=0)
    singular = np.linalg.svd(values - values.mean(axis=0, keepdims=True), full_matrices=False, compute_uv=False)
    return {
        "min_std": float(std.min()) if std.size else 0.0,
        "mean_std": float(std.mean()) if std.size else 0.0,
        "rank": int(np.sum(singular > 1e-3)),
    }


def _embedding_rows(probe: str, split: str, modality: str, embeddings: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for row_idx, row_values in enumerate(np.asarray(embeddings, dtype=float)):
        row = {"probe": probe, "split": split, "modality": modality, "row": row_idx}
        row.update({f"dim_{idx}": value for idx, value in enumerate(row_values)})
        rows.append(row)
    return rows


def _write_report(path: Path, rows: list[dict[str, Any]], *, wallclock_minutes: float) -> None:
    frame = pd.DataFrame(rows)
    test = frame[frame["split"] == "test"].sort_values(
        ["pass_gate", "rna_to_image_recall@1", "rna_bio_latent_r2"],
        ascending=[False, False, False],
    )
    passing = test[test["pass_gate"]]
    lines = [
        "# Pseudobulk Whitening Probe Report",
        "",
        f"Wallclock minutes: `{wallclock_minutes:.3f}`",
        "",
        "## Summary",
        "",
        f"- Probes evaluated: `{frame['probe'].nunique()}`",
        f"- Passing test probes: `{len(passing)}`",
        "",
        "## Test Results",
        "",
    ]
    for _, row in test.iterrows():
        lines.append(
            f"- `{row['probe']}`: pass={bool(row['pass_gate'])}, "
            f"collapse={bool(row['collapse_flag'])}, "
            f"recall@1={row['rna_to_image_recall@1']:.4f}, "
            f"rna_min_std={row['rna_min_std']:.4f}, image_min_std={row['image_min_std']:.4f}, "
            f"rna_R2={row['rna_bio_latent_r2']:.4f}, image_R2={row['image_bio_latent_r2']:.4f}, "
            f"batch_bal_acc={row['batch_balanced_accuracy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A passing probe means the synthetic data contains enough condition-level RNA and image information to build a safe shared space without changing JEPA. "
            "That shared-space geometry should then be used as the target for the next minimal model repair.",
            "",
            "## Artifacts",
            "",
            "- `PROBE_RESULTS.tsv`",
            "- `EMBEDDING_VALUES.tsv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
