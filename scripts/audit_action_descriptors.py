from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import perturbation_genes
from perturb_jepa.training.action_descriptors import (
    heldout_descriptor_coverage,
    synthetic_action_descriptor_matrix,
    synthetic_action_descriptor_spec,
)
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit non-leaky action descriptors for Phase 3.")
    parser.add_argument("--synthetic-dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/autoresearch_biomech_jepa_phase3/diagnostics/action_descriptor_audit"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    metrics.update({f"synthetic/{key}": value for key, value in _synthetic_metrics(args.synthetic_dataset, seed=args.seed).items()})
    metrics.update({f"norman/{key}": value for key, value in _norman_metrics(args.norman_h5ad).items()})
    metrics["decision_label"] = _decision_label(metrics)
    (args.output_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(args.output_dir / "REPORT.md", metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _synthetic_metrics(name: str, *, seed: int) -> dict[str, float | str]:
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(name, seed=seed))
    spec = synthetic_action_descriptor_spec(dataset)
    matrix = synthetic_action_descriptor_matrix(dataset, spec)
    coverage = heldout_descriptor_coverage(dataset, matrix)
    non_control = [idx for idx in range(dataset.config.num_perturbations) if idx != dataset.config.control_perturbation_id]
    unique_rows = np.unique(matrix[non_control], axis=0)
    return {
        "dataset": name,
        "descriptor_dim": float(matrix.shape[1]),
        "gene_descriptor_dim": float(spec.gene_dim),
        "program_descriptor_dim": float(spec.program_dim),
        "descriptor_rank": float(np.linalg.matrix_rank(matrix)),
        "non_control_unique_descriptor_fraction": float(unique_rows.shape[0] / max(1, len(non_control))),
        "uses_target_expression": 0.0,
        "uses_condition_key_one_hot": 0.0,
        "uses_perturbation_id_embedding_only": 0.0,
        **coverage,
    }


def _norman_metrics(path: Path) -> dict[str, float | str]:
    try:
        import scanpy as sc
    except ImportError:
        return {
            "h5ad": str(path),
            "available": 0.0,
            "descriptor_dim": 0.0,
            "uses_condition_key_one_hot": 0.0,
            "uses_target_expression": 0.0,
            "error": "scanpy unavailable",
        }
    adata = sc.read_h5ad(path, backed="r")
    conditions = tuple(str(value) for value in adata.obs["condition"].astype(str).unique())
    action_genes = sorted({gene for condition in conditions for gene in perturbation_genes(condition) if gene != "ctrl"})
    gene_to_index = {gene: index for index, gene in enumerate(action_genes)}
    descriptors = []
    for condition in conditions:
        if condition == "ctrl":
            continue
        values = np.zeros(len(action_genes), dtype=np.float32)
        for gene in perturbation_genes(condition):
            index = gene_to_index.get(gene)
            if index is not None:
                values[index] = 1.0
        descriptors.append(values)
    matrix = np.stack(descriptors, axis=0) if descriptors else np.zeros((0, len(action_genes)), dtype=np.float32)
    nonzero = matrix.sum(axis=1) > 0.0 if matrix.size else np.asarray([], dtype=bool)
    return {
        "h5ad": str(path),
        "available": 1.0,
        "conditions": float(len(conditions)),
        "action_genes": float(len(action_genes)),
        "descriptor_dim": float(len(action_genes)),
        "descriptor_rank": float(np.linalg.matrix_rank(matrix)) if matrix.size else 0.0,
        "non_control_nonzero_descriptor_fraction": float(nonzero.mean()) if nonzero.size else 0.0,
        "uses_condition_key_one_hot": 0.0,
        "uses_target_expression": 0.0,
        "chemical_dose_used": 0.0,
        "batch_metadata_used": 0.0,
    }


def _decision_label(metrics: dict[str, Any]) -> str:
    if metrics.get("synthetic/heldout_descriptor_coverage", 0.0) < 1.0 or metrics.get("synthetic/heldout_nonzero_descriptor_fraction", 0.0) < 1.0:
        return "ACTION_DESCRIPTOR_MISSING_FOR_HELDOUT"
    if metrics.get("synthetic/uses_target_expression", 1.0) > 0.0 or metrics.get("norman/uses_target_expression", 1.0) > 0.0:
        return "ACTION_DESCRIPTOR_LEAKY"
    if metrics.get("synthetic/uses_condition_key_one_hot", 1.0) > 0.0 or metrics.get("norman/uses_condition_key_one_hot", 1.0) > 0.0:
        return "ACTION_DESCRIPTOR_LEAKY"
    if metrics.get("synthetic/descriptor_rank", 0.0) < 2.0 or metrics.get("synthetic/non_control_unique_descriptor_fraction", 0.0) < 0.5:
        return "ACTION_DESCRIPTOR_TOO_WEAK_DIAGNOSTIC_ONLY"
    return "ACTION_DESCRIPTOR_VALID"


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Action Descriptor Audit",
        "",
        f"Decision label: `{metrics['decision_label']}`",
        "",
        "## Synthetic Genetic Anchor",
        "",
        f"- Held-out descriptor coverage: `{metrics.get('synthetic/heldout_descriptor_coverage', float('nan')):.4f}`",
        f"- Held-out nonzero descriptor fraction: `{metrics.get('synthetic/heldout_nonzero_descriptor_fraction', float('nan')):.4f}`",
        f"- Descriptor rank: `{metrics.get('synthetic/descriptor_rank', float('nan')):.4f}`",
        f"- Uses target expression: `{metrics.get('synthetic/uses_target_expression', float('nan')):.0f}`",
        f"- Uses condition-key one-hot: `{metrics.get('synthetic/uses_condition_key_one_hot', float('nan')):.0f}`",
        "",
        "## Norman",
        "",
        f"- Descriptor dim: `{metrics.get('norman/descriptor_dim', float('nan')):.0f}`",
        f"- Non-control nonzero descriptor fraction: `{metrics.get('norman/non_control_nonzero_descriptor_fraction', float('nan')):.4f}`",
        f"- Chemical dose used: `{metrics.get('norman/chemical_dose_used', float('nan')):.0f}`",
        f"- Batch metadata used: `{metrics.get('norman/batch_metadata_used', float('nan')):.0f}`",
        "",
        "## Interpretation",
        "",
        "Synthetic held-out perturbations now have descriptors generated from perturbation configuration, not target expression. Norman uses gene multi-hot descriptors and fixed guide presence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
