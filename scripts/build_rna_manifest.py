from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an RNA .h5ad file and build condition-bag manifests.")
    parser.add_argument("--input-h5ad", required=True, type=Path, help="Input AnnData .h5ad file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for normalized RNA outputs.")
    parser.add_argument("--output-h5ad", type=Path, default=None, help="Normalized .h5ad path. Defaults under --output-dir.")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="RNA metadata CSV path. Defaults under --output-dir.")
    parser.add_argument(
        "--condition-bags-parquet",
        type=Path,
        default=None,
        help="Condition-bag parquet path. Defaults under --output-dir.",
    )
    parser.add_argument("--qc-json", type=Path, default=None, help="QC JSON path. Defaults under --output-dir.")
    parser.add_argument("--qc-csv", type=Path, default=None, help="Per-condition QC CSV path. Defaults under --output-dir.")
    return parser.parse_args(argv)


def _default_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir
    return {
        "h5ad": args.output_h5ad or output_dir / "rna_normalized.h5ad",
        "metadata": args.metadata_csv or output_dir / "rna_metadata.csv",
        "bags": args.condition_bags_parquet or output_dir / "rna_condition_bags.parquet",
        "qc_json": args.qc_json or output_dir / "rna_qc_summary.json",
        "qc_csv": args.qc_csv or output_dir / "rna_condition_qc.csv",
    }


def _load_runtime_deps() -> tuple[Any, Any, Any, Any]:
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install data dependencies to build RNA manifests: pip install -e '.[data]'") from exc

    from perturb_jepa.data.schema import normalize_scrna_obs, normalize_scrna_var

    return ad, pd, normalize_scrna_obs, normalize_scrna_var


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _condition_bags(obs: Any) -> Any:
    frame = obs.copy()
    frame["_obs_index_for_manifest"] = frame.index.astype(str)
    grouped = frame.groupby("condition_key", sort=True, dropna=False)
    rows: list[dict[str, Any]] = []
    for condition_key, group in grouped:
        first = group.iloc[0]
        rows.append(
            {
                "condition_key": str(condition_key),
                "condition_key_fine": str(first["condition_key_fine"]),
                "condition_id": str(first["condition_key"]),
                "perturbation": str(first["perturbation"]),
                "dose": str(first["dose"]),
                "time": str(first["time"]),
                "cell_line": str(first["cell_line"]),
                "n_cells": int(len(group)),
                "obs_indices": json.dumps(group["_obs_index_for_manifest"].astype(str).tolist()),
            }
        )
    return rows


def build_rna_manifest(args: argparse.Namespace) -> dict[str, Any]:
    ad, pd, normalize_scrna_obs, normalize_scrna_var = _load_runtime_deps()
    paths = _default_paths(args)
    if not args.input_h5ad.exists():
        raise ValueError(f"input h5ad does not exist: {args.input_h5ad}")

    adata = ad.read_h5ad(args.input_h5ad)
    adata.obs = normalize_scrna_obs(adata.obs)
    adata.var = normalize_scrna_var(adata.var)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    adata.write_h5ad(paths["h5ad"])
    metadata = adata.obs.copy()
    metadata = metadata.drop(columns=["obs_index"], errors="ignore")
    metadata.insert(0, "obs_index", metadata.index.astype(str))
    metadata.to_csv(paths["metadata"], index=False)

    bag_rows = _condition_bags(adata.obs)
    bags = pd.DataFrame(bag_rows)
    try:
        bags.to_parquet(paths["bags"], index=False)
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pyarrow or fastparquet; install one of them.") from exc

    per_condition = bags[["condition_key", "perturbation", "dose", "time", "cell_line", "n_cells"]].copy()
    per_condition.to_csv(paths["qc_csv"], index=False)
    qc = {
        "input_h5ad": str(args.input_h5ad),
        "normalized_h5ad": str(paths["h5ad"]),
        "rna_metadata_csv": str(paths["metadata"]),
        "rna_condition_bags_parquet": str(paths["bags"]),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_conditions": int(len(bags)),
        "condition_key_columns": ["perturbation", "dose", "time", "cell_line"],
        "excluded_from_condition_key": ["batch", "plate", "well", "site"],
        "min_cells_per_condition": int(bags["n_cells"].min()) if len(bags) else 0,
        "max_cells_per_condition": int(bags["n_cells"].max()) if len(bags) else 0,
    }
    _write_json(paths["qc_json"], qc)
    return qc


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        qc = build_rna_manifest(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote normalized RNA manifest for {qc['n_cells']:,} cells across {qc['n_conditions']:,} conditions.")


if __name__ == "__main__":
    main()
