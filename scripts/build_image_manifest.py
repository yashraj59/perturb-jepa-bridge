from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize an image manifest and validate image paths.")
    parser.add_argument("--input-manifest", required=True, type=Path, help="Input image manifest CSV.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Output normalized image manifest CSV.")
    parser.add_argument("--image-root", type=Path, default=Path(""), help="Root used for relative image paths.")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Write the manifest even when image_path entries do not exist.",
    )
    parser.add_argument("--qc-json", type=Path, default=None, help="QC JSON path. Defaults next to output manifest.")
    parser.add_argument("--qc-csv", type=Path, default=None, help="Per-condition QC CSV path. Defaults next to output manifest.")
    return parser.parse_args(argv)


def _load_runtime_deps() -> tuple[Any, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to build image manifests: pip install -e .") from exc

    from perturb_jepa.data.schema import normalize_image_manifest

    return pd, normalize_image_manifest


def _read_manifest(pd: Any, path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"input manifest does not exist: {path}")
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _resolved_image_path(image_root: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not str(image_root):
        return path
    return image_root / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_image_manifest(args: argparse.Namespace) -> dict[str, Any]:
    pd, normalize_image_manifest = _load_runtime_deps()
    output_manifest = args.output_manifest
    qc_json = args.qc_json or output_manifest.with_name(output_manifest.stem + "_qc_summary.json")
    qc_csv = args.qc_csv or output_manifest.with_name(output_manifest.stem + "_condition_qc.csv")

    manifest = normalize_image_manifest(_read_manifest(pd, args.input_manifest))
    resolved = manifest["image_path"].map(lambda value: _resolved_image_path(args.image_root, value))
    exists = resolved.map(Path.exists)
    missing = manifest.loc[~exists].copy()
    if len(missing) and not args.allow_missing_images:
        examples = ", ".join(str(path) for path in resolved.loc[~exists].head(5).tolist())
        raise ValueError(
            f"{len(missing)} image_path entries do not exist under image_root {args.image_root!s}; "
            f"examples: {examples}. Pass --allow-missing-images to write anyway."
        )

    manifest = manifest.copy()
    manifest["image_exists"] = exists.astype(bool)
    manifest["resolved_image_path"] = resolved.map(str)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)

    per_condition = (
        manifest.groupby(["condition_key", "perturbation", "dose", "time", "cell_line"], sort=True, dropna=False)
        .agg(n_images=("image_path", "size"), n_missing_images=("image_exists", lambda values: int((~values).sum())))
        .reset_index()
    )
    qc_csv.parent.mkdir(parents=True, exist_ok=True)
    per_condition.to_csv(qc_csv, index=False)
    qc = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(output_manifest),
        "image_root": str(args.image_root),
        "n_rows": int(len(manifest)),
        "n_conditions": int(manifest["condition_key"].nunique()),
        "n_missing_images": int((~exists).sum()),
        "allow_missing_images": bool(args.allow_missing_images),
        "condition_key_columns": ["perturbation", "dose", "time", "cell_line"],
        "excluded_from_condition_key": ["batch", "plate", "well", "site"],
    }
    _write_json(qc_json, qc)
    return qc


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        qc = build_image_manifest(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote normalized image manifest with {qc['n_rows']:,} rows and {qc['n_missing_images']:,} missing images.")


if __name__ == "__main__":
    main()
