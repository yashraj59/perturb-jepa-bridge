from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DATASET_PATHS = {
    "sciplex3_raw": ROOT / "data/raw/SrivatsanTrapnell2020_sciplex3.h5ad",
    "sciplex3_h5ad": ROOT / "data/processed/sciplex3/rna_normalized.h5ad",
    "sciplex3_metadata": ROOT / "data/processed/sciplex3/rna_metadata.csv",
    "bfmoa_tables": ROOT / "data/raw/bf_moa_data_tables.tar.gz",
    "bfmoa_raw_manifest": ROOT / "data/interim/bf_moa_manifest_raw.csv",
    "bfmoa_manifest": ROOT / "data/processed/bf_moa/image_manifest.csv",
    "paired_manifest": ROOT / "data/processed/paired/paired_manifest.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real Perturb-JEPA benchmark from manifests to report.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--rna-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = _build_plan(args)
    _ensure_dirs(args.output_dir)

    commands_run: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    print("Perturb-JEPA real benchmark runner")
    print(f"mode={plan['run_mode']} pairing_mode={plan['pairing_mode']} dry_run={args.dry_run}")

    if not args.skip_download:
        commands_run.append(
            _run_or_print(
                [sys.executable, "scripts/download_public_data.py", "--dry-run", "--dataset", "all-metadata"],
                dry_run=False,
                description="Check public metadata download commands",
            )
        )

    blockers.extend(_check_required_inputs(args))
    if blockers:
        _print_blockers(blockers)
        if not args.dry_run:
            manifest = _write_repro_manifest(args, commands_run, blockers, plan, status="blocked_missing_data")
            _write_report(args, manifest, blockers, status="blocked_missing_data")
            return 2

    if args.dry_run:
        commands_run.extend(_dry_run_commands(args, plan))
        commands_run.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/evaluate_real_benchmark.py",
                    "--dry-run",
                    "--output-dir",
                    str(args.output_dir / "evaluation"),
                    "--baselines-dir",
                    str(args.output_dir / "baselines"),
                ],
                dry_run=False,
                description="Write unavailable evaluation scaffold",
            )
        )
        manifest = _write_repro_manifest(args, commands_run, blockers, plan, status="dry_run")
        _write_report(args, manifest, blockers, status="dry_run")
        _print_summary(args, manifest, commands_run, blockers, status="dry_run")
        return 0

    commands_run.extend(_run_manifest_steps(args))
    if not args.image_only:
        commands_run.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/train_pretrain_rna.py",
                    "--config",
                    "configs/real/pretrain_rna_sciplex3.yaml",
                    "--checkpoint-out",
                    "checkpoints/real/pretrain_rna_sciplex3.pt",
                    "--max-cells",
                    str(plan["max_cells"]),
                ],
                dry_run=False,
                description="Train RNA encoder",
            )
        )
    if not args.rna_only:
        commands_run.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/train_pretrain_image.py",
                    "--config",
                    "configs/real/pretrain_image_bfmoa.yaml",
                    "--checkpoint-out",
                    "checkpoints/real/pretrain_image_bfmoa.pt",
                    "--max-images",
                    str(plan["max_images"]),
                ],
                dry_run=False,
                description="Train image encoder",
            )
        )
    if not args.rna_only and not args.image_only:
        commands_run.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/train_bridge.py",
                    "--config",
                    "configs/real/bridge_sciplex3_bfmoa.yaml",
                    "--checkpoint-out",
                    "checkpoints/real/bridge_sciplex3_bfmoa.pt",
                    "--max-cells",
                    str(plan["max_cells"]),
                    "--max-images",
                    str(plan["max_images"]),
                ],
                dry_run=False,
                description="Train bridge",
            )
        )
    manifest = _write_repro_manifest(args, commands_run, blockers, plan, status="completed")
    _write_report(args, manifest, blockers, status="completed")
    _print_summary(args, manifest, commands_run, blockers, status="completed")
    return 0


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_mode = "small"
    if args.medium:
        run_mode = "medium"
    if args.full:
        run_mode = "full"
    limits = {
        "small": {"max_cells": 10000, "max_images": 10000},
        "medium": {"max_cells": 50000, "max_images": 50000},
        "full": {"max_cells": 0, "max_images": 0},
    }[run_mode]
    pairing_mode = "both" if args.paired and args.unpaired else "paired" if args.paired else "unpaired" if args.unpaired else "none"
    return {"run_mode": "dry-run" if args.dry_run else run_mode, "pairing_mode": pairing_mode, **limits}


def _ensure_dirs(output_dir: Path) -> None:
    for path in [
        ROOT / "data/raw",
        ROOT / "data/interim",
        ROOT / "data/processed/sciplex3",
        ROOT / "data/processed/bf_moa",
        ROOT / "data/processed/paired",
        ROOT / "checkpoints/real",
        output_dir / "baselines",
        output_dir / "biology",
        output_dir / "evaluation",
        ROOT / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _check_required_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not args.image_only and not DATASET_PATHS["sciplex3_raw"].exists() and not DATASET_PATHS["sciplex3_h5ad"].exists():
        blockers.append(
            {
                "missing_file": str(DATASET_PATHS["sciplex3_raw"].relative_to(ROOT)),
                "creates_or_downloads": (
                    "uv run python scripts/download_public_data.py --dataset sciplex3 --download-large "
                    "or stage a normalized h5ad at data/processed/sciplex3/rna_normalized.h5ad"
                ),
                "can_continue": "dry-run only",
            }
        )
    if not args.rna_only and not DATASET_PATHS["bfmoa_tables"].exists() and not DATASET_PATHS["bfmoa_manifest"].exists():
        blockers.append(
            {
                "missing_file": str(DATASET_PATHS["bfmoa_tables"].relative_to(ROOT)),
                "creates_or_downloads": "uv run python scripts/download_public_data.py --dataset bf-moa",
                "can_continue": "dry-run only; full images require user-supplied image root",
            }
        )
    return blockers


def _dry_run_commands(args: argparse.Namespace, plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [
        ["uv", "run", "python", "scripts/build_rna_manifest.py", "--input-h5ad", str(DATASET_PATHS["sciplex3_raw"].relative_to(ROOT)), "--output-dir", "data/processed/sciplex3"],
        ["uv", "run", "python", "scripts/build_bf_moa_manifest.py", "--data-tables", str(DATASET_PATHS["bfmoa_tables"].relative_to(ROOT)), "--output", str(DATASET_PATHS["bfmoa_raw_manifest"].relative_to(ROOT)), "--image-root", "data/raw/bf_moa_images"],
        ["uv", "run", "python", "scripts/build_image_manifest.py", "--input-manifest", str(DATASET_PATHS["bfmoa_raw_manifest"].relative_to(ROOT)), "--output-manifest", str(DATASET_PATHS["bfmoa_manifest"].relative_to(ROOT)), "--image-root", "data/raw/bf_moa_images", "--allow-missing-images"],
        ["uv", "run", "python", "scripts/build_paired_manifest.py", "--rna-metadata", str(DATASET_PATHS["sciplex3_metadata"].relative_to(ROOT)), "--image-manifest", str(DATASET_PATHS["bfmoa_manifest"].relative_to(ROOT)), "--output-manifest", str(DATASET_PATHS["paired_manifest"].relative_to(ROOT))],
        ["uv", "run", "python", "scripts/train_pretrain_rna.py", "--config", "configs/real/pretrain_rna_sciplex3.yaml", "--max-cells", str(plan["max_cells"])],
        ["uv", "run", "python", "scripts/train_pretrain_image.py", "--config", "configs/real/pretrain_image_bfmoa.yaml", "--max-images", str(plan["max_images"])],
        ["uv", "run", "python", "scripts/train_bridge.py", "--config", "configs/real/bridge_sciplex3_bfmoa.yaml", "--max-cells", str(plan["max_cells"]), "--max-images", str(plan["max_images"])],
        ["uv", "run", "python", "scripts/evaluate_real_benchmark.py", "--help"],
        ["uv", "run", "python", "scripts/run_biological_validation.py", "--help"],
    ]
    records = []
    for command in commands:
        print("$ " + " ".join(command))
        records.append({"description": "dry-run planned command", "command": command, "returncode": None, "status": "planned"})
    return records


def _run_manifest_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    commands = []
    if DATASET_PATHS["sciplex3_raw"].exists() and not DATASET_PATHS["sciplex3_h5ad"].exists():
        commands.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/build_rna_manifest.py",
                    "--input-h5ad",
                    str(DATASET_PATHS["sciplex3_raw"].relative_to(ROOT)),
                    "--output-dir",
                    "data/processed/sciplex3",
                ],
                dry_run=False,
                description="Build RNA manifest",
            )
        )
    if DATASET_PATHS["bfmoa_tables"].exists() and not DATASET_PATHS["bfmoa_raw_manifest"].exists():
        commands.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/build_bf_moa_manifest.py",
                    "--data-tables",
                    str(DATASET_PATHS["bfmoa_tables"].relative_to(ROOT)),
                    "--output",
                    str(DATASET_PATHS["bfmoa_raw_manifest"].relative_to(ROOT)),
                    "--image-root",
                    "data/raw/bf_moa_images",
                ],
                dry_run=False,
                description="Build BF-MoA raw manifest",
            )
        )
    if DATASET_PATHS["bfmoa_raw_manifest"].exists() and not DATASET_PATHS["bfmoa_manifest"].exists():
        commands.append(
            _run_or_print(
                [
                    sys.executable,
                    "scripts/build_image_manifest.py",
                    "--input-manifest",
                    str(DATASET_PATHS["bfmoa_raw_manifest"].relative_to(ROOT)),
                    "--output-manifest",
                    str(DATASET_PATHS["bfmoa_manifest"].relative_to(ROOT)),
                    "--image-root",
                    "data/raw/bf_moa_images",
                    "--allow-missing-images",
                ],
                dry_run=False,
                description="Normalize BF-MoA image manifest",
            )
        )
    return commands


def _run_or_print(command: list[str], *, dry_run: bool, description: str) -> dict[str, Any]:
    print(f"\n## {description}")
    print("$ " + " ".join(command))
    if dry_run:
        return {"description": description, "command": command, "returncode": None, "status": "planned"}
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return {"description": description, "command": command, "returncode": completed.returncode, "status": "completed"}


def _write_repro_manifest(
    args: argparse.Namespace,
    commands_run: list[dict[str, Any]],
    blockers: list[dict[str, str]],
    plan: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    manifest = {
        "status": status,
        "git_commit_hash": _git(["rev-parse", "HEAD"]),
        "branch_name": _git(["branch", "--show-current"]),
        "python_version": sys.version,
        "uv_version": _command_output(["uv", "--version"]),
        "os_platform": platform.platform(),
        "cuda_available": _cuda_available(),
        "torch_version": _torch_version(),
        "dependency_lockfile_hash": _sha256(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None,
        "command_used": " ".join(sys.argv),
        "commands_run": commands_run,
        "dataset_paths": {key: str(path.relative_to(ROOT)) for key, path in DATASET_PATHS.items()},
        "config_paths": [str(path.relative_to(ROOT)) for path in sorted((ROOT / "configs/real").glob("*.yaml"))],
        "random_seed": args.seed,
        "output_directories": {
            "results": str(args.output_dir.relative_to(ROOT) if args.output_dir.is_relative_to(ROOT) else args.output_dir),
            "reports": "reports",
            "checkpoints": "checkpoints/real",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_mode": plan["run_mode"],
        "pairing_mode": plan["pairing_mode"],
        "blockers": blockers,
    }
    path = args.output_dir / "reproducibility_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_report(args: argparse.Namespace, manifest: dict[str, Any], blockers: list[dict[str, str]], *, status: str) -> None:
    blockers_text = "\n".join(
        f"- missing `{item['missing_file']}`; create with `{item['creates_or_downloads']}`; continuation: {item['can_continue']}"
        for item in blockers
    ) or "- none recorded"
    report = f"""# Real Perturb-JEPA Benchmark Report

## Executive Summary

Verified fact: this repository implements condition-bag Perturb-JEPA Bridge with biological keys `perturbation|dose|time|cell_line`.

Implementation result: runner status is `{status}`. No full real-data training is claimed unless checkpoints and metrics were generated in this run. Run mode: `{manifest['run_mode']}`. Pairing mode: `{manifest['pairing_mode']}`.

Inference: public data can support a biologically meaningful benchmark only when learned embeddings outperform metadata-only and batch-only baselines under held-out perturbation, dose/time, batch/plate, and pairing splits.

Open uncertainty: real-data performance is not available until the required datasets below are staged and the non-dry-run pipeline completes.

## Dataset Table

Verified fact: see `docs/public_datasets.md` and `docs/paired_datasets.md` for source URLs, licenses, and pairing tiers.

Implementation result: dataset paths expected by this run are listed in `results/reproducibility_manifest.json`.

Open uncertainty / missing files:
{blockers_text}

## Model Description

Verified fact: the model contains RNA and image encoders, masked reconstruction / JEPA-style teacher targets, bridge alignment, batch adversarial diagnostics, and condition-bag counterfactual heads.

Implementation result: real configs are under `configs/real/`.

Inference: held-out perturbation extrapolation is not supported by perturbation-ID embeddings alone; descriptor features are needed before claiming out-of-support perturbation prediction.

## Split Strategy

Verified fact: biological condition keys exclude `batch`, `plate`, `well`, `site`, `z_plane`, `channel_or_z`, `sequencing_lane`, `library_id`, `image_acquisition_id`, `file_name`, and `image_path`.

Implementation result: real configs default to held-out perturbation or held-out batch splits. Manifest builders reject condition keys that include technical fields.

Open uncertainty: exact train/test overlaps are not available until real manifests are built and evaluated.

## Main Results

Implementation result: metrics are `not available` for this report if the run was dry-run or blocked before training.

Open uncertainty: retrieval, paired retrieval, counterfactual, and biological validation tables must be read from `results/evaluation/`, `results/baselines/`, and `results/biology/` after a completed run.

## Baseline Comparison

Verified fact: required baselines include metadata-only, batch-only, mean prototype oracle, mean prototype trainfit, shuffled pairing, and random embeddings.

Implementation result: `scripts/evaluate_real_benchmark.py` writes baseline CSV/JSON outputs and marks unavailable metrics explicitly.

## Paired Dataset Analysis

Verified fact: true paired public resources identified here are spatial transcriptomics image-expression datasets; BF-MoA/JUMP/RxRx1 are not RNA-image paired. Optical pooled screens pair images to barcode/guide identity, not expression.

Implementation result: `scripts/build_paired_manifest.py` infers cell/spot/tile/well/sample/condition pairing only from explicit metadata columns.

Inference: condition-only overlap supports weak condition-bag alignment, not same-cell paired learning.

## Biological Interpretation

Implementation result: `scripts/run_biological_validation.py` computes pseudobulk logFC recovery, top-k DE overlap, direction accuracy, optional GMT pathway scores, MoA summaries, and dose/time summaries.

Open uncertainty: pathway or MoA recovery must not be claimed unless those metrics were computed from observed and predicted outputs.

## Failure Modes

Verified fact: batch, plate, well, site, and library identifiers can leak perturbation identity.

Inference: strong metadata-only or batch-only retrieval means learned biological structure is not established.

## Reproducibility

Implementation result: branch `{manifest['branch_name']}`, commit `{manifest['git_commit_hash']}`, Python `{manifest['python_version'].split()[0]}`, CUDA availability `{manifest['cuda_available']}`.

Use `uv sync --all-extras --dev`, then rerun `bash scripts/run_end_to_end_real_model.sh --small` after staging data.

## Next Steps

Implementation result: remaining blockers are listed above.

Inference: for a paper-quality result, add perturbation descriptors, bootstrap confidence intervals, replicate-aware tests, seed sweeps, and at least one verified Tier 1 or Tier 3 paired task.
"""
    report_path = ROOT / "reports/real_benchmark_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def _print_blockers(blockers: list[dict[str, str]]) -> None:
    print("\nMissing required real-data inputs:")
    for item in blockers:
        print(f"- missing: {item['missing_file']}")
        print(f"  create/download: {item['creates_or_downloads']}")
        print(f"  can continue: {item['can_continue']}")


def _print_summary(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    commands_run: list[dict[str, Any]],
    blockers: list[dict[str, str]],
    *,
    status: str,
) -> None:
    print("\nFinal summary")
    print(f"- status: {status}")
    print(f"- branch: {manifest['branch_name']}")
    print(f"- git commit hash: {manifest['git_commit_hash']}")
    print(f"- dataset paths: {json.dumps(manifest['dataset_paths'], sort_keys=True)}")
    print(f"- commands run/planned: {len(commands_run)}")
    print("- training status: not run in dry-run; see checkpoints/real after non-dry-run")
    print("- validation metrics: results/evaluation/real_benchmark_metrics.csv")
    print("- biological validation summary: results/biology/interpretation.md when run")
    print(f"- remaining blockers: {len(blockers)}")
    print("- push status: not attempted by runner")
    print(f"- reproducibility manifest: {args.output_dir / 'reproducibility_manifest.json'}")
    print("- final report: reports/real_benchmark_report.md")


def _git(args: list[str]) -> str:
    return _command_output(["git", *args])


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "not available"
    if completed.returncode != 0:
        return "not available"
    return completed.stdout.strip()


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return "not available"


def _cuda_available() -> bool | str:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return "not available"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
