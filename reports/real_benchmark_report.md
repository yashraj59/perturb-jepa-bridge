# Real Perturb-JEPA Benchmark Report

## Executive Summary

Verified fact: this repository implements condition-bag Perturb-JEPA Bridge with biological keys `perturbation|dose|time|cell_line`.

Implementation result: runner status is `dry_run`. No full real-data training is claimed unless checkpoints and metrics were generated in this run. Run mode: `dry-run`. Pairing mode: `none`.

Inference: public data can support a biologically meaningful benchmark only when learned embeddings outperform metadata-only and batch-only baselines under held-out perturbation, dose/time, batch/plate, and pairing splits.

Open uncertainty: real-data performance is not available until the required datasets below are staged and the non-dry-run pipeline completes.

## Dataset Table

Verified fact: see `docs/public_datasets.md` and `docs/paired_datasets.md` for source URLs, licenses, and pairing tiers.

Implementation result: dataset paths expected by this run are listed in `results/reproducibility_manifest.json`.

Open uncertainty / missing files:
- missing `data/raw/SrivatsanTrapnell2020_sciplex3.h5ad`; create with `uv run python scripts/download_public_data.py --dataset sciplex3 --download-large or stage a normalized h5ad at data/processed/sciplex3/rna_normalized.h5ad`; continuation: dry-run only
- missing `data/raw/bf_moa_data_tables.tar.gz`; create with `uv run python scripts/download_public_data.py --dataset bf-moa`; continuation: dry-run only; full images require user-supplied image root

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

Implementation result: branch `codex/leakage-safe-bridge-colab`, commit `ee537b39077ca3023016786835c7271cedae24b9`, Python `3.11.15`, CUDA availability `False`.

Use `uv sync --all-extras --dev`, then rerun `bash scripts/run_end_to_end_real_model.sh --small` after staging data.

## Next Steps

Implementation result: remaining blockers are listed above.

Inference: for a paper-quality result, add perturbation descriptors, bootstrap confidence intervals, replicate-aware tests, seed sweeps, and at least one verified Tier 1 or Tier 3 paired task.
