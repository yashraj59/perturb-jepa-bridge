# Real Perturb-JEPA Benchmark

This guide runs Perturb-JEPA Bridge as a real-data, leakage-aware benchmark. It
does not assume same-cell RNA-image pairing. The biological condition key is
always `perturbation|dose|time|cell_line`; technical fields such as `batch`,
`plate`, `well`, `site`, `channel_or_z`, `sequencing_lane`, `library_id`, and
`image_path` are diagnostics only.

## Run everything from scratch

```bash
# 1. Install uv and create environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync --all-extras --dev

# 2. Check dataset download commands
uv run python scripts/download_public_data.py --dry-run --dataset all-metadata

# 3. Run small real-data benchmark
bash scripts/run_end_to_end_real_model.sh --small

# 4. Run paired-data benchmark if paired data is available
bash scripts/run_end_to_end_real_model.sh --small --paired

# 5. Run evaluation only
uv run python scripts/evaluate_real_benchmark.py --help

# 6. Open final report
cat reports/real_benchmark_report.md
```

## Expected Data Layout

Use these paths unless overriding configs:

```text
data/raw/SrivatsanTrapnell2020_sciplex3.h5ad
data/raw/bf_moa_data_tables.tar.gz
data/raw/bf_moa_images/
data/processed/sciplex3/rna_normalized.h5ad
data/processed/sciplex3/rna_metadata.csv
data/processed/bf_moa/image_manifest.csv
data/processed/paired/paired_manifest.csv
```

Large images are not downloaded automatically. Stage them under the configured
image root or run manifest commands with `--allow-missing-images` for metadata
and dry-run checks.

## Download and Manifest Commands

```bash
uv run python scripts/download_public_data.py --dry-run --dataset all-metadata
uv run python scripts/download_public_data.py --dataset bf-moa
uv run python scripts/download_public_data.py --dataset sciplex3 --download-large

uv run python scripts/build_rna_manifest.py \
  --input-h5ad data/raw/SrivatsanTrapnell2020_sciplex3.h5ad \
  --output-dir data/processed/sciplex3

uv run python scripts/build_bf_moa_manifest.py \
  --data-tables data/raw/bf_moa_data_tables.tar.gz \
  --output data/interim/bf_moa_manifest_raw.csv \
  --image-root data/raw/bf_moa_images

uv run python scripts/build_image_manifest.py \
  --input-manifest data/interim/bf_moa_manifest_raw.csv \
  --output-manifest data/processed/bf_moa/image_manifest.csv \
  --image-root data/raw/bf_moa_images \
  --allow-missing-images

uv run python scripts/build_paired_manifest.py \
  --rna-metadata data/processed/sciplex3/rna_metadata.csv \
  --image-manifest data/processed/bf_moa/image_manifest.csv \
  --output-manifest data/processed/paired/paired_manifest.csv
```

## Training and Evaluation

```bash
uv run python scripts/train_pretrain_rna.py --config configs/real/pretrain_rna_sciplex3.yaml
uv run python scripts/train_pretrain_image.py --config configs/real/pretrain_image_bfmoa.yaml
uv run python scripts/train_bridge.py --config configs/real/bridge_sciplex3_bfmoa.yaml
uv run python scripts/train_counterfactual.py --config configs/real/counterfactual_rna_sciplex3.yaml
uv run python scripts/evaluate_real_benchmark.py --help
uv run python scripts/run_biological_validation.py --help
```

The main runner writes `results/reproducibility_manifest.json` and
`reports/real_benchmark_report.md`.

## Small and Larger GPUs

Use `configs/real/full_train_small_gpu.yaml` for staged tests on limited GPUs.
Use `configs/real/full_train_medium_gpu.yaml` after manifests, image roots, and
split diagnostics are verified. Start with `--small`; use `--medium` or `--full`
only after the metadata-only and batch-only baselines are understood.

## Interpreting Results

Learned retrieval is meaningful only if it exceeds metadata-only and batch-only
baselines on leakage-safe splits. Tier 1 paired results require explicit
cell/spot/tile pairing keys. Well, sample, compound, or condition overlap is
weakly paired condition-bag learning, not same-cell RNA-image learning.

## Troubleshooting

- Missing `.h5ad`: run the downloader with `--dataset sciplex3 --download-large`
  or stage a compatible AnnData file at the configured path.
- Missing BF-MoA images: build metadata with `--allow-missing-images`, then stage
  raw images before image training.
- Missing labels: metrics will be written as `not available` with the reason.
- Unexpected high performance: inspect `results/baselines/` and
  `results/evaluation/leakage_diagnostics.csv` before making biological claims.
