# Perturb-JEPA Bridge

This repository implements the v1 scaffold for unpaired perturbation modeling across
single-cell RNA-seq and label-free microscopy. RNA and image encoders are trained
separately with masked reconstruction and JEPA teacher/student losses, then aligned
at the condition-bag level rather than by assuming paired cells and images.

The code is intentionally data-source agnostic. It provides normalized manifests,
group-safe splits, PyTorch model modules, losses, metrics, and a synthetic smoke
training entrypoint.

## Install

```bash
python -m pip install -e ".[data,dev]"
```

## Data Interfaces

Expected scRNA `AnnData` metadata:

```text
obs: perturbation, perturbation_type, dose, time, cell_line, batch
var: gene_id, gene_symbol
```

Expected image manifest columns:

```text
image_path, plate, well, site, channel_or_z, perturbation, compound, moa,
target_gene, dose, time, cell_line, batch
```

The shared condition key is:

```text
perturbation | perturbation_type | dose | time | cell_line
```

## Metadata-First Downloads

The download script prints or runs explicit public download commands. Large
image/RNA files are never pulled by default.

```bash
python scripts/download_public_data.py --dry-run --all
python scripts/download_public_data.py --dry-run --dataset bf-moa-metadata
```

For BF-MoA metadata:

```bash
curl -L "https://ndownloader.figshare.com/files/37984380" -o bf_moa_data_tables.tar.gz
python scripts/build_bf_moa_manifest.py \
  --data-tables bf_moa_data_tables.tar.gz \
  --output bf_moa_manifest.csv \
  --image-root /path/to/extracted/images
```

## Smoke Train

Run a one-step synthetic model check without real data:

```bash
python scripts/train_smoke.py --steps 2
```

For config-driven training with checkpointing:

```bash
python scripts/train_synthetic.py \
  --steps 10 \
  --checkpoint-out checkpoints/synthetic.pt
```

## Baseline Evaluation

Expression baselines expect `.npy` expression matrices and CSV metadata:

```bash
python scripts/evaluate_expression_baselines.py \
  --train-expression train_expression.npy \
  --train-metadata train_metadata.csv \
  --eval-expression eval_expression.npy \
  --eval-metadata eval_metadata.csv \
  --output expression_baselines.csv
```

Retrieval baselines expect `.npy` embedding matrices and CSV metadata:

```bash
python scripts/evaluate_retrieval_baselines.py \
  --gallery-embeddings image_centroids.npy \
  --gallery-metadata image_metadata.csv \
  --query-embeddings rna_centroids.npy \
  --query-metadata rna_metadata.csv \
  --output retrieval_baselines.csv
```

## Main Modules

- `perturb_jepa.data.schema`: metadata validation and condition keys.
- `perturb_jepa.data.conditions`: metadata vocabularies, condition bags, and prototypes.
- `perturb_jepa.data.scrna`: `SCRNATokenDataset` and collator for masked gene batches.
- `perturb_jepa.data.images`: `ImageManifestDataset` and collator for label-free image batches.
- `perturb_jepa.data.splits`: group-safe train/val/test splits.
- `perturb_jepa.models.bridge`: dual encoder model with EMA teachers.
- `perturb_jepa.losses`: masked reconstruction, JEPA, InfoNCE, MMD.
- `perturb_jepa.training.trainer`: trainer loop, loss assembly, optimizer steps, and EMA updates.
- `perturb_jepa.training.checkpoint`: checkpoint save/load helpers.
- `perturb_jepa.evaluation.metrics`: RNA/image/cross-modal retrieval metrics.
- `perturb_jepa.evaluation.baselines`: control mean, perturbation mean, centroid retrieval, and label-shuffle controls.

## Design Guardrails

- Cross-modal alignment is condition-level only; no cell-image pairing is assumed.
- Splits must be grouped by perturbation, dose, time, cell line, and batch.
- Fluorescence channels can be used as optional teacher targets, but label-free
  model input should remain brightfield or phase contrast.
- v1 predicts RNA distributions and image embeddings/prototypes, not full images.
