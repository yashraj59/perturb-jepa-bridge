# LINCS L1000 + Cell Painting Switch Report

## Dataset

Switched the real-data bridge target from sci-Plex3/BF-MoA to LINCS L1000 plus
LINCS Cell Painting processed morphology profiles.

Sources used:

- L1000 profiling complementarity data: `https://figshare.com/articles/dataset/L1000_data_for_profiling_comparison/13181966`
- LINCS Cell Painting processed release: `https://zenodo.org/records/5008187`
- LINCS profiling complementarity repository: `https://github.com/broadinstitute/lincs-profiling-complementarity`
- LINCS Cell Painting repository: `https://github.com/broadinstitute/lincs-cell-painting`

## Files Downloaded

| File | SHA256 |
|---|---|
| `data/raw/lincs_l1000/level_5_modz_n9482x978.gctx` | `b8334889886d33fe69932b9a8528207c8c426c94b4b1a804fe315ba201fb6987` |
| `data/raw/lincs_l1000/col_meta_level_5_REP.A_A549_only_n9482.txt` | `0aa98e449a2d293819f1906315590116118d67f277c7fba8f4153459c9880a5e` |
| `data/raw/lincs_l1000/REP.A_A549_pert_info.txt` | `f9bdf61df028ff2d2c57f96664bc7ef300aad0a142e6537e623a131f99692833` |
| `data/raw/lincs_cell_painting/lincs-cell-painting-v1.zip` | `3acc2e886b92cfa58c7bb25a7aed8600f3c30aef2b6fc97b051e01818d7690ca` |

Also downloaded 136 Level4b Cell Painting per-plate profile files from the
release manifest into `data/raw/lincs_cell_painting/level4b/`.

## Prepared Outputs

| Output | Description |
|---|---|
| `data/processed/lincs/l1000_a549_10uM_24h.h5ad` | L1000 A549 level-5 expression signatures for the aligned 10uM benchmark |
| `data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv` | Image-manifest-compatible Cell Painting morphology profile rows |
| `data/processed/lincs/cell_painting_profile_arrays/` | One `1 x 36 x 36` `.npy` feature-grid per Cell Painting profile |
| `metrics/lincs/compound_intersection.csv` | Per-compound overlap table |
| `metrics/lincs/prepare_lincs_summary.json` | Machine-readable preparation summary |

## Overlap

The 10uM aligned benchmark has:

| Quantity | Value |
|---|---:|
| Shared perturbation-dose conditions | 1,384 |
| Shared perturbations | 1,384 |
| L1000 profiles | 1,883 |
| Cell Painting profiles | 9,957 |
| L1000 genes | 978 |
| Cell Painting morphology features | 1,212 |

This clears the `>=6` shared-condition gate by a wide margin.

## Condition Key

The switch uses `condition_key_lincs = perturbation|dose`.

Reason: the public L1000 bundle is A549 24h, while the LINCS Cell Painting pilot
profiles are A549 48h. The times are retained in metadata and this mismatch must
be reported, but including `time` in the bridge key would prevent the intended
LINCS cross-modality comparison.

## Code Changes

- Added `scripts/prepare_lincs_pairing.py`.
- Added `configs/lincs_l1000_cellpainting.yaml`.
- Added `DataConfig.rna_normalize`; set it to `false` for L1000 because the
  input is already a processed expression signature, not raw count data.
- Propagated `rna_normalize` through RNA pretrain, bridge training, retrieval
  evaluation, and counterfactual train/eval paths.

## Smoke Checks

Commands run:

```bash
uv run python scripts/prepare_lincs_pairing.py --include-controls
uv run python scripts/train_pretrain_rna.py --config configs/lincs_l1000_cellpainting.yaml --steps 2 --device cpu --rna-anndata data/processed/lincs/l1000_a549_10uM_24h.h5ad --n-top-genes 978 --batch-size 8 --checkpoint-out checkpoints/lincs/pretrain_rna_smoke.pt
uv run python scripts/train_pretrain_image.py --config configs/lincs_l1000_cellpainting.yaml --steps 2 --device cpu --image-manifest data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv --image-root . --batch-size 8 --checkpoint-out checkpoints/lincs/pretrain_image_smoke.pt
uv run python scripts/train_bridge.py --config configs/lincs_l1000_cellpainting.yaml --steps 2 --device cpu --rna-anndata data/processed/lincs/l1000_a549_10uM_24h.h5ad --image-manifest data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv --image-root . --split-strategy heldout_perturbation --eval-split-value test --checkpoint-out checkpoints/lincs/bridge_smoke.pt
uv run python scripts/evaluate_retrieval.py --checkpoint checkpoints/lincs/bridge_smoke.pt --rna-anndata data/processed/lincs/l1000_a549_10uM_24h.h5ad --image-manifest data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv --image-root . --split-col split --eval-split-value test --save-embeddings-dir metrics/lincs/embeddings_smoke --output metrics/lincs/retrieval_smoke.csv --device cpu
uv run python scripts/evaluate_retrieval_baselines.py --gallery-embeddings metrics/lincs/embeddings_smoke/image_embeddings.npy --gallery-metadata metrics/lincs/embeddings_smoke/image_metadata.csv --query-embeddings metrics/lincs/embeddings_smoke/rna_embeddings.npy --query-metadata metrics/lincs/embeddings_smoke/rna_metadata.csv --label-col condition_key --ks 1,5,10 --output metrics/lincs/centroid_baselines_smoke.csv
uv run pytest
```

Results:

- Full test suite: `96 passed`.
- Bridge smoke: `1,107` shared training condition bags after held-out perturbation split.
- Retrieval smoke completed; the two-step metrics are only an integration check,
  not a scientific performance claim.

## Limitations

- This first switch uses processed Cell Painting morphology profiles, not raw
  microscopy pixels. They are represented as feature-grid `.npy` arrays so the
  existing image encoder path can run without multi-terabyte raw-image downloads.
- L1000 time is 24h; Cell Painting pilot time is 48h.
- The bridge key deliberately drops time for this LINCS benchmark and uses
  `perturbation|dose`; this must be stated in any report.
- A full claim still requires non-smoke training and the full leakage-safe
  evaluation suite against metadata-only, batch-only, centroid, and label-shuffle
  baselines.

## Citations

- Way et al., "Morphology and gene expression profiling provide complementary information for mapping cell state." *Cell Systems* (2022). DOI: 10.1016/j.cels.2022.10.001.
- Natoli et al., "broadinstitute/lincs-cell-painting: Full release of LINCS Cell Painting dataset." Zenodo (2021). DOI: 10.5281/zenodo.5008187.
- Natoli et al., "L1000 data for LINCS profiling complementarity analysis." figshare (2020). DOI: 10.6084/m9.figshare.13181966.v2.
