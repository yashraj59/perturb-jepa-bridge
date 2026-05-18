# LINCS Full Training Report

## Run Artifacts

Full LINCS L1000 plus LINCS Cell Painting training completed on 2026-05-18.

| Artifact | Path |
|---|---|
| RNA pretrain checkpoint | `checkpoints/lincs/pretrain_rna_full.pt` |
| image/profile pretrain checkpoint | `checkpoints/lincs/pretrain_image_full.pt` |
| bridge checkpoint | `checkpoints/lincs/bridge_full.pt` |
| counterfactual checkpoint | `checkpoints/lincs/counterfactual_full.pt` |
| retrieval metrics | `metrics/lincs/retrieval_full.csv` |
| centroid baselines | `metrics/lincs/centroid_baselines_full.csv` |
| counterfactual metrics | `metrics/lincs/counterfactual_full.csv` |
| expression baselines | `metrics/lincs/expression_baselines_full.csv` |

## Data

The run uses `condition_key_lincs = perturbation|dose` over the aligned LINCS
A549 benchmark:

| Quantity | Value |
|---|---:|
| shared perturbation-dose conditions | 1,384 |
| L1000 profiles | 1,883 |
| Cell Painting profile rows | 9,957 |
| L1000 genes | 978 |
| Cell Painting morphology features | 1,212 |

The bridge training split retained `1,107` shared biological condition bags after
the held-out perturbation split. DMSO controls were retained in the L1000 training
set for counterfactual and expression baselines.

## Configuration

Primary config: `configs/lincs_l1000_cellpainting.yaml`.

Important run choices:

- RNA data: `data/processed/lincs/l1000_a549_10uM_24h.h5ad`
- image/profile manifest: `data/processed/lincs/cell_painting_manifest_10uM_48h_profiles.csv`
- image root: `.`
- split strategy: `heldout_perturbation`
- eval split: `test`
- RNA normalization: `false`, because L1000 profiles are already processed expression signatures.

## Training Curves

| Stage | First logged value | Last logged value | Late-window note |
|---|---:|---:|---|
| RNA pretrain `rna_mask` | 2.5674 | 1.9812 | last-5 mean 1.4616; noisy final log |
| image/profile pretrain `image_mask` | 3.5704 | 0.6807 | clear decrease |
| bridge `align` | 2.8049 | 2.2444 | lower than step 100 value 2.7746 |
| counterfactual `counterfactual_nll` | 1.2125 | -0.0177 | clear decrease |

Bridge batch-adversary logs at step 1950:

| Term | Value | Reference |
|---|---:|---:|
| `rna_batch_adv` | 1.7268 | `log(138) = 4.9273` |
| `image_batch_adv` | 4.6430 | `log(138) = 4.9273` |

The RNA adversary is below half of `log(num_batches)`, which is a leakage warning.
The image adversary remains close to chance.

## Retrieval Metrics

| Method | mAP | recall@1 | recall@5 | recall@10 | median rank | same-MoA enrichment@10 |
|---|---:|---:|---:|---:|---:|---:|
| learned RNA to image | 0.051539 | 0.010830 | 0.050542 | 0.104693 | 64 | 2.200065 |
| batch-only | 0.022394 | 0.003610 | 0.018051 | 0.036101 | 139 | NA |
| mean prototype oracle | 0.095567 | 0.028881 | 0.090253 | 0.220217 | 24 | 4.110332 |
| metadata-only | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1 | NA |

The metadata-only row is a key-matching diagnostic and is not directly comparable
to learned embedding same-MoA enrichment. It remains included because the required
evaluation suite emits it.

## Centroid And Label Shuffle

| Baseline | mAP | recall@1 | recall@5 | recall@10 |
|---|---:|---:|---:|---:|
| centroid retrieval | 0.051539 | 0.010830 | 0.050542 | 0.104693 |
| label shuffle centroid | 0.021302 | 0.000000 | 0.025271 | 0.036101 |

The centroid result beats label shuffle, but the mAP margin is `0.030237`, below
the requested `0.05` margin.

## Counterfactual

| Metric | Value |
|---|---:|
| RNA pseudobulk correlation | 0.978836 |
| RNA logFC correlation | 0.937393 |
| RNA direction accuracy | 0.800772 |
| RNA top20 DE overlap | 0.242733 |
| RNA top50 DE overlap | 0.342068 |
| RNA top100 DE overlap | 0.440506 |

## Expression Baselines

Expression baselines were run from the actual held-out perturbation split rather
than the paired counterfactual export, because that export only labels treated
conditions and does not contain control rows for fitting `ControlMeanBaseline`.

| Baseline | Group | n | pearson_delta | delta_mae | top50_de_recovery |
|---|---|---:|---:|---:|---:|
| control mean | overall | 277 | 0.000000 | 0.903065 | 0.075307 |
| perturbation mean | overall | 277 | 0.000000 | 0.903065 | 0.075307 |

All eval perturbations are held out, so `perturbation_mean` falls back to the
control mean on this split.

## Acceptance Check

| Criterion | Result | Evidence |
|---|---|---|
| at least 6 shared conditions | PASS | 1,384 aligned conditions; 1,107 train bags |
| image pretrain loss drops at least 30 percent | PASS | 3.5704 to 0.6807 |
| RNA pretrain loss drops at least 30 percent | MIXED | last logged 1.9812 is only 22.8 percent lower than step 0; last-5 mean is 43.1 percent lower |
| bridge align loss decreases | PASS | 2.7746 at step 100 to 2.2444 at step 1950 |
| learned beats batch-only recall@5 by 1.5x | PASS | 0.050542 versus 1.5 x 0.018051 = 0.027076 |
| learned beats metadata-only same-MoA check | PASS by the literal requested comparison | 2.200065 enrichment versus metadata-only recall@10 of 1.0 |
| centroid beats label-shuffle by mAP margin >= 0.05 | FAIL | margin is 0.030237 |
| learned beats mean prototype oracle | FAIL | learned mAP 0.051539 versus oracle mAP 0.095567 |
| batch adversarial pressure is near chance | FAIL for RNA, PASS for image | RNA 1.7268 is below half of log(138); image 4.6430 is near log(138) |

## Result

This is a completed full training run, but it is not a clean publishable win.
The learned bridge shows signal over batch-only and label-shuffle controls, but
the oracle is stronger, the label-shuffle margin is too small, and the RNA shared
space still appears to leak batch information late in bridge training.

## Limitations

- This run uses processed Cell Painting morphology profiles projected to
  `1 x 36 x 36` feature grids, not raw microscopy pixels.
- L1000 profiles are A549 at 24h; LINCS Cell Painting profiles are A549 at 48h.
- The bridge key drops time and uses `perturbation|dose` to make the LINCS
  cross-modality benchmark possible.
- The held-out perturbation expression baseline cannot learn per-perturbation
  means for eval compounds, so it falls back to DMSO/control mean.

## Citations

- Way et al., "Morphology and gene expression profiling provide complementary
  information for mapping cell state." Cell Systems (2022). DOI:
  10.1016/j.cels.2022.10.001.
- Natoli et al., "broadinstitute/lincs-cell-painting: Full release of LINCS
  Cell Painting dataset." Zenodo (2021). DOI: 10.5281/zenodo.5008187.
- Natoli et al., "L1000 data for LINCS profiling complementarity analysis."
  figshare (2020). DOI: 10.6084/m9.figshare.13181966.v2.
