# Baseline Registry

## Active Model Of Record

- Protected model: rank-3 train-split-only PLS raw-linear readout
- RNA readout: `raw_linear_pseudobulk`
- Image readout: `raw_linear_pooled`
- Role: protected baseline / initializer / audit reference only
- Promotion rule: only a Tier 3 BioTech-JEPA pass can supersede this model
- PLS restriction: may be used only as protected baseline or short annealed bootstrap teacher; never final BioTech-JEPA representation path

## Fixed References From Phase 1

| Dataset | Split | Metric | Value |
| --- | --- | --- | ---: |
| `synth_micro` | `test` | exact match fraction | 1.0000 |
| `synth_micro` | `test` | protected PLS RNA->image recall@1 | 0.28125 |
| `synth_heldout_perturbation_lite` | `test_heldout_perturbation` | exact match fraction | 0.0000 |
| `synth_heldout_perturbation_lite` | `test_heldout_perturbation` | protected PLS RNA->image recall@1 | 0.1852 |
| `synth_dose_extrapolation_lite` | `test_heldout_dose` | exact match fraction | 0.0000 |
| `synth_dose_extrapolation_lite` | `test_heldout_dose` | protected PLS RNA->image recall@1 | 0.1806 |

## Previous BioAction Audit References

| Experiment | Dataset/split | RNA->image recall@1 | Transition improvement | Joint batch probe |
| --- | --- | ---: | ---: | ---: |
| EXP001 | `synth_heldout_perturbation_lite/test_heldout_perturbation` | 0.0625 | +0.00968 | 0.4787 vs majority 0.4063 |
| EXP002 | `synth_dose_extrapolation_lite/test_heldout_dose` | 0.03125 | +0.02624 | 0.6316 vs majority 0.5000 |
| EXP003 | `synth_micro/test` | 0.0000 | +0.04168 | 0.9412 vs majority 0.5313 |
| EXP005 | `synth_micro/test` | 0.0000 | +0.03042 | 0.9412 vs majority 0.5313 |

## Genetic Amendment References

The user clarified that Norman/CRISPR-style genetic perturbation should not require a chemical dose axis.

| Dataset | Split | Seed list | Metric | Value |
| --- | --- | --- | --- | ---: |
| `synth_genetic_anchor_lite` | `test_heldout_perturbation` | `0,1,2` | train biological-key cross-batch anchor fraction | 1.0000 |
| `synth_genetic_anchor_lite` | `test_heldout_perturbation` | `0,1,2` | eval biological-key cross-batch replicate fraction | 1.0000 |
| `synth_genetic_anchor_lite` | `test_heldout_perturbation` | `0,1,2` | max split-half RNA->image same-bio recall@1 | 0.6389 |
| `synth_genetic_anchor_lite` | `test_heldout_perturbation` | `0,1,2` | max raw/protected batch-probe excess over majority | 0.6667 |

Norman context reference:

- File: `data/raw/gears_norman/norman/perturb_processed.h5ad`
- Shape: `91,205 x 5,045`
- Conditions: `284`
- Cell type: `A549`
- Exposed batch metadata: none
- Chemical dose axis: none; `dose_val` is `1` or `1+1` guide-count notation

## BioTech-JEPA Diagnostic References

These are not protected baselines and do not supersede PLS.

| Experiment | Dataset/split | Seed | RNA->image recall@1 | Transition gain | Batch allocation |
| --- | --- | ---: | ---: | ---: | --- |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | 0.1875 | +0.0161 | joint `z_tech` batch probe 0.4375 vs `z_bio` 0.1875 |
| `BTJ002` | `norman_gears_processed/test` | 0 | n/a RNA-only | +0.0313 | n/a; no exposed batch metadata |

BioTech identity checks for both runs:

- encoder path used: `1.0`
- PLS raw-linear main path: `0.0`
- exact condition-key features: `0.0`
- separate `z_bio` and `z_tech` latents: `1.0`
- teacher stop-gradient verified: `1.0`
