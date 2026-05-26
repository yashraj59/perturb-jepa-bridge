# Baseline Registry

## Active Model Of Record

- Protected model: closed-form rank-3 train-split-only PLS readout
- RNA readout: `raw_linear_pseudobulk`
- Image readout: `raw_linear_pooled`
- Role: protected baseline / initializer / audit reference only
- Promotion rule: only a Tier 3 BioAction-JEPA pass can supersede this model
- PLS restrictions: may be used only as protected baseline or short annealed bootstrap teacher; never final BioAction-JEPA representation path

## Protected Synthetic Shared Geometry

Dataset: `synth_micro`
Seeds: `0,1,2`
Split: exact-key `test`
Source: `outputs/autoresearch_synth_lite/CURRENT_STATUS_AND_BEST_MODEL_CODE.md`

| Metric | Value | Direction |
| --- | ---: | --- |
| RNA->image recall@1 | 0.2396 +/- 0.0295 | higher |
| RNA->image recall@5 | 0.6667 +/- 0.1284 | higher |
| RNA latent R2 | 0.5929 +/- 0.1395 | higher |
| Image latent R2 | 0.9134 +/- 0.0206 | higher |
| Batch balanced accuracy | 0.4792 +/- 0.0780 | lower / no biological domination |

## Family N Expression-Space Reference

Dataset: `synth_micro`
Seed: `2`
Split: exact-key `test`
Source: `outputs/autoresearch_synth_lite/CURRENT_STATUS_AND_BEST_MODEL_CODE.md`

| Metric | Value | Direction |
| --- | ---: | --- |
| Exact train biological-key coverage | 1.0000 | diagnostic only |
| Program recovery | 0.7520 | higher |
| Direction accuracy | 0.6899 | higher |
| logFC correlation | 0.7502 | higher |
| Pseudobulk correlation | 0.8725 | higher |
| Top50 DE overlap | 0.5683 | higher |
| Mean delta/target ratio | 0.7400 | bounded near 1 |

## Family O Count-Aware Reference

Dataset: `synth_micro`
Seed: `2`
Split: exact-key `test`
Source: `outputs/autoresearch_synth_lite/CURRENT_STATUS_AND_BEST_MODEL_CODE.md`

| Metric | Value | Direction |
| --- | ---: | --- |
| Exact train biological-key coverage | 1.0000 | diagnostic only |
| Program recovery | 0.7433 | higher |
| Direction accuracy | 0.7679 | higher |
| logFC correlation | 0.7562 | higher |
| Pseudobulk correlation | 0.8815 | higher |
| Top50 DE overlap | 0.6392 | higher |
| Poisson NLL | 48.4387 | lower |
| NB NLL | 4.9933 | lower |

## Caveats

- Exact-key `test` is matching-baseline dominated and cannot support novelty claims.
- Held-out perturbation and held-out dose splits are required for serious BioAction-JEPA evaluation.
- Step 0 reruns for this BioAction session are written under `step0_baselines/` and should be treated as verification artifacts, not as new model promotion.

## BioAction Step 0 Verification Artifacts

Source directory: `outputs/autoresearch_bioaction_jepa_v1/step0_baselines/`

| Dataset | Split | Seed | Source file | Exact match fraction | Key baseline numbers |
| --- | --- | ---: | --- | ---: | --- |
| `synth_micro` | `test` | 2 | `FAMILY_M_synth_micro_seed2_test/FAMILY_M_RESULTS.tsv` | 1.0000 | matched mean program `0.7520`; pseudobulk `0.8725`; protected PLS RNA->image recall@1 `0.28125`; batch probe balanced `0.3750` |
| `synth_micro` | `test` | 2 | `FAMILY_N_synth_micro_seed2_test/FAMILY_N_RESULTS.tsv` | 1.0000 | Family N program `0.7520`; direction `0.6899`; logFC `0.7502`; top50 `0.5683` |
| `synth_micro` | `test` | 2 | `FAMILY_O_synth_micro_seed2_test/FAMILY_O_RESULTS.tsv` | 1.0000 | Family O program `0.7433`; direction `0.7679`; logFC `0.7562`; top50 `0.6392`; NB NLL `4.9933` |
| `synth_heldout_perturbation_lite` | `test_heldout_perturbation` | 2 | `FAMILY_M_synth_heldout_perturbation_seed2_test_heldout_perturbation/FAMILY_M_RESULTS.tsv` | 0.0000 | matched mean program `0.1168`; pseudobulk `0.9613`; protected PLS RNA->image recall@1 `0.1852`; batch probe balanced `0.1975` |
| `synth_dose_extrapolation_lite` | `test_heldout_dose` | 2 | `FAMILY_M_synth_dose_extrapolation_seed2_test_heldout_dose/FAMILY_M_RESULTS.tsv` | 0.0000 | matched mean program `0.0892`; pseudobulk `0.9946`; protected PLS RNA->image recall@1 `0.1806`; batch probe balanced `0.2037` |

## BioAction Initial Encoder Baselines

Zero-step BioAction encoder-only baselines were created to satisfy the Tier 1 comparison against random/initial encoders:

| Dataset | Split | Source | RNA->image recall@1 | Image->RNA recall@1 | Transition-source improvement | Joint batch probe balanced |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `synth_heldout_perturbation_lite` | `test_heldout_perturbation` | `baselines/INIT_familyA_heldout_perturb_seed0_s32/metrics_eval.json` | 0.0000 | 0.03125 | -0.03882 | 0.4758 |
| `synth_dose_extrapolation_lite` | `test_heldout_dose` | `baselines/INIT_familyA_heldout_dose_seed0_s32/metrics_eval.json` | 0.0625 | 0.03125 | -0.06180 | 0.5429 |
| `synth_micro` | `test` | `baselines/INIT_familyA_synth_micro_seed0_s32/metrics_eval.json` | 0.03125 | 0.03125 | -0.04248 | 0.8824 |
