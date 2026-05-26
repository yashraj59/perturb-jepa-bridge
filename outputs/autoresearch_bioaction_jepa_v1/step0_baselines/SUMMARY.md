# Step 0 Baseline Summary

Controlling prompt: `prompt/codex_novel_bioaction_jepa_autoresearch_prompt.md`

Step 0 was run before BioAction architecture search. These artifacts are verification baselines only; they do not promote any BioAction candidate.

## Protected Model Of Record

The active protected model remains the rank-3 train-split-only PLS readout installed into raw-linear RNA/image heads. Family N and Family O remain expression/count audit references.

## Verification Runs

| Run | Dataset | Split | Seed | Key result |
| --- | --- | --- | ---: | --- |
| `FAMILY_M_synth_micro_seed2_test` | `synth_micro` | `test` | 2 | exact match fraction `1.0`; protected PLS RNA->image recall@1 `0.28125`; protected RNA latent R2 `0.6069` |
| `FAMILY_N_synth_micro_seed2_test` | `synth_micro` | `test` | 2 | Family N train-only condition mean program `0.7520`, direction `0.6899`, logFC `0.7502` |
| `FAMILY_O_synth_micro_seed2_test` | `synth_micro` | `test` | 2 | Family O count-aware table program `0.7433`, direction `0.7679`, top50 `0.6392`, NB NLL `4.9933` |
| `FAMILY_M_synth_heldout_perturbation_seed2_test_heldout_perturbation` | `synth_heldout_perturbation_lite` | `test_heldout_perturbation` | 2 | exact match fraction `0.0`; matched mean program `0.1168`; protected PLS RNA->image recall@1 `0.1852` |
| `FAMILY_M_synth_dose_extrapolation_seed2_test_heldout_dose` | `synth_dose_extrapolation_lite` | `test_heldout_dose` | 2 | exact match fraction `0.0`; matched mean program `0.0892`; protected PLS RNA->image recall@1 `0.1806` |

## Interpretation

The exact `synth_micro/test` split is lookup/matching dominated and cannot support novelty claims by itself. Held-out perturbation and held-out dose splits are the relevant checks for BioAction-JEPA transition generalization. The protected baseline and Family N/O references are still stronger than the current BioAction candidates on the registered expression-space and protected retrieval metrics.
