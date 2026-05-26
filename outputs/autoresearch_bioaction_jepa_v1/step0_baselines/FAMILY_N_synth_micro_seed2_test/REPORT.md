# Family N Train-Only Matched-Mean Distillation

- Dataset: `synth_micro`
- Seed: `2`
- Device: `cpu`
- Biological key: `perturbation_id, cell_line_id, dose, time`
- Batch ID excluded from matching and features: `true`
- Teacher targets fit on train split only: `true`
- Test target rows used for teacher construction: `0`
- Real data used: `false`
- Marker/pathway/pretrained biological assets used: `false`

## Source-As-Target Reference

- program recovery: `0.0000`
- direction accuracy: `0.0000`
- logFC correlation: `0.0000`
- pseudobulk correlation: `0.7549`
- top50 overlap: `0.4058`

## Core Candidates

### seed2_train_only_condition_mean_table
- method: `train_only_condition_mean_table`
- exact train key coverage on test: `1.0000`
- leakage gate pass: `True`
- program recovery: `0.7520`
- direction accuracy: `0.6899`
- logFC correlation: `0.7502`
- pseudobulk correlation: `0.8725`
- top50 overlap: `0.5683`
- mean delta/target ratio: `0.7400`
- counterfactual gate pass: `True`

### seed2_distilled_linear_condition_mean
- method: `distilled_linear_condition_mean`
- exact train key coverage on test: `1.0000`
- leakage gate pass: `True`
- program recovery: `0.7353`
- direction accuracy: `0.6854`
- logFC correlation: `0.7261`
- pseudobulk correlation: `0.8506`
- top50 overlap: `0.5583`
- mean delta/target ratio: `0.7422`
- counterfactual gate pass: `True`

### seed2_distilled_mlp_condition_mean
- method: `distilled_mlp_condition_mean`
- exact train key coverage on test: `1.0000`
- leakage gate pass: `True`
- program recovery: `0.7033`
- direction accuracy: `0.6738`
- logFC correlation: `0.7286`
- pseudobulk correlation: `0.8574`
- top50 overlap: `0.5525`
- mean delta/target ratio: `0.7205`
- counterfactual gate pass: `True`

## Best Shrinkage Hybrid

### seed2_linear_condition_mean_hybrid_alpha0p1
- method: `shrinkage_distillation_hybrid`
- exact train key coverage on test: `1.0000`
- leakage gate pass: `True`
- program recovery: `0.7526`
- direction accuracy: `0.6882`
- logFC correlation: `0.7500`
- pseudobulk correlation: `0.8717`
- top50 overlap: `0.5658`
- mean delta/target ratio: `0.7389`
- counterfactual gate pass: `True`

## Required Comparators

### seed2_train_only_condition_mean_table
- source: `Family N`
- program recovery: `0.7520`
- direction accuracy: `0.6899`
- logFC correlation: `0.7502`
- pseudobulk correlation: `0.8725`
- top50 overlap: `0.5683`

### seed2_distilled_linear_condition_mean
- source: `Family N`
- program recovery: `0.7353`
- direction accuracy: `0.6854`
- logFC correlation: `0.7261`
- pseudobulk correlation: `0.8506`
- top50 overlap: `0.5583`

### seed2_distilled_mlp_condition_mean
- source: `Family N`
- program recovery: `0.7033`
- direction accuracy: `0.6738`
- logFC correlation: `0.7286`
- pseudobulk correlation: `0.8574`
- top50 overlap: `0.5525`

## Interpretation

Candidate A is the leakage-safe train-only teacher and should match the Family M direct matched perturbed mean when exact train biological keys cover the test split.
The learned students are useful only if they approximate that teacher without batch features or test target statistics; they do not change protected bridge geometry because no bridge weights are trained.

## Artifacts

- `FAMILY_N_RESULTS.tsv`
- `FAMILY_N_RESULTS.json`
- `COMPARATOR_RESULTS.tsv`
- `generation_config.json`
