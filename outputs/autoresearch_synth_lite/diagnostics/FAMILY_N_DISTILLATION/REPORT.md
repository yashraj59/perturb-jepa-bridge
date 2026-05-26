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

### seed2_no_batch_matched_perturbed_mean
- source: `Family M`
- program recovery: `0.7520`
- direction accuracy: `0.6899`
- logFC correlation: `0.7502`
- pseudobulk correlation: `0.8725`
- top50 overlap: `0.5683`

### seed2_no_batch_residualized_matching
- source: `Family M`
- program recovery: `0.3502`
- direction accuracy: `0.5312`
- logFC correlation: `0.1268`
- pseudobulk correlation: `0.7491`
- top50 overlap: `0.4150`

### synth_micro_sparsepert_deltaonly_rank4_topde50_topprog2_gdw0p05_tdw8p0_pgw4p0_dw1p0_sw0p25_spw0p02_dcw0p01_lr0p001_wd0p0001_bs8_seed2_pls3_s100
- source: `Family L`
- program recovery: `0.2877`
- direction accuracy: `0.5793`
- logFC correlation: `-0.0830`
- pseudobulk correlation: `0.7536`
- top50 overlap: `0.4058`

### synth_micro_sparsepert_deltaonly_rank8_topde50_topprog2_gdw0p5_tdw12p0_pgw2p0_dw1p0_sw0p25_spw0p002_dcw0p001_lr0p003_wd0p0001_bs8_seed2_pls3_s150
- source: `Family L`
- program recovery: `0.2750`
- direction accuracy: `0.5622`
- logFC correlation: `0.1088`
- pseudobulk correlation: `0.7318`
- top50 overlap: `0.3917`

### synth_micro_pertfrozen_rnaresidual_pfact_nowres_srcprog_metactx_linearprog_prefitridge_absmse_dw0p2_pw4p0_lr0p001_wd0p0001_bs8_ra0p01_rr4_seed2_rank3_s50
- source: `prefit ridge best`
- program recovery: `0.2494`
- direction accuracy: `0.5587`
- logFC correlation: `0.0636`
- pseudobulk correlation: `0.7652`
- top50 overlap: `0.4258`

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
