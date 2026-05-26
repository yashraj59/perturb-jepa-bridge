# Family O Count-Likelihood Perturbation Training

- Dataset: `synth_micro`
- Seed: `2`
- Device: `cpu`
- Biological key: `perturbation_id, cell_line_id, dose, time`
- Batch ID excluded from features and matching: `true`
- Training targets/statistics split: `train`
- Test target rows used for teacher/training construction: `0`
- Real data used: `false`
- Marker/pathway/pretrained biological assets used: `false`

## Count Path Audit

- Raw count-like RNA values available: `True`
- Count path: `raw_synthetic_observed_counts`
- Pseudo-count path used: `False`
- Pseudo-count scale: `None`
- Zero fraction: `0.2026`
- Count mean / variance: `107.9808` / `49344.0223`
- Mean library size: `13821.55`
- log mean-variance correlation: `0.9593`
- MoM dispersion median: `1.4854`

## Candidate Results

### seed2_count_audit_source_as_target
- method: `count_audit_source_as_target`
- stage: `A`
- leakage gate pass: `True`
- Poisson NLL test: `513.3838`
- NB NLL test: `NA`
- program recovery: `-0.0393`
- direction accuracy: `0.4052`
- logFC correlation: `0.1191`
- pseudobulk correlation: `0.7549`
- top50 overlap: `0.2925`
- mean delta/target ratio: `0.0000`
- counterfactual gate pass: `False`

### seed2_train_global_count_mean_poisson_baseline
- method: `train_global_count_mean_poisson_baseline`
- stage: `A`
- leakage gate pass: `True`
- Poisson NLL test: `63.2684`
- NB NLL test: `NA`
- program recovery: `0.4624`
- direction accuracy: `0.6854`
- logFC correlation: `0.7444`
- pseudobulk correlation: `0.8681`
- top50 overlap: `0.6017`
- mean delta/target ratio: `0.8888`
- counterfactual gate pass: `True`

### seed2_poisson_train_only_count_mean_table
- method: `poisson_train_only_count_mean_table`
- stage: `B`
- leakage gate pass: `True`
- Poisson NLL test: `48.4387`
- NB NLL test: `NA`
- program recovery: `0.7433`
- direction accuracy: `0.7679`
- logFC correlation: `0.7562`
- pseudobulk correlation: `0.8815`
- top50 overlap: `0.6392`
- mean delta/target ratio: `0.8357`
- counterfactual gate pass: `True`

### seed2_nb_train_only_count_mean_table
- method: `nb_train_only_count_mean_table`
- stage: `C`
- leakage gate pass: `True`
- Poisson NLL test: `48.4387`
- NB NLL test: `4.9933`
- program recovery: `0.7433`
- direction accuracy: `0.7679`
- logFC correlation: `0.7562`
- pseudobulk correlation: `0.8815`
- top50 overlap: `0.6392`
- mean delta/target ratio: `0.8357`
- counterfactual gate pass: `True`

### seed2_poisson_mlp_no_batch_condition_source
- method: `poisson_mlp_no_batch_condition_source`
- stage: `D`
- leakage gate pass: `True`
- Poisson NLL test: `133.1362`
- NB NLL test: `NA`
- program recovery: `0.4855`
- direction accuracy: `0.6532`
- logFC correlation: `0.4293`
- pseudobulk correlation: `0.6715`
- top50 overlap: `0.5433`
- mean delta/target ratio: `0.8157`
- counterfactual gate pass: `False`

### seed2_nb_mlp_no_batch_condition_source
- method: `nb_mlp_no_batch_condition_source`
- stage: `E`
- leakage gate pass: `True`
- Poisson NLL test: `148.4786`
- NB NLL test: `8.1260`
- program recovery: `0.5770`
- direction accuracy: `0.6545`
- logFC correlation: `0.4057`
- pseudobulk correlation: `0.6338`
- top50 overlap: `0.5342`
- mean delta/target ratio: `0.8386`
- counterfactual gate pass: `False`

## NLL Winners

- Best Poisson NLL: `seed2_nb_train_only_count_mean_table` with `48.4387`
- Best NB NLL: `seed2_nb_train_only_count_mean_table` with `4.9933`

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

## Interpretation

Family O tests whether count likelihood changes the seed-2 counterfactual signal rather than replacing Family N. The raw synthetic count path is available, so no pseudo-count fallback was used in this run.
Poisson and NB table candidates evaluate likelihood calibration around the train-only condition mean; MLP candidates test a learned no-batch condition/source-feature decoder with positive log-mean parameterization and stable positive dispersion for NB.

## Artifacts

- `FAMILY_O_RESULTS.tsv`
- `FAMILY_O_RESULTS.json`
- `COMPARATOR_RESULTS.tsv`
- `generation_config.json`
