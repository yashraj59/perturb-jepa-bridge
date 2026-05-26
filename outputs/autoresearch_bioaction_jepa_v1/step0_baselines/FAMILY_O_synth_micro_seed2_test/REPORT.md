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

## Interpretation

Family O tests whether count likelihood changes the seed-2 counterfactual signal rather than replacing Family N. The raw synthetic count path is available, so no pseudo-count fallback was used in this run.
Poisson and NB table candidates evaluate likelihood calibration around the train-only condition mean; MLP candidates test a learned no-batch condition/source-feature decoder with positive log-mean parameterization and stable positive dispersion for NB.

## Artifacts

- `FAMILY_O_RESULTS.tsv`
- `FAMILY_O_RESULTS.json`
- `COMPARATOR_RESULTS.tsv`
- `generation_config.json`
