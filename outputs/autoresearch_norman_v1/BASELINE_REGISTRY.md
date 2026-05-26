# Baseline Registry

## Model Of Record

- Active published baseline: GEARS published Norman 2019 numbers.
- Active carried reference: Family N train-only condition-mean table recomputed on the Norman split.
- Rebasing rule: no Step 0 or Tier 1/Tier 2 result can supersede GEARS; only a Tier 3 pass can rebase.

## Dataset And Split

- Dataset file: `data/raw/gears_norman/norman/perturb_processed.h5ad`
- Source URL: https://dataverse.harvard.edu/api/access/datafile/6154020
- Conditions: `284`
- Genes: `5045`
- Canonical split implementation: `perturb_jepa/data/norman2019.py::gears_simulation_split`.
- Split fidelity unit test: `tests/test_norman2019_split.py`.
- Known metadata caveat: the processed GEARS h5ad has `cell_type == A549`; the Norman/GEARS text describes the screen as K562. This run preserves the file metadata and records the ambiguity.

## Split Counts

| seed   | train_gene_set_size | combo_seen2_train_frac | train_conditions | val_conditions | test_conditions | exact_train_combo | unseen_single | test_combo_seen0 | test_combo_seen1 | test_combo_seen2 | test_unseen_single |
| ------ | ------------------- | ---------------------- | ---------------- | -------------- | --------------- | ----------------- | ------------- | ---------------- | ---------------- | ---------------- | ------------------ |
| 1.0000 | 0.7500              | 0.7500                 | 138.0000         | 30.0000        | 116.0000        | 14.0000           | 37.0000       | 9.0000           | 52.0000          | 18.0000          | 37.0000            |

## Recomputed Baselines

| baseline                             | subset            | condition_count | pearson_delta_all_mean | pearson_delta_de20_mean | top20_de_overlap_mean | mse_delta_all_mean | mse_expression_de20_mean | direction_accuracy_de20_mean | prediction_delta_variance_ratio | mean_collapse_flag |
| ------------------------------------ | ----------------- | --------------- | ---------------------- | ----------------------- | --------------------- | ------------------ | ------------------------ | ---------------------------- | ------------------------------- | ------------------ |
| global_train_mean                    | all_test          | 116             | 0.5573                 | 0.6644                  | 0.2457                | 0.0045             | 0.3889                   | 0.8125                       | 0.0000                          | True               |
| global_train_mean                    | exact_train_combo | 14              | 0.5458                 | 0.6508                  | 0.2393                | 0.0030             | 0.3389                   | 0.8500                       | 0.0000                          | True               |
| global_train_mean                    | unseen_single     | 37              | 0.4983                 | 0.5241                  | 0.2284                | 0.0025             | 0.2171                   | 0.7392                       | 0.0000                          | True               |
| single_perturbation_additive         | all_test          | 116             | 0.5998                 | 0.7024                  | 0.2862                | 0.0043             | 0.3549                   | 0.8306                       | 0.1359                          | False              |
| single_perturbation_additive         | exact_train_combo | 14              | 0.8981                 | 0.9652                  | 0.5750                | 0.0016             | 0.0574                   | 1.0000                       | 1.4616                          | False              |
| single_perturbation_additive         | unseen_single     | 37              | 0.4983                 | 0.5241                  | 0.2284                | 0.0025             | 0.2171                   | 0.7392                       | 0.0000                          | True               |
| family_n_condition_mean_table        | all_test          | 116             | 0.5573                 | 0.6644                  | 0.2457                | 0.0045             | 0.3889                   | 0.8125                       | 0.0000                          | True               |
| family_n_condition_mean_table        | exact_train_combo | 14              | 0.5458                 | 0.6508                  | 0.2393                | 0.0030             | 0.3389                   | 0.8500                       | 0.0000                          | True               |
| family_n_condition_mean_table        | unseen_single     | 37              | 0.4983                 | 0.5241                  | 0.2284                | 0.0025             | 0.2171                   | 0.7392                       | 0.0000                          | True               |
| closed_form_pls_perturbation_readout | all_test          | 116             | 0.6215                 | 0.7012                  | 0.3250                | 0.0037             | 0.3046                   | 0.8461                       | 0.1963                          | False              |
| closed_form_pls_perturbation_readout | exact_train_combo | 14              | 0.6413                 | 0.7472                  | 0.3429                | 0.0031             | 0.2385                   | 0.9071                       | 0.7434                          | False              |
| closed_form_pls_perturbation_readout | unseen_single     | 37              | 0.5058                 | 0.5140                  | 0.2351                | 0.0025             | 0.2214                   | 0.7473                       | 0.0000                          | True               |

## Published Baselines

| baseline | source | MSE | Pearson DE | top-20 DE overlap | caveat |
|---|---|---:|---:|---:|---|
| GEARS published | Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6 | 0.216 +/- 0.053 | 0.556 +/- 0.030 | not reported | not reported in GEARS Supplementary Table 6 |
| CPA Original published | Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6 | 0.354 +/- 0.049 | 0.440 +/- 0.036 | not reported | not reported in GEARS Supplementary Table 6 |
| CPA + KG published | Roohani, Huang, Leskovec, Nature Biotechnology 2024, Supplementary Table 6 | 0.333 +/- 0.046 | 0.504 +/- 0.029 | not reported | not reported in GEARS Supplementary Table 6 |

## Caveats

- GEARS Supplementary Table 6 reports MSE and Pearson DE, not subset-specific exact-train-combo/unseen-single metrics.
- GEARS Supplementary Table 6 does not report top-20 DE overlap, so Tier 3 top-20 comparison remains unresolved until raw GEARS predictions or a paper-faithful rerun are available.
- Closed-form PLS is RNA-only here: perturbation multi-hot features are mapped to pseudobulk delta by PLS regression because no image modality exists in the Norman h5ad.
