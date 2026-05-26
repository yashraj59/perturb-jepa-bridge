# Pseudobulk Whitening Probe Report

Wallclock minutes: `0.060`

## Summary

- Probes evaluated: `5`
- Passing test probes: `5`

## Test Results

- `oracle_true_z_bio`: pass=True, collapse=False, recall@1=1.0000, rna_min_std=0.1207, image_min_std=0.1207, rna_R2=0.9999, image_R2=0.9999, batch_bal_acc=0.3750
- `ridge_supervised_z_bio`: pass=True, collapse=False, recall@1=0.4375, rna_min_std=0.1402, image_min_std=0.1103, rna_R2=0.4156, image_R2=0.9880, batch_bal_acc=0.2188
- `ridge_rna_to_image_pca`: pass=True, collapse=False, recall@1=0.3750, rna_min_std=0.8861, image_min_std=0.9936, rna_R2=0.6008, image_R2=0.8429, batch_bal_acc=0.3125
- `regularized_cca`: pass=True, collapse=False, recall@1=0.2500, rna_min_std=0.9121, image_min_std=0.9912, rna_R2=0.6267, image_R2=0.8431, batch_bal_acc=0.3125
- `pls_cross_covariance`: pass=True, collapse=False, recall@1=0.2188, rna_min_std=24.3300, image_min_std=41.8944, rna_R2=0.5940, image_R2=0.8445, batch_bal_acc=0.3438

## Interpretation

A passing probe means the synthetic data contains enough condition-level RNA and image information to build a safe shared space without changing JEPA. That shared-space geometry should then be used as the target for the next minimal model repair.

## Artifacts

- `PROBE_RESULTS.tsv`
- `EMBEDDING_VALUES.tsv`
