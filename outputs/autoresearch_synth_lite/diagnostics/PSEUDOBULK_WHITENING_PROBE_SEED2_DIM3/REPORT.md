# Pseudobulk Whitening Probe Report

Wallclock minutes: `0.066`

## Summary

- Probes evaluated: `5`
- Passing test probes: `5`

## Test Results

- `oracle_true_z_bio`: pass=True, collapse=False, recall@1=1.0000, rna_min_std=0.1207, image_min_std=0.1207, rna_R2=0.9999, image_R2=0.9999, batch_bal_acc=0.3750
- `ridge_supervised_z_bio`: pass=True, collapse=False, recall@1=0.4375, rna_min_std=0.1402, image_min_std=0.1103, rna_R2=0.4156, image_R2=0.9880, batch_bal_acc=0.2188
- `ridge_rna_to_image_pca`: pass=True, collapse=False, recall@1=0.3125, rna_min_std=0.8861, image_min_std=0.9863, rna_R2=0.4792, image_R2=0.9150, batch_bal_acc=0.2188
- `pls_cross_covariance`: pass=True, collapse=False, recall@1=0.2812, rna_min_std=12.1573, image_min_std=19.2276, rna_R2=0.5902, image_R2=0.9146, batch_bal_acc=0.3750
- `regularized_cca`: pass=True, collapse=False, recall@1=0.2500, rna_min_std=0.8879, image_min_std=0.9350, rna_R2=0.6080, image_R2=0.9049, batch_bal_acc=0.3125

## Interpretation

A passing probe means the synthetic data contains enough condition-level RNA and image information to build a safe shared space without changing JEPA. That shared-space geometry should then be used as the target for the next minimal model repair.

## Artifacts

- `PROBE_RESULTS.tsv`
- `EMBEDDING_VALUES.tsv`
