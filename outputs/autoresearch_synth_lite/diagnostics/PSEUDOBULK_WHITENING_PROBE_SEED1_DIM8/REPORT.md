# Pseudobulk Whitening Probe Report

Wallclock minutes: `0.053`

## Summary

- Probes evaluated: `5`
- Passing test probes: `2`

## Test Results

- `oracle_true_z_bio`: pass=True, collapse=False, recall@1=1.0000, rna_min_std=0.1836, image_min_std=0.1836, rna_R2=0.9999, image_R2=0.9999, batch_bal_acc=0.3438
- `regularized_cca`: pass=True, collapse=False, recall@1=0.1562, rna_min_std=0.9617, image_min_std=0.9687, rna_R2=0.7938, image_R2=0.8933, batch_bal_acc=0.5312
- `pls_cross_covariance`: pass=False, collapse=False, recall@1=0.3750, rna_min_std=2.5922, image_min_std=2.1003, rna_R2=0.7280, image_R2=0.9810, batch_bal_acc=0.6562
- `ridge_rna_to_image_pca`: pass=False, collapse=False, recall@1=0.2500, rna_min_std=1.0028, image_min_std=0.9743, rna_R2=0.6263, image_R2=0.9813, batch_bal_acc=0.7188
- `ridge_supervised_z_bio`: pass=False, collapse=False, recall@1=0.2500, rna_min_std=0.2158, image_min_std=0.1658, rna_R2=0.6187, image_R2=0.9916, batch_bal_acc=0.6562

## Interpretation

A passing probe means the synthetic data contains enough condition-level RNA and image information to build a safe shared space without changing JEPA. That shared-space geometry should then be used as the target for the next minimal model repair.

## Artifacts

- `PROBE_RESULTS.tsv`
- `EMBEDDING_VALUES.tsv`
