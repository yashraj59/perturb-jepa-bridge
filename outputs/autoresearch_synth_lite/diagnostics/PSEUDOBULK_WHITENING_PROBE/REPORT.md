# Pseudobulk Whitening Probe Report

Wallclock minutes: `0.040`

## Summary

- Probes evaluated: `5`
- Passing test probes: `4`

## Test Results

- `oracle_true_z_bio`: pass=True, collapse=False, recall@1=1.0000, rna_min_std=0.2293, image_min_std=0.2293, rna_R2=0.9999, image_R2=0.9999, batch_bal_acc=0.4062
- `pls_cross_covariance`: pass=True, collapse=False, recall@1=0.2188, rna_min_std=1.4935, image_min_std=0.4946, rna_R2=0.3093, image_R2=0.9827, batch_bal_acc=0.5938
- `ridge_supervised_z_bio`: pass=True, collapse=False, recall@1=0.1875, rna_min_std=0.2477, image_min_std=0.2315, rna_R2=0.2547, image_R2=0.9867, batch_bal_acc=0.4688
- `ridge_rna_to_image_pca`: pass=True, collapse=False, recall@1=0.1250, rna_min_std=0.9927, image_min_std=0.9678, rna_R2=0.2510, image_R2=0.9845, batch_bal_acc=0.5625
- `regularized_cca`: pass=False, collapse=False, recall@1=0.0938, rna_min_std=0.8083, image_min_std=0.9435, rna_R2=0.2573, image_R2=0.9844, batch_bal_acc=0.5625

## Interpretation

A passing probe means the synthetic data contains enough condition-level RNA and image information to build a safe shared space without changing JEPA. That shared-space geometry should then be used as the target for the next minimal model repair.

## Artifacts

- `PROBE_RESULTS.tsv`
- `EMBEDDING_VALUES.tsv`
