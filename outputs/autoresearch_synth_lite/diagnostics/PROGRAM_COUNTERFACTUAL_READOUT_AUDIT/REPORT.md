# Program Counterfactual Readout Audit

- Dataset: `synth_micro`
- Seeds: `0,1,2`
- Real data used: `false`
- External biological resources used: `false`

## Summary

### source_as_target
- program_level_effect_recovery: `0.0000 +/- 0.0000`
- direction_accuracy: `0.0000 +/- 0.0000`
- logfc_correlation: `0.0000 +/- 0.0000`
- pseudobulk_correlation: `0.6320 +/- 0.0871`
- top50_de_overlap: `0.3972 +/- 0.0062`

### condition_train_mean_delta
- program_level_effect_recovery: `0.1933 +/- 0.1124`
- direction_accuracy: `0.5412 +/- 0.0235`
- logfc_correlation: `0.1511 +/- 0.0423`
- pseudobulk_correlation: `0.6334 +/- 0.1033`
- top50_de_overlap: `0.4064 +/- 0.0140`

### metadata_ridge
- program_level_effect_recovery: `0.1955 +/- 0.0397`
- direction_accuracy: `0.5391 +/- 0.0163`
- logfc_correlation: `0.1511 +/- 0.0423`
- pseudobulk_correlation: `0.6334 +/- 0.1033`
- top50_de_overlap: `0.4100 +/- 0.0072`

### metadata_no_batch_ridge
- program_level_effect_recovery: `0.2025 +/- 0.0300`
- direction_accuracy: `0.5434 +/- 0.0193`
- logfc_correlation: `0.1511 +/- 0.0423`
- pseudobulk_correlation: `0.6334 +/- 0.1033`
- top50_de_overlap: `0.4142 +/- 0.0085`

### source_program_metadata_ridge
- program_level_effect_recovery: `0.4137 +/- 0.1472`
- direction_accuracy: `0.5897 +/- 0.0526`
- logfc_correlation: `0.2186 +/- 0.0385`
- pseudobulk_correlation: `0.6416 +/- 0.1157`
- top50_de_overlap: `0.4158 +/- 0.0083`

### source_program_no_batch_metadata_ridge
- program_level_effect_recovery: `0.4157 +/- 0.1369`
- direction_accuracy: `0.5818 +/- 0.0213`
- logfc_correlation: `0.2533 +/- 0.0213`
- pseudobulk_correlation: `0.6244 +/- 0.0819`
- top50_de_overlap: `0.4092 +/- 0.0095`

### oracle_observed_delta
- program_level_effect_recovery: `1.0000 +/- 0.0000`
- direction_accuracy: `0.6701 +/- 0.0261`
- logfc_correlation: `0.3209 +/- 0.0572`
- pseudobulk_correlation: `0.6692 +/- 0.0962`
- top50_de_overlap: `0.4228 +/- 0.0070`

## Interpretation

The condition lookup is a technical-duplicate ceiling for this split: train and test contain cells from the same synthetic conditions.
Metadata ridge and source-program metadata ridge test whether a simple closed-form readout can recover program deltas without neural counterfactual decoder training.

## Artifacts

- `PROGRAM_READOUT_RESULTS.tsv`
