# Sparse Perturbation Residual Head Report

- Dataset: `synth_micro`
- Seed: `2`
- PLS rank: `3`
- Steps: `150`
- Dictionary rank: `8`
- Top DE genes: `50`
- Top active programs: `2`
- Device used: `cuda`
- Max GPU memory GB: `0.0508`
- Protected geometry preserved: `True`
- Counterfactual gate pass: `False`
- Frozen readout max abs drift: `0.00000000`

## Counterfactual Metrics

- direction accuracy before: `0.0000`
- direction accuracy after: `0.5622`
- logFC correlation before: `0.0000`
- logFC correlation after: `0.1088`
- pseudobulk correlation before: `0.7549`
- pseudobulk correlation after: `0.7318`
- program recovery before: `0.0000`
- program recovery after: `0.2750`
- top50 overlap before: `0.4058`
- top50 overlap after: `0.3917`

## Matching-Mean Baseline

- rows: `24`
- exact no-batch key coverage: `1.0000`
- direction accuracy: `0.5312`
- logFC correlation: `0.1268`
- top50 overlap: `0.4150`
- program recovery: `0.3502`

## Sparse Head Diagnostics

- mean delta/target ratio: `0.4116`
- mean program contribution ratio: `0.5395`
- mean low-rank contribution ratio: `0.6403`
- mean effect mask fraction: `0.5200`
- mean decorrelation loss: `0.356482`
- mean outside sparsity loss: `0.385322`

## Protected Deltas

- `model_rna_to_image_recall@1`: `0.00000000`
- `model_rna_to_image_recall@5`: `0.00000000`
- `model_bio_latent_r2_rna_shared`: `0.00000000`
- `model_bio_latent_r2_image_shared`: `0.00000000`
- `model_rna_shared_min_std`: `0.00000000`
- `model_image_shared_min_std`: `0.00000000`
- `model_batch_probe_balanced_accuracy`: `0.00000000`

## Counterfactual Deltas

- `model_rna_counterfactual_direction_accuracy`: `0.56215659`
- `model_rna_counterfactual_logfc_correlation`: `0.10883118`
- `model_rna_counterfactual_pseudobulk_correlation`: `-0.02309500`
- `model_program_level_effect_recovery`: `0.27501528`
- `model_rna_counterfactual_top50_de_overlap`: `-0.01416667`

## Artifacts

- `BEFORE_METRICS.json`
- `AFTER_METRICS.json`
- `TRAIN_HISTORY.json`
- `MATCHING_MEAN_BASELINE.json`
- `SPARSE_RESIDUAL_CONFIG.json`
- `prefit_pls_readout.json`
- `sparse_perturbation_residual_head.pt`
