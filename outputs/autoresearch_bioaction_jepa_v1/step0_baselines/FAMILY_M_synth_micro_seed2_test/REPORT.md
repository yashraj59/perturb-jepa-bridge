# Family M Transport Baselines

- Dataset: `synth_micro`
- Seed: `2`
- Device: `cpu`
- Biological matching key: `perturbation_id, cell_line_id, dose, time`
- Batch ID excluded from matching/features: `true`
- Real data used: `false`
- Marker/pathway/pretrained biological assets used: `false`

## Source-As-Target Reference

- program recovery: `0.0000`
- direction accuracy: `0.0000`
- logFC correlation: `0.0000`
- pseudobulk correlation: `0.7549`
- top50 overlap: `0.4058`

## Candidate Results

### seed2_no_batch_matched_perturbed_mean
- method: `matched_perturbed_mean`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.7520`
- direction accuracy: `0.6899`
- logFC correlation: `0.7502`
- pseudobulk correlation: `0.8725`
- top50 overlap: `0.5683`
- mean delta/target ratio: `0.7400`
- beats residualized matching: `True`
- counterfactual gate pass: `True`

### seed2_no_batch_residualized_matching
- method: `residualized_matching`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.3502`
- direction accuracy: `0.5312`
- logFC correlation: `0.1268`
- pseudobulk correlation: `0.7491`
- top50 overlap: `0.4150`
- mean delta/target ratio: `0.3421`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k1
- method: `knn_residual_transport`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.5146`
- direction accuracy: `0.5161`
- logFC correlation: `0.0513`
- pseudobulk correlation: `0.6216`
- top50 overlap: `0.4075`
- mean delta/target ratio: `0.7829`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k3
- method: `knn_residual_transport`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.5293`
- direction accuracy: `0.5488`
- logFC correlation: `0.0623`
- pseudobulk correlation: `0.6954`
- top50 overlap: `0.4308`
- mean delta/target ratio: `0.5087`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k5
- method: `knn_residual_transport`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.5145`
- direction accuracy: `0.5450`
- logFC correlation: `-0.0038`
- pseudobulk correlation: `0.7136`
- top50 overlap: `0.4217`
- mean delta/target ratio: `0.4262`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k8
- method: `knn_residual_transport`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.4593`
- direction accuracy: `0.5433`
- logFC correlation: `0.0795`
- pseudobulk correlation: `0.7361`
- top50 overlap: `0.4167`
- mean delta/target ratio: `0.3825`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_sinkhorn_residual_transport_program_eps0p5
- method: `sinkhorn_residual_transport`
- exact no-batch key coverage: `1.0000`
- program recovery: `0.4211`
- direction accuracy: `0.5343`
- logFC correlation: `0.0657`
- pseudobulk correlation: `0.7290`
- top50 overlap: `0.4217`
- mean delta/target ratio: `0.3827`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

## Interpretation

Family M tests the CellOT/CINEMA-OT/scDRP/Conditional-Monge intuition on the synthetic seed-2 task without using batch features.
Systema-style discipline is enforced by treating residualized matching as the baseline that transport must beat before neural transport is justified.

## Artifacts

- `FAMILY_M_RESULTS.tsv`
- `FAMILY_M_RESULTS.json`
- `generation_config.json`
- `prefit_pls_readout.json`
