# Family M Transport Baselines

- Dataset: `synth_dose_extrapolation_lite`
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
- pseudobulk correlation: `0.9698`
- top50 overlap: `0.1386`

## Candidate Results

### seed2_no_batch_matched_perturbed_mean
- method: `matched_perturbed_mean`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0892`
- direction accuracy: `0.6061`
- logFC correlation: `0.9193`
- pseudobulk correlation: `0.9946`
- top50 overlap: `0.2433`
- mean delta/target ratio: `0.7174`
- beats residualized matching: `True`
- counterfactual gate pass: `True`

### seed2_no_batch_residualized_matching
- method: `residualized_matching`
- exact no-batch key coverage: `0.0000`
- program recovery: `-0.0085`
- direction accuracy: `0.5467`
- logFC correlation: `0.5549`
- pseudobulk correlation: `0.9781`
- top50 overlap: `0.1798`
- mean delta/target ratio: `0.1512`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k1
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0288`
- direction accuracy: `0.5117`
- logFC correlation: `0.2482`
- pseudobulk correlation: `0.7702`
- top50 overlap: `0.1616`
- mean delta/target ratio: `1.6575`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k3
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0440`
- direction accuracy: `0.5085`
- logFC correlation: `0.1513`
- pseudobulk correlation: `0.7969`
- top50 overlap: `0.1633`
- mean delta/target ratio: `1.5500`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k5
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0623`
- direction accuracy: `0.5111`
- logFC correlation: `0.1824`
- pseudobulk correlation: `0.7937`
- top50 overlap: `0.1606`
- mean delta/target ratio: `1.5220`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k8
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0829`
- direction accuracy: `0.5109`
- logFC correlation: `0.1716`
- pseudobulk correlation: `0.8112`
- top50 overlap: `0.1601`
- mean delta/target ratio: `1.5017`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_sinkhorn_residual_transport_program_eps0p5
- method: `sinkhorn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0619`
- direction accuracy: `0.4853`
- logFC correlation: `-0.1367`
- pseudobulk correlation: `0.8321`
- top50 overlap: `0.1752`
- mean delta/target ratio: `0.5713`
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
