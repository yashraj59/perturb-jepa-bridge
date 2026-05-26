# Family M Transport Baselines

- Dataset: `synth_heldout_perturbation_lite`
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
- pseudobulk correlation: `0.9570`
- top50 overlap: `0.1412`

## Candidate Results

### seed2_no_batch_matched_perturbed_mean
- method: `matched_perturbed_mean`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.1168`
- direction accuracy: `0.6417`
- logFC correlation: `0.3425`
- pseudobulk correlation: `0.9613`
- top50 overlap: `0.2711`
- mean delta/target ratio: `0.8264`
- beats residualized matching: `False`
- counterfactual gate pass: `True`

### seed2_no_batch_residualized_matching
- method: `residualized_matching`
- exact no-batch key coverage: `0.0000`
- program recovery: `-0.0038`
- direction accuracy: `0.5233`
- logFC correlation: `0.3425`
- pseudobulk correlation: `0.9613`
- top50 overlap: `0.1398`
- mean delta/target ratio: `0.1672`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k1
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0013`
- direction accuracy: `0.5026`
- logFC correlation: `0.1320`
- pseudobulk correlation: `0.8323`
- top50 overlap: `0.1681`
- mean delta/target ratio: `1.7879`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k3
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0823`
- direction accuracy: `0.5050`
- logFC correlation: `0.1155`
- pseudobulk correlation: `0.8490`
- top50 overlap: `0.1721`
- mean delta/target ratio: `1.6757`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k5
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.0821`
- direction accuracy: `0.5020`
- logFC correlation: `0.1010`
- pseudobulk correlation: `0.8464`
- top50 overlap: `0.1719`
- mean delta/target ratio: `1.6439`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_knn_residual_transport_program_k8
- method: `knn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.1536`
- direction accuracy: `0.5029`
- logFC correlation: `0.1099`
- pseudobulk correlation: `0.8520`
- top50 overlap: `0.1714`
- mean delta/target ratio: `1.6291`
- beats residualized matching: `False`
- counterfactual gate pass: `False`

### seed2_sinkhorn_residual_transport_program_eps0p5
- method: `sinkhorn_residual_transport`
- exact no-batch key coverage: `0.0000`
- program recovery: `0.1471`
- direction accuracy: `0.5033`
- logFC correlation: `0.0420`
- pseudobulk correlation: `0.8327`
- top50 overlap: `0.1398`
- mean delta/target ratio: `0.5621`
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
