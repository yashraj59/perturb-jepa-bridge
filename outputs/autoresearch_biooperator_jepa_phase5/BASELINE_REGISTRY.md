# Phase 5 Baseline Registry

## Active Model Of Record

- Protected model: rank-3 train-split-only PLS raw-linear readout
- Role: protected baseline and audit reference only
- Status: no Phase 1-4 JEPA candidate has been promoted
- PLS restriction: never used as the BioOperator-JEPA main representation path

## Phase 4 Frozen-Latent Reference

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Latent cache model: BTJ001 BioTech-JEPA checkpoint
- Forbidden shortcuts checked: no `condition_key`, no biological-key one-hot, no test target means, no pooled train+test targets

| Metric | Value |
| --- | ---: |
| train delta effective rank | 13.5627 |
| train delta mean norm | 0.4310 |
| train delta std mean | 0.0846 |
| train source-to-target cosine mean | 0.8977 |
| eval delta effective rank | 11.7819 |
| eval delta mean norm | 0.4252 |
| eval delta std mean | 0.0832 |
| eval source-to-target cosine mean | 0.9031 |
| eval action_ridge_delta transition improvement | +0.0057 |
| eval action_ridge_delta delta cosine | 0.3980 |
| eval action_ridge_delta delta rank | 10.2835 |
| eval action_low_rank_ridge transition improvement | +0.0046 |
| train action_ridge_delta transition improvement | +0.0769 |
| BFJ001 transition improvement | -0.0104 |
| BFJ001 delta cosine | -0.1054 |

Raw sources:

- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md`
- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_BASELINE_RESULTS.tsv`
- `outputs/autoresearch_bioflow_jepa_phase4/final_report.md`
