# Final Report

## 1. Executive Summary

PARTIAL: some pieces show micro retrieval signal, but collapse dominates and no candidate is safe for real data.

## 2. Compute Summary

- Total experiments: 100
- Total GPU runs: 0
- Approximate wallclock: 34.8 minutes
- Max GPU memory used: 0 GB
- Skipped GPU use: yes, by conservative CPU-only policy

## 3. Active Model Of Record

`PerturbJEPABridge`; no Tier 3 promotion occurred.

## 4. Synthetic Generator Validity

Generator covers count-like RNA, dropout, library size, perturbation programs, dose response, batch effects, held-out splits, and paired cross-modal latent structure.

## 5. Baseline Comparison

Best scratch candidate by raw micro recall@1 tied the weak baseline-level signal and remained invalid when collapsed. The most useful non-collapsed signal came from Family B mean-pooling variants:

- Experiment 029 `mean_p1`: recall@1 0.0625, collapse false, discarded for batch leakage.
- Experiment 030 `mean_p2`: recall@1 0.0625, collapse false, discarded for batch leakage.
- Experiment 031 `mean_p4`: recall@1 0.0625, collapse false, discarded for batch leakage.

These runs show that simpler pooling can avoid the hard collapse failure, but they do not prove biological alignment because retrieval is weak and batch diagnostics fail.

## 6. Evidence For JEPA

Some runs produce above-random micro retrieval, but this is not sufficient without non-collapsed representations.

## 7. Evidence Against JEPA

Repeated collapse, failure to beat simple baselines cleanly, and no Tier 2-worthy candidate. Out of 100 experiments, 97 were hard collapse failures and 3 were non-collapsed but failed batch-leakage/no-signal gates.

## 8. Family Findings

- Family A: 13 experiments
- Family B: 6 experiments
- Family C: 13 experiments
- Family D: 11 experiments
- Family E: 14 experiments
- Family F: 12 experiments
- Family G: 15 experiments
- Family H: 12 experiments
- Family I: 4 experiments

Decision labels:
- TIER1_DISCARD_BATCH_LEAKAGE: 3
- TIER1_DISCARD_MODE_COLLAPSE: 97

No candidate reached Tier 2. Families G, H, and I were added in the scratch run:

- G: redundancy-reduction geometry based on covariance and cross-correlation penalties.
- H: optimization/regularization probes.
- I: minimal objective ablations.

## 9. Recommendation

DO_NOT_USE_REAL_DATA_YET

## 10. Next Phase

Redesign objective diagnostics and anti-collapse losses before any real-data pilot.

## Verification

- Clean scratch result ledger contains exactly 100 experiments, numbered 001 through 100.
- Step 0 metrics exist for 6 synthetic datasets.
- GPU memory recorded by the run: 0 GB.
- Full test suite after the run: `124 passed, 18 warnings`.
