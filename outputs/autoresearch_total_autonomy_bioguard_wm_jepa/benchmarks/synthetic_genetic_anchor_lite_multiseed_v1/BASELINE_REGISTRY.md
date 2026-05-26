# Synthetic Genetic Anchor Lite Multiseed v1 Baseline Registry

## Status
`REGISTRY_BUILT_PROPOSED_GATES_NOT_ACTIVE`

This registry creates a new named benchmark reference. It does not modify the old `test_heldout_perturbation` benchmark or its protected seed0 floor.

## Seeds
Seed caches used: `[0, 1, 2, 3, 4]`. All baselines are fit only on each seed's train split and evaluated on that seed's held-out perturbation split.

## Full Action-Ridge Registry

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| transition improvement | 0.011281 | 0.008060 | 0.000424 | 0.021756 |
| delta cosine | 0.451515 | 0.057127 | 0.397963 | 0.540428 |
| condition recall@1 | 0.333333 | 0.077690 | 0.259259 | 0.481481 |
| effective rank | 10.245809 | 0.434351 | 9.632915 | 10.888690 |
| magnitude ratio | 0.755643 | 0.093544 | 0.591863 | 0.877834 |

## Proposed Future Gate Principles

These are proposals for a future amendment, not active gates:

- Primary continuous geometry gate: transition improvement and delta cosine should be judged against multi-seed mean/std, not seed0 alone.
- Retrieval gate should be secondary/no-regression until condition recall is shown stable.
- Candidate promotion should require rank and magnitude preservation.
- New architecture runs must declare whether they target the old benchmark or this new named benchmark.

## Files

- `baseline_metrics_by_seed.tsv`
- `baseline_registry_summary.tsv`
- `metrics_eval.json`
