# Reopening Decision

Decision label: `PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN`

Reopen architecture search: `False`

## Criteria

- `measurable_biological_signal_not_fully_explained_by_batch`: `True`
- `enough_cross_batch_biological_anchors_or_valid_substitute`: `False`
- `identified_source_of_batch_leakage`: `True`
- `targeted_mechanism_beyond_increase_invariance_weight`: `True`
- `updated_gates_with_exact_baselines`: `True`

## Key Values

- Minimum held-out cross-batch train-anchor fraction: `0.0000`
- Valid substitute for cross-batch teacher: `False`
- Max split-half RNA->image same-bio recall@1: `0.5625`
- Max raw/protected batch-probe excess over majority: `0.6667`
- Max Phase1/zero-step representation batch-probe excess over majority: `0.6667`

## Action

write final_report.md and stop
