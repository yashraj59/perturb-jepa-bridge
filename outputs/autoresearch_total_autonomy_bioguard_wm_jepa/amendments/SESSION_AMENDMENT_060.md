# Session Amendment 060: Delta-Direction JEPA Tier 2 Validation

## Trigger
`F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA_TIER1_PASS`

## Evidence
F030 passed the active Tier 1 gates with a real cross-modal/action JEPA, but Tier 1 cannot promote and does not prove stability. The next required step is multi-seed validation with explicit protected-floor and same-representation ridge-floor checks.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Validation Branch
`F031_DELTA_DIRECTION_TIER2_VALIDATION`

## Implementation Tasks
- Rerun the F030 architecture across synthetic seeds 0-4.
- Keep the same low-compute setting and do not train longer by default.
- Report mean/std, per-seed rows, and whether std is smaller than the claimed protected-floor effect.
- Check protected historical floor gates and same-representation train-only ridge floor preservation separately.
- Preserve JEPA identity: online encoders, EMA targets, stop-gradient latent targets, query predictors, cross-modal prediction, action-conditioned transition.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F031 passes protected Tier 2 but fails the same-representation ridge floor, pivot to a floor-preserving transition head before Tier 3. If it preserves both, design Tier 3/no-regression validation. No Tier 2 result can promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F031 Tier 2 delta-direction validation
