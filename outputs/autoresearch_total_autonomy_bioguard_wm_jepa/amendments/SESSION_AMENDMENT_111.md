# Session Amendment 111: Fresh-Seed Delta-Calibrated JEPA Validation

## Trigger
`F081_DELTA_CALIBRATED_JEPA_TIER1_PASS_NONPROMOTING`

## Evidence
F081 passed a Tier 1 diagnostic by showing that F080's JEPA predicted deltas contain train-only linearly recoverable heldout delta direction. Because Tier 1 cannot promote, the next autonomous step is fresh-seed validation under the same no-leakage rules.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F082 is Tier 2-style validation and cannot promote.

## New Branch
`F082_DELTA_CALIBRATED_TIER2_VALIDATION`

## Implementation Tasks
- Rerun the F081 real JEPA plus train-only delta calibrator on fresh seeds 37-42.
- Keep the same three split policies.
- Report mean/std/min transition, delta cosine, recall, rank, magnitude, cross-modal retrieval, and action-negative gap.
- Treat any seed/policy collapse as a pivot event.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F082 is stable, write a Tier 3 design amendment with no-regression validators and a locked candidate code path. If it is unstable, localize the failing policy/seed before changing architecture.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F082 fresh-seed delta-calibrated JEPA validation
