# Session Amendment 093: Train-Heldout Selector Mismatch Audit

## Trigger
`F063_TRAIN_CONE_SELECTOR_HELDOUT_BELOW_FLOOR`

## Evidence
F063 replaced held-out oracle cone selection with train-only cone selection. It selected nonzero residuals on all fresh seeds, but the selected residual fell below the protected held-out floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F064 is read-only diagnostic work and cannot promote.

## New Diagnostic Branch
`F064_TRAIN_HELDOUT_SELECTOR_AUDIT`

## Implementation Tasks
- Read F063 seed, cone-grid, and query-row artifacts.
- Compare train selected transition/delta/recall gaps against held-out selected gaps.
- Count train-positive but heldout-negative seeds.
- Separate continuous transition/delta failure from retrieval-row breakage.
- Document whether another residual rerun is justified before a representation/data pivot.
- Do not train a new model and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F064 train-heldout selector mismatch audit
