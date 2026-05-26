# Session Amendment 071: Fresh-Seed Tiny-Cap Validation

## Trigger
`F041_SIMPLE_TRAIN_PROXY_ABLATION_RECOVERS_SAFE_NONZERO_DIAGNOSTIC`

## Evidence
F041 showed the full F039 proxy rejected every oracle-safe nonzero row because near-tie erosion failed on all oracle-safe nonzero rows. A small fixed residual cap of `0.05` preserved held-out gates in the diagnostic grid, but that rule was selected with held-out evidence.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Validation Branch
`F042_FRESH_SMALL_CAP_VALIDATION`

## Implementation Tasks
- Use fresh synthetic seeds 5-9, not the F038/F041 diagnostic seeds.
- Train the same low-compute PCA-bootstrap JEPA and floor-initialized transition head.
- Select residual scale only from `(0.0, 0.025, 0.05)` using train continuous transition/delta/recall preservation.
- Score held-out rows only after train-only scale selection.
- Do not promote this result; a pass can only motivate a stricter future Tier 2/Tier 3 design.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F042 fresh-seed tiny-cap validation
