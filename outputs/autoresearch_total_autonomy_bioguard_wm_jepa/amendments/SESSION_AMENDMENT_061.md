# Session Amendment 061: Floor-Preserving JEPA Residual Calibration

## Trigger
`F031_PROTECTED_TIER2_PASS_LOCAL_RIDGE_FLOOR_NOT_PRESERVED`

## Evidence
F031 passed protected Tier 2 gates and maintained cross-modal retrieval, but the direct JEPA transition head fell below the same-representation train-only ridge floor on transition, delta cosine, and recall. The next experiment must preserve the local floor exactly before allowing any JEPA residual.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F032_FLOOR_PRESERVING_JEPA_RESIDUAL_CALIBRATION`

## Implementation Tasks
- Train the same low-compute F030 real JEPA on seeds 0-4.
- Fit the same-representation train-only ridge floor on frozen learned latents.
- Build residuals as `direct_jepa_delta - local_ridge_floor_delta`.
- Select residual scale using train rows only, requiring transition, delta cosine, and recall to preserve the train floor.
- Score held-out rows only after selection.
- Keep action input as non-exact program descriptors only.

## Decision Use
If nonzero residual scale preserves the held-out local floor, design a Tier 3/no-regression amendment. If scale zero is selected, pivot to an operator-initialized transition predictor instead of treating direct residuals as safe.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F032 train-only floor-preserving JEPA residual calibration
