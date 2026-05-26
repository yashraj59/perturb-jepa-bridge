# Session Amendment 077: Non-Exact Program Action Contract

## Trigger
`F047_HELDOUT_EXACT_ACTION_SUPPORT_GAP_CONFIRMED`

## Evidence
F047 showed held-out perturbations have zero exact action matches and only partial active action-dimension support, while teacher-delta and residual latent support are not absent. This suggests the residual head may be overfitting exact train perturbation descriptors rather than using shared biological action structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F048_NON_EXACT_ACTION_CONTRACT`

## Implementation Tasks
- Remove exact perturbation one-hot action columns from the active operator input.
- Keep only deterministic non-exact program-level action descriptors.
- Reproduce the local program-action ridge floor.
- Train the same exact floor-initialized residual head under the tiny-cap train-only selection rule.
- Require exact held-out local floor preservation before any future full JEPA wrapping.

## Decision Use
If non-exact actions repair the train-heldout mismatch, design a proper BioGuard-WM action-token contract around shared biological programs. If they still fall below floor or select zero, pivot away from residual operator heads toward representation/data contract redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F048 non-exact program action contract
