# Session Amendment 064: Inner-Validation Operator Gate

## Trigger
`F034_OPERATOR_INITIALIZED_RESIDUAL_DISCARDED_HELDOUT_BELOW_FLOOR`

## Evidence
F034 initialized the transition head exactly at the train-only local ridge floor and improved mean held-out transition, delta cosine, and recall, but it still had localized no-regression failures. Selection was based on train-fit metrics, so the next test must use train-only inner validation to choose residual scale.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F035_INNER_VAL_OPERATOR_GATE`

## Implementation Tasks
- Keep the F034 floor-initialized transition head.
- Split train condition rows into inner-fit and inner-validation sets.
- Fit the floor and residual head on inner-fit only.
- Select residual scale on inner-validation only.
- Refit the operator-initialized head on full train and deploy the selected scale on held-out rows for scoring only.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F035 preserves held-out local floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero or still violates local floor, pivot to a different operator/representation family.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F035 inner-validation operator residual gate
