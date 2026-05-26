# Session Amendment 074: Exact Floor-Initialized Operator Wrapper

## Trigger
`F044_ACTION_ADALN_FAILURE_LOCALIZED_TO_UNSTABLE_NO_ACTION_SIGNAL`

## Evidence
F044 localized the Action-AdaLN tiny-cap failure to unstable train evidence and zero action-negative separation. The earlier F042 result showed a tiny-cap residual can be safe in the exact floor-initialized operator path, so the next step is to test that path on the active multiseed latent benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Diagnostic
`F045_ACTIVE_OPERATOR_WRAPPER`

## Implementation Tasks
- Use active cached multiseed latent bundles, seeds 0-4.
- Fit the train-only local ridge floor.
- Initialize the operator exactly at the ridge floor.
- Train only a zero-initialized residual head.
- Select scale from `(0.0, 0.025, 0.05)` using train-only continuous metrics.
- Require exact held-out floor preservation and zero broken retrieval rows.
- Do not promote; a pass only motivates stricter validation or full JEPA wrapping.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F045 exact floor-initialized operator wrapper
