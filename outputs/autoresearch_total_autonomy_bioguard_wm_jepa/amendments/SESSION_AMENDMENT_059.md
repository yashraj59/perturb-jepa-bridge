# Session Amendment 059: Delta-Direction Cross-Modal/Action JEPA Repair

## Trigger
`F029_REPRESENTATION_FLOOR_GOOD_DIRECT_JEPA_UNDER_FLOOR`

## Evidence
F029 produced positive direct transition improvement and healthy RNA/image retrieval, but its direct JEPA transition delta cosine remained below the active gate. Training longer is not the right first response; the predictor needs explicit direction supervision in latent delta space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA`

## Implementation Tasks
- Preserve the F029 real JEPA identity: online encoders, EMA target encoders, stop-gradient latent targets, query-conditioned predictors, cross-modal JEPA losses, and program-action transition loss.
- Add a stop-gradient teacher-delta direction loss.
- Add a source-improvement hinge so the predicted endpoint must improve over the source state.
- Keep action input as non-exact program descriptors only.
- Score direct JEPA transitions, representation ridge floor, cross-modal retrieval, action-negative gap, rank, identity, and leakage diagnostics.

## Decision Use
If F030 passes all direct transition gates and cross-modal retrieval remains healthy, design a Tier 2 amendment. If delta cosine improves but remains below gate, test an operator-initialized transition predictor rather than training longer. If it regresses, close this repair path.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F030 delta-direction transition repair
