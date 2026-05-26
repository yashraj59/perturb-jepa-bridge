# Session Amendment 113: Rank-Floor Comparability Audit

## Trigger
`F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS`

## Evidence
F083 showed F082 passes transition, delta cosine, recall, magnitude, and cross-modal gates but fails the locked delta-rank no-regression gate. Before adding rank capacity, the loop must determine whether the locked rank floor is comparable to the current image-teacher latent target space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F084_RANK_FLOOR_COMPARABILITY`

## Implementation Tasks
- Read F082 per-seed metrics.
- Compare calibrated delta rank, raw direct delta rank, image-teacher delta rank, and target-image effective rank against the locked rank floor.
- Decide whether the rank failure is repairable in the current target space or requires a representation-specific floor registry before Tier 3.
- Do not train and do not promote.

## Decision Use
If the teacher/target rank ceiling is below the locked floor, reproduce a train-only floor inside the current image-teacher latent registry before Tier 3. If the teacher rank supports the floor but the calibrator loses rank, implement a rank-preserving calibrator.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F084 rank-floor comparability audit
