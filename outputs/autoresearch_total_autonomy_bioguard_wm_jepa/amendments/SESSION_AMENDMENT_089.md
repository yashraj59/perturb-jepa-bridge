# Session Amendment 089: Floor-Direction-Projected Row-Wise Residual

## Trigger
`F059_TIER2_WEAKNESS_IS_TRANSITION_GENERALIZATION_NOISE`

## Evidence
F059 localized F058's Tier 2 miss to transition-source cosine generalization noise. Delta cosine was positive on every seed, recall was preserved, and no retrieval rows broke, but train transition gains were anti-correlated with held-out transition gaps. A train transition gate alone is therefore not trustworthy.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F060 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F060_FLOOR_DIRECTION_PROJECTION`

## Implementation Tasks
- Use fresh seeds 28, 29, and 30.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Project the residual onto the positive ridge floor-delta direction before train-only row-wise abstention.
- Score held-out rows only after train-only rule selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F060 floor-direction-projected row-wise residual
