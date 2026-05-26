# Session Amendment 092: Train-Only Residual-Cone Selector

## Trigger
`F062_ORACLE_CONE_CAPACITY_EXISTS_ALL_SEEDS`

## Evidence
F062 found held-out oracle residual-cone capacity on all fresh seeds, but it is not deployable because held-out labels selected the cone mix. The next step is to replace oracle cone selection with a train-only cone selector and score fresh held-out seeds once.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F063 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F063_TRAIN_CONE_SELECTOR`

## Implementation Tasks
- Use fresh seeds 34, 35, and 36.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Evaluate cone mixes using train-only row-wise abstention metrics only.
- Select one cone mix per seed before held-out scoring.
- Score held-out rows once and document whether the oracle capacity is recoverable without leakage.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F063 train-only residual-cone selector
