# Session Amendment 044: F017 Environment Blend Oracle Capacity

## Trigger
`F016_TRAIN_ONLY_BLEND_SELECTS_FLOOR_FALLBACK`

## Evidence
F016 found nonzero environment blends with mean transition gains, but every train-only candidate that mattered carried perturbation-fold recall risk. The calibrated rule therefore selected the exact floor fallback.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No oracle diagnostic can promote.

## New Or Reopened Family
Family F: Metric And Data Redesign, environment-blend capacity branch.

## Exact Next Experiment
`F017_ENVIRONMENT_BLEND_ORACLE_CAPACITY`

## Implementation Tasks
- Fit the same train-only per-seed floor and pooled/environment candidates.
- On held-out eval rows, measure an oracle row-level mask that only activates a blend when it preserves top-1 condition retrieval and row-level transition improvement relative to the floor.
- Report whether any candidate has nonzero capacity under this idealized mask.
- Treat the result as capacity-only; do not use it as a calibration rule.

## Gates
Diagnostic only. If oracle capacity is absent, cool environment blending. If oracle capacity exists, the next step must be a train-only risk proxy, not direct promotion.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F017 environment blend oracle capacity
