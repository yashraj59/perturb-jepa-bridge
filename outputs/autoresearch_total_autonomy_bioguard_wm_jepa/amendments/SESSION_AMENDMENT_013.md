# Session Amendment 013: Condition-Level Blend Calibration

## Trigger
`D001_PROTOTYPE_TRANSITION_BELOW_FLOOR`

## Evidence
D001 prototype transition diagnostic fell below the protected full-ridge transition floor and reduced delta rank, so prototype transport is not a safe continuation path under the current synthetic benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, simple source/action blend deployment, and prototype transition are cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, stricter condition-level blend calibration diagnostic.

## Exact Next Experiment
`E003_CONDITION_LEVEL_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Use train-only condition/action grouped folds for source/action blend selection.
- Require nonnegative fold-level recall preservation and near-nonnegative delta cosine preservation before selecting a nonzero source-only component.
- Evaluate held-out perturbations only after train-only selection.

## Gates
Diagnostic only; no model promotion and no replacement of the protected transition floor.

## Do-Not-Run List
Do not select blend weight from held-out perturbation rows or promote an eval-only improvement.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E003 condition-level source/action blend calibration
