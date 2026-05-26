# Session Amendment 004: Train-Only Leave-Action-Out Floor Audit

## Trigger
`F004_SUPPORT_DISTANCE_DOES_NOT_EXPLAIN_NEGATIVE_ACTION`

## Evidence
F004 showed descriptor support distance is identical for held-out perturbations and does not explain the negative perturbation group.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Action descriptor engineering is deferred until train-only generalization failures are quantified.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only action generalization audit.

## Exact Next Experiment
`F005_TRAIN_ONLY_LEAVE_ACTION_OUT_FLOOR_AUDIT`

## Implementation Tasks
- Leave each train perturbation group out.
- Fit the ridge floor on remaining train rows only.
- Score the left-out train action group.
- Use no eval/test target rows for fitting or selection.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch new residual or graph-action model until this audit is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F005 train-only leave-action-out floor audit
