# Session Amendment 005: Ridge Ensemble Uncertainty Audit

## Trigger
`F005_TRAIN_ONLY_ACTION_FLOOR_FAILURES_CONFIRMED`

## Evidence
F005 showed train-only leave-action-out ridge floors can be negative for some action groups, so a safe candidate needs a train-only way to detect floor failure risk.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, uncertainty/abstention feasibility branch.

## Exact Next Experiment
`F006_RIDGE_ENSEMBLE_UNCERTAINTY_AUDIT`

## Implementation Tasks
- Fit ridge floor bootstrap ensemble on train rows only.
- Compute held-out prediction disagreement as a diagnostic uncertainty signal.
- Compare uncertainty to held-out floor failures for analysis only.

## Gates
Diagnostic only; no model promotion and no eval-driven gate deployment.

## Do-Not-Run List
Do not deploy an abstention gate unless train-only calibration can select it.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F006 ridge ensemble uncertainty audit
