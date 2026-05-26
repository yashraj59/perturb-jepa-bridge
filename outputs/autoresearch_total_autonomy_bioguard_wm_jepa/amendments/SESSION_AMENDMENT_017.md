# Session Amendment 017: High-Support Abstention Calibration

## Trigger
`E006_ORACLE_SUPPORT_GATE_HAS_CAPACITY_CALIBRATION_FAILURE`

## Evidence
E006 showed support-threshold gating has held-out oracle capacity, but only in a high-threshold, low-active-fraction regime. Because E006 used held-out labels, the next run must use train-only calibration and remain diagnostic only.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, and low-threshold support blending remain non-promotable.

## New Or Reopened Family
Family E: Program And Graph Action Priors, high-support abstention calibration diagnostic.

## Exact Next Experiment
`E007_HIGH_SUPPORT_ABSTENTION_CALIBRATION_DIAGNOSTIC`

## Implementation Tasks
- Select only high support thresholds with low active fraction using train-only leave-action-out folds.
- Require recall preservation and bounded transition/delta worst-case regressions on train folds.
- Evaluate held-out rows after selection, but mark any pass as oracle-informed diagnostic only.

## Gates
Diagnostic only; no model promotion because this family was designed after an E006 held-out oracle audit.

## Do-Not-Run List
Do not promote E007 or use it as a model of record without a fresh split/Tier 2 amendment.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E007 high-support abstention calibration
