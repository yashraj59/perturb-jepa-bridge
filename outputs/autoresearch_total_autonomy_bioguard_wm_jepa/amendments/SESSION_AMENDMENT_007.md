# Session Amendment 007: Representation/Error Audit

## Trigger
`F007_REPLICATE_CEILING_DOES_NOT_EXPLAIN_ACTION_FAILURE`

## Evidence
F001-F007 show the residual failure is action-specific and not explained by action descriptor support, bootstrap uncertainty, or replicate ceiling.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and action-support-only fixes remain cooled.

## New Or Reopened Family
Family C: Representation Repair Before Operator Learning.

## Exact Next Experiment
`C001_REPRESENTATION_ERROR_AUDIT`

## Implementation Tasks
- Audit online/teacher consistency.
- Audit z_bio target/delta/floor-error ranks.
- Decompose floor error by held-out action.
- Check whether floor error couples to z_tech movement.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not retrain residuals until representation error modes are localized.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C001 representation/error audit
