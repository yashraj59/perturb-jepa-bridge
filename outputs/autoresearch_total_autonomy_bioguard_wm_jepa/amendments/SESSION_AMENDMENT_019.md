# Session Amendment 019: Train-Only Heldout-Set Geometry Audit

## Trigger
`E008_NESTED_HIGH_SUPPORT_CALIBRATION_FAILED_TRAIN_ONLY`

## Evidence
E008 failed nested train-only calibration: support-gated blending selected nonzero rules in all outer folds but had negative mean outer transition gap and a negative delta-cosine worst case. This closes the current support-gated blend family under the present data geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated source/action blending, global blending, prototype transport, and residual-only predictor heads remain non-promotable on this split.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only heldout-set geometry audit.

## Exact Next Experiment
`F008_TRAIN_ONLY_HELDOUT_SET_GEOMETRY_AUDIT`

## Implementation Tasks
- Enumerate train-only pseudo-heldout triplets of perturbations.
- Refit the full action-ridge floor on remaining train perturbations.
- Score pseudo-heldout triplets and relate failures to train delta support.
- Use this only to decide whether benchmark/split geometry needs redesign before more JEPA candidates.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not use held-out eval rows in this audit and do not select a model from these pseudo-heldout sweeps.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F008 train-only heldout-set geometry audit
