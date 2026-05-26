# Session Amendment 018: Nested Train-Only Calibration Audit

## Trigger
`E007_HIGH_SUPPORT_CALIBRATION_HELDOUT_RECALL_DROP`

## Evidence
E007 still dropped held-out recall, so its train-only calibration may be unstable rather than merely too conservative or too weak.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and support-gated source/action blending remain non-promotable on the current held-out split.

## New Or Reopened Family
Family E: Program And Graph Action Priors, nested train-only calibration stability audit.

## Exact Next Experiment
`E008_NESTED_TRAIN_ONLY_HIGH_SUPPORT_CALIBRATION_AUDIT`

## Implementation Tasks
- Use outer train perturbations as pseudo-heldout groups.
- Select the high-support rule only from inner train perturbations.
- Score the selected rule on the outer train perturbation and compare to that fold's full-ridge floor.

## Gates
Diagnostic only. Failure means the support-gated blend family lacks stable train-only calibration under current data.

## Do-Not-Run List
Do not use held-out eval rows in this audit.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E008 nested train-only high-support calibration audit
