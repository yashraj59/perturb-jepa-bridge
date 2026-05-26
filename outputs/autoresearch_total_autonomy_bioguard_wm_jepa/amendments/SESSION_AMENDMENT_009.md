# Session Amendment 009: Train-Only Delta Support Audit

## Trigger
`C002_WEAK_DELTA_MANIFOLD_SUPPORT_FOR_BAD_ACTION`

## Evidence
C002 showed the weakest held-out action also had the weakest cosine support from train action mean deltas.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and simple uncertainty gates remain cooled.

## New Or Reopened Family
Family C: Representation Repair, train-only delta support validation.

## Exact Next Experiment
`C003_TRAIN_ONLY_DELTA_SUPPORT_AUDIT`

## Implementation Tasks
- Leave out each train perturbation action.
- Compute nearest remaining train action delta support.
- Fit ridge on remaining train rows only and score the left-out train action.
- Test whether weak delta support predicts floor failure.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch graph/action descriptor architecture until this train-only validation is logged.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C003 train-only delta support audit
