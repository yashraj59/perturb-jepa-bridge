# Session Amendment 015: Predicted Support Score Audit

## Trigger
`E004_SUPPORT_AWARE_BLEND_HELDOUT_BELOW_FLOOR`

## Evidence
E004 selected a nonzero support-aware source/action blend by train-only calibration, but the held-out score still dropped below the protected full-ridge floor on recall. The support score may not distinguish held-out unsafe actions.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, support-aware blending, and prototype transition remain non-promotable under current evidence.

## New Or Reopened Family
Family E: Program And Graph Action Priors, predicted-support failure localization.

## Exact Next Experiment
`E005_PREDICTED_SUPPORT_SCORE_AUDIT`

## Implementation Tasks
- Compare deployable predicted-delta support against teacher-delta support as an audit only.
- Measure whether predicted support correlates with held-out action transition success.
- Use the result to choose between representation-support repair and synthetic benchmark geometry expansion.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not use teacher support for model selection or held-out gating.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E005 predicted support score failure audit
