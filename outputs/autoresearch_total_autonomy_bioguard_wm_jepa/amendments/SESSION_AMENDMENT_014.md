# Session Amendment 014: Support-Aware Blend Calibration

## Trigger
`E003_CONDITION_BLEND_REJECTED_KEEP_FULL_FLOOR`

## Evidence
E003 showed that nonzero source/action blending can improve mean train-only transition metrics but violates worst-case fold preservation. C003 showed weak latent delta-manifold support predicts floor failures, so the next test must use a deployable support score rather than held-out targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global source/action blending, and prototype transition remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, support-aware safe floor-composition diagnostic.

## Exact Next Experiment
`E004_SUPPORT_AWARE_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Compute support from predicted deltas against train-only delta references.
- Select a support threshold and source-only blend weight by train-only leave-action-out calibration.
- Default to the protected full-ridge floor unless train-only gates preserve recall and delta-cosine worst cases.

## Gates
Diagnostic only; no model promotion and no replacement of the protected transition floor.

## Do-Not-Run List
Do not use held-out target deltas, condition keys, or exact target means for support scoring or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E004 support-aware source/action blend calibration
