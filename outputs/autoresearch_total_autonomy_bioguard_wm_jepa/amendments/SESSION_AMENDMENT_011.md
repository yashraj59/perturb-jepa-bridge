# Session Amendment 011: Train-Only Source/Action Blend Audit

## Trigger
`E001_SOURCE_ONLY_TRANSITION_IMPROVES_BUT_RECALL_COLLAPSES`

## Evidence
E001 showed source-only improves transition and delta cosine but collapses retrieval, while source+action preserves recall but has lower transition gain.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and action-only descriptor fixes remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, safe floor-composition diagnostic.

## Exact Next Experiment
`E002_TRAIN_ONLY_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Blend source-only and source+action ridge floors.
- Select blend weight by train-only leave-action-out calibration.
- Evaluate held-out perturbations only after selection.

## Gates
Diagnostic only; no model promotion and no replacement of protected floor.

## Do-Not-Run List
Do not use held-out rows for selecting blend weight.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E002 train-only source/action floor blend calibration
