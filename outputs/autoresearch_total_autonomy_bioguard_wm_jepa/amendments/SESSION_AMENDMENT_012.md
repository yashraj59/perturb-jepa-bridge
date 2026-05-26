# Session Amendment 012: Prototype Transition Diagnostic

## Trigger
`E002_BLEND_HELDOUT_RECALL_DROP`

## Evidence
E002 found a train-only selected source/action blend improves transition and delta cosine but still drops held-out recall below the protected floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and simple floor-blend deployment remain cooled.

## New Or Reopened Family
Family D: Population Transport JEPA, prototype diagnostic branch.

## Exact Next Experiment
`D001_PROTOTYPE_TRANSITION_DIAGNOSTIC`

## Implementation Tasks
- Build train-only source-state prototypes.
- Fit prototype+action ridge transition on train rows.
- Score held-out perturbations and action heterogeneity.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not call this a JEPA candidate or promote it.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: D001 prototype transition diagnostic
