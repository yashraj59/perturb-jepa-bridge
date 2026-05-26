# Session Amendment 008: Latent Delta Manifold Support Audit

## Trigger
`C001_ACTION9_HIGH_FLOOR_ERROR_WITH_HEALTHY_RANK`

## Evidence
C001 showed the worst action has high floor error even though online/teacher consistency and latent rank are healthy.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and simple support-distance fixes remain cooled.

## New Or Reopened Family
Family C: Representation Repair, latent response manifold branch.

## Exact Next Experiment
`C002_LATENT_DELTA_MANIFOLD_SUPPORT_AUDIT`

## Implementation Tasks
- Compute train action mean teacher deltas.
- Compare held-out action mean deltas to train delta manifold.
- Relate delta-manifold support to action-level floor failure.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not train new architecture until response support is measured.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C002 latent delta manifold support audit
