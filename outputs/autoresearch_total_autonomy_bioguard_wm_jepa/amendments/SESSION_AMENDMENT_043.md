# Session Amendment 043: F016 Environment Blend Calibration

## Trigger
`F015_POOLED_ENVIRONMENT_TRANSITION_PASSES_GATES_BELOW_PER_SEED_FLOOR`

## Evidence
F015 found pooled/environment transition headroom, but the strongest transition baselines regressed condition recall below the per-seed action-ridge floor. This makes immediate neural residual training premature.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Phase 8 v3 or Family F diagnostic can promote.

## Protected Transition Floor
The full train-only action-ridge floor remains the transition floor for residual/operator candidates. Zero blend scale must reproduce the per-seed floor exactly.

## New Or Reopened Family
Family F: Metric And Data Redesign, environment-blend calibration branch.

## Exact Next Experiment
`F016_ENVIRONMENT_BLEND_CALIBRATION`

## Implementation Tasks
- Build convex blends from each seed's per-seed action-ridge floor toward pooled source-only, pooled full, and environment-centered transition predictions.
- Select blend family and scale only by train-split global perturbation-group cross-fitting.
- Score held-out eval rows only after calibration is fixed.
- Default to the floor if no train-only rule preserves transition, recall, and delta cosine.

## Gates
Diagnostic only. A nonzero blend must preserve the per-seed floor on held-out scoring before it can justify a later JEPA mechanism. It cannot promote the model of record.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F016 train-only environment blend calibration
