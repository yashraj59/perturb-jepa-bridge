# Session Amendment 076: Target/Action Support Geometry Audit

## Trigger
`F046_OPERATOR_TRAIN_HELDOUT_MISMATCH_DOMINATES`

## Evidence
F046 showed that the exact floor-initialized operator wrapper selected nonzero tiny residuals from train-only metrics, but held-out transition and delta metrics fell below the protected floor on most seeds. The next branch must identify whether this is caused by held-out action/delta/residual targets being outside train support.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F047_TARGET_GEOMETRY_AUDIT`

## Implementation Tasks
- Use active cached multiseed latent bundles, seeds 0-4.
- Fit only the train full-ridge transition floor per seed.
- Measure held-out perturbation/action exact support, active action-dimension support, action cosine support, teacher-delta support, predicted-delta support, residual support, source support, and target support.
- Compare held-out geometry against train-only leave-action pseudoheldout geometry.
- Do not fit, calibrate, select, or promote a new candidate.

## Decision Use
If held-out exact actions or residual targets are outside train support, pivot to action-target contract or representation/benchmark redesign. If support is sufficient, reopen a targeted operator mechanism with the support failure ruled out.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F047 target/action support geometry audit
