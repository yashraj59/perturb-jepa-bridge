# Session Amendment 078: Protected Floor Feature Contract Audit

## Trigger
`F048_NON_EXACT_ACTION_OPERATOR_STILL_BELOW_FLOOR`

## Evidence
F048 removed exact perturbation action one-hots and used only shared program action descriptors. The program-action floor and program-action residual stayed below the protected full floor, while the full floor remained much stronger. This suggests the protected floor may benefit from train exact-action feature centering or extrapolation even though candidate JEPA paths cannot use exact perturbation one-hots.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. This audit does not demote or replace it.

## New Diagnostic Branch
`F049_FLOOR_FEATURE_CONTRACT`

## Implementation Tasks
- Fit the protected full action-ridge floor on train only.
- Decompose held-out predictions into source, train exact-action, eval-unseen exact-action, and program-action feature contributions.
- Score the full prediction and counterfactual predictions with train exact-action contribution removed.
- Quantify whether the full floor advantage depends on exact train-action feature centering.

## Decision Use
If the full floor depends materially on exact train-action contribution, keep it as protected audit baseline but stop treating it as a candidate-legal action contract. The next branch should target a valid biological action descriptor or benchmark contract.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F049 protected floor feature contract audit
