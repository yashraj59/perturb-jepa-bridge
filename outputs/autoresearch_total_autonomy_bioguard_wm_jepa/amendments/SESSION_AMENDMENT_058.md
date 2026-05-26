# Session Amendment 058: PCA-Bootstrap Cross-Modal/Action JEPA

## Trigger
`F028_PCA_DISTILLED_RNA_ENCODER_PRESERVES_PROGRAM_FLOOR`

## Evidence
F028 showed that a neural RNA encoder can preserve the repaired non-exact program-action floor when bootstrapped from train-only PCA. The next step should no longer be another representation-only diagnostic; it should test a small real JEPA world-model path while keeping the PCA anchor auxiliary.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F029_PCA_BOOTSTRAP_CROSS_MODAL_ACTION_JEPA`

## Implementation Tasks
- Train online RNA and image encoders with EMA target encoders.
- Use stop-gradient latent targets.
- Use query-conditioned predictors.
- Include RNA->image and image->RNA latent JEPA losses.
- Include control RNA + non-exact program action -> perturbed RNA latent transition loss.
- Keep train-only PCA anchor auxiliary and lower-weight than JEPA losses.
- Use program descriptors only, not exact perturbation IDs.
- Score direct JEPA transitions, representation ridge floor, cross-modal retrieval, action-negative gap, rank, identity, and leakage diagnostics.

## Decision Use
If direct JEPA transitions pass gates and cross-modal retrieval is healthy, continue to a multi-seed/Tier 2 amendment. If the representation ridge floor remains good but direct JEPA transitions fall below it, pivot to transition predictor optimization rather than representation repair.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F029 PCA-bootstrap cross-modal/action JEPA
