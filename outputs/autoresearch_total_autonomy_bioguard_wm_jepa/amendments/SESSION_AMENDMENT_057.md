# Session Amendment 057: Train-Only PCA-Distilled RNA Encoder

## Trigger
`F027_CELLJEPA_BELOW_PCA_FLOOR_USE_REPRESENTATION_REPAIR`

## Evidence
F027 Cell-JEPA was JEPA-dominant and non-leaky but fell far below the observed RNA PCA transition floor on the repaired benchmark. This suggests the immediate issue is representation extraction/objective mismatch, not action-target geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F028_TRAIN_ONLY_PCA_DISTILLED_RNA_ENCODER`

## Implementation Tasks
- Fit an observed-RNA PCA teacher on train rows only.
- Train a tiny neural RNA encoder to predict that train-only PCA embedding from masked expression values.
- Score non-exact program-action transition floors on frozen learned embeddings.
- Compare to true synthetic `z_bio` and the original observed RNA PCA teacher.

## Decision Use
If the neural encoder preserves the PCA floor, use it as a bootstrap representation path for a full cross-modal/action JEPA probe. If it underfits, repair the learned RNA encoder objective before training more transition machinery.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F028 train-only PCA-distilled RNA encoder
