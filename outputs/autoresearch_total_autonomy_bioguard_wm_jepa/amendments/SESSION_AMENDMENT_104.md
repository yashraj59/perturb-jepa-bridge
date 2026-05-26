# Session Amendment 104: Relational Image-Teacher RNA-Student JEPA

## Trigger
`F074_FAILURE_IS_PERTURBATION_STRUCTURE_LOSS`

## Evidence
F074 showed the F073 RNA student did not simply underfit image targets. It reached moderate image-latent cosine but lost perturbation-probe signal, recall, and transition geometry relative to the image teacher. The next smallest repair is to preserve image-teacher pairwise structure during RNA-student training.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F075 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F075_RELATIONAL_IMAGE_TEACHER_RNA_STUDENT`

## Implementation Tasks
- Keep the F073 masked-RNA student and stop-gradient train-only image PCA teacher.
- Add a train-only pairwise teacher-similarity distillation term on mini-batches.
- Keep JEPA/cosine endpoint prediction dominant with light MSE and variance anchors.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare directly against F073 and the image-teacher ceiling.
- Do not use synthetic `z_bio`, `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If relational training preserves image-teacher transition structure, reopen a small cross-modal BioGuard-WM-JEPA representation branch. If it only partially repairs structure, localize whether condition grouping or action descriptors are needed. If it fails, pivot away from per-cell image-teacher distillation.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F075 relational image-teacher RNA-student JEPA
