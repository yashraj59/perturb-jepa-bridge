# Session Amendment 105: Condition-Centroid Image-Teacher RNA-Student

## Trigger
`F075_RELATIONAL_IMAGE_TEACHER_STILL_LOSES_STRUCTURE`

## Evidence
F075 added pairwise image-teacher structure but still lost perturbation/action geometry relative to the image teacher. The remaining local hypothesis is that per-cell image targets are too noisy for the small RNA student, so the next diagnostic should test train-only condition-centroid image targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F076 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F076_CONDITION_CENTROID_IMAGE_TEACHER`

## Implementation Tasks
- Fit image PCA on train rows only.
- Compute train-only image-target centroids by perturbation ID, cell line, and dose.
- Train the masked-RNA student against these centroid targets with no label features as inputs.
- Use a light relational term only inside train mini-batches.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare against F075 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F076 repairs structure, reopen representation learning with train-only pseudobulk teachers. If it fails, pivot toward action-descriptor or source-state objectives because per-cell and condition-level image distillation both failed.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F076 condition-centroid image-teacher RNA-student
