# Session Amendment 106: Source-State Plus Program-Action Image Teacher

## Trigger
`F076_CONDITION_CENTROID_TEACHER_STILL_LOSES_STRUCTURE`

## Evidence
F073-F076 all failed to transfer image-teacher perturbation structure into an RNA-side endpoint representation. Per-cell targets, pairwise relational image targets, and train-only condition-centroid targets all preserved only moderate image alignment while losing recall and perturbation-probe signal.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F077 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F077_SOURCE_PROGRAM_IMAGE_TEACHER`

## Implementation Tasks
- Build train-only control/source expression centroids by cell line and dose.
- Use coarse program-action one-hot descriptors only; do not use exact perturbation one-hot features.
- Predict train-only condition-centroid image PCA targets from source state plus program action.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare against F076 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F077 repairs structure, build a real source/action-conditioned JEPA wrapper around this factorization. If it fails, the issue is not endpoint noise or missing source-action factorization alone; pivot to action descriptor adequacy or synthetic benchmark identifiability audits.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F077 source-state plus program-action image teacher
