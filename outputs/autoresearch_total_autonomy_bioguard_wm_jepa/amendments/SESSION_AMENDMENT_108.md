# Session Amendment 108: Source Delta Rank Repair

## Trigger
`F078_SOURCE_PROGRAM_RECALL_WITH_LOW_TRANSITION_AND_ACTION_SIGNAL`

## Evidence
F078 localized the F077 weak pass as high recall but low transition improvement, low perturbation-probe signal, and low effective rank. This suggests endpoint/prototype prediction is insufficient; the next diagnostic must directly optimize source improvement and delta geometry before trying richer action descriptors or a full JEPA wrapper.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F079 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F079_SOURCE_DELTA_RANK_REPAIR`

## Implementation Tasks
- Keep coarse program-action descriptors only; do not use exact perturbation one-hot features.
- Use train-only control expression centroids as source state.
- Use train-only control image PCA centroids as source teacher targets.
- Use train-only condition-centroid image PCA as perturbed teacher targets.
- Add source endpoint, target endpoint, delta-direction, source-improvement hinge, magnitude, relational, and delta-rank variance losses.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, image-latent cosine, and train-loss diagnostics.
- Compare against F077 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F079 repairs transition while preserving recall/delta, build a real source/action-conditioned JEPA wrapper. If transition stays low, pivot to action descriptor adequacy or benchmark identifiability audits.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F079 source/delta/rank-constrained program repair
