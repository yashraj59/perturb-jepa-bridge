# Session Amendment 109: Source Delta Rank JEPA Wrapper

## Trigger
`F079_SOURCE_DELTA_RANK_REPAIR_READY_FOR_WRAPPER`

## Evidence
F079 repaired the source-program branch enough to justify the previously deferred full wrapper: transition improved to a useful Tier 1 diagnostic level while delta cosine, recall, and rank were preserved. The next falsifiable question is whether that repair remains valid when implemented as a real cross-modal, action-conditioned JEPA rather than a standalone image-teacher predictor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F080 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F080_SOURCE_DELTA_RANK_JEPA_WRAPPER`

## Implementation Tasks
- Use ProgramBootstrapJEPA online RNA/image encoders and EMA target encoders.
- Preserve stop-gradient latent teachers and query-conditioned predictors.
- Include RNA->image, image->RNA, and control+coarse-program-action transition losses.
- Carry forward F079 source teacher, target teacher, delta-direction, source-improvement, magnitude, relational, and delta-rank constraints.
- Use train-only control expression/image PCA centroids for source state/teacher.
- Use train-only condition-centroid image PCA for training target anchors.
- Use coarse program one-hot action descriptors only; do not use exact perturbation one-hot.
- Score learned-latent transition and image-teacher-aligned transition separately.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F080 preserves F079 transition signal and cross-modal retrieval, design a low-compute Tier 2/no-regression validation. If the full wrapper loses transition signal, localize whether EMA teacher lag, target anchoring, or coarse action descriptors are the bottleneck.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F080 full source/delta/rank JEPA wrapper
