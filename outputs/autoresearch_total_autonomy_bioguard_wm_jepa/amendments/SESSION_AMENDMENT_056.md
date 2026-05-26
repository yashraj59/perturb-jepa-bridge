# Session Amendment 056: Cell-JEPA Warmstart On Descriptor-Aligned Benchmark

## Trigger
`F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN`

## Evidence
F026 approved the new `synth_program_aligned_genetic_lite` benchmark: true synthetic `z_bio` plus non-exact program actions passed the active Step 0 gates, and observed RNA PCA also passed. Before spending more compute on a full cross-modal/action JEPA, the representation path should be checked on the repaired benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F027_CELLJEPA_PROGRAM_ALIGNED_WARMSTART`

## Implementation Tasks
- Train a tiny Cell-JEPA-style RNA warmstart on train expression rows only.
- Use masked student RNA, unmasked EMA teacher RNA, stop-gradient latent targets, cosine JEPA loss, and light reconstruction anchor.
- Compare frozen Cell-JEPA `z_bio` against true synthetic `z_bio` and train-only observed RNA PCA under non-exact program-action transition floors.
- Do not use exact held-out perturbation one-hot features, `condition_key`, `biological_key`, held-out target means, or pooled train+test statistics.

## Decision Use
If Cell-JEPA preserves the repaired benchmark floor, proceed to a small full cross-modal/action JEPA probe. If Cell-JEPA falls below observed RNA PCA, prioritize representation repair or PCA-teacher distillation instead of transition residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F027 Cell-JEPA warmstart on descriptor-aligned benchmark
