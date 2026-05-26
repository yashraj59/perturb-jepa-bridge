# Session Amendment 100: z_bio Observability Audit

## Trigger
`F070_TRUE_ZBIO_ONLY_REDUCES_MISMATCH_RNA_OBSERVATION_LIMIT`

## Evidence
F070 showed that observed RNA PCA, observed-count PCA, and train-only batch residualization do not reduce the calibration mismatch. Only true synthetic `z_bio` clearly reduces both transition and delta optimism, while clean RNA helps delta but not transition. Before training another JEPA representation, the loop must test whether `z_bio` is recoverable from the available observed modalities at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F071 is diagnostic and cannot promote.

## New Diagnostic Branch
`F071_ZBIO_OBSERVABILITY_AUDIT`

## Implementation Tasks
- Reuse the F068/F070 split-contract benchmarks and seeds.
- Fit train-only ridge probes from observed RNA, batch-residualized RNA, clean RNA, image, and RNA+image features to synthetic `z_bio`.
- Apply probes to eval rows without refitting or using eval targets.
- Rerun train-inner versus real-heldout transition/delta/recall optimism on the recovered latent states.
- Report eval cell-level `z_bio` recovery cosine, latent rank, perturbation probe, and batch probe.
- Do not promote and do not use this supervised oracle path as a candidate main representation.

## Decision Use
If train-only observation-to-`z_bio` recovery reduces mismatch, design a self-supervised/cross-modal representation repair branch to approximate this target without oracle labels. If not, pivot to generator/data-contract redesign before more architecture search.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F071 train-only z_bio observability audit
