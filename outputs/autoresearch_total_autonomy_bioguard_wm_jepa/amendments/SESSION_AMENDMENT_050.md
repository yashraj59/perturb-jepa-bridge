# Session Amendment 050: F021 Cell-JEPA RNA Representation Warmstart

## Trigger
`C016_ONLINE_LATENT_REPRESENTATION_HAS_REPAIR_SIGNAL`

## User Amendment Integrated
The user added `papers/2602.02093v1.pdf` and instructed that Cell-JEPA should be used as a single-cell RNA representation warmstart, not as a promoted perturbation-transition solution.

## Paper Consulted
`papers/2602.02093v1.pdf`

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## Protected Transition Floor
The protected full train-only action-ridge transition floor remains the transition reference for residual/operator candidates.

## New Diagnostic Branch
`F021_CellJEPA_RNA_Representation_Warmstart`

## Implementation Tasks
- Train a tiny Cell-JEPA-style RNA warmstart on synthetic train rows only.
- Student RNA encoder receives masked expression values.
- EMA teacher RNA encoder receives unmasked expression values.
- Student predictor predicts stop-gradient teacher cell embeddings.
- Use cosine JEPA loss plus a light reconstruction anchor; `w_jepa` must dominate `w_rec`.
- Support per-cell quantile binning, random expressed-gene subsampling, and expression-value masking only.
- Evaluate representation before transition: same-cell/same-condition retrieval, RNA-image retrieval, rank, perturbation probe, batch probe, dropout robustness, and leakage controls.
- Rerun the protected train-only action-ridge floor on frozen Cell-JEPA `z_bio`.

## Gates
F021 cannot promote a transition model. Residual/risk-gate search reopens only if frozen warmstarted `z_bio` improves or preserves transition improvement, recall@1, delta cosine, effective rank, and per-seed stability.

## Forbidden Inputs
No perturbation ID one-hot as held-out shortcut, no `condition_key` or `biological_key` model input, no held-out target means, and no promotion based only on absolute post-perturbation Pearson.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F021_CellJEPA_RNA_Representation_Warmstart
