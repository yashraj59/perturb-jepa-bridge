# Session Amendment 099: Denoised Representation Ceiling Diagnostic

## Trigger
`F069_OBSERVED_RNA_REPRESENTATION_AMPLIFIES_CALIBRATION_MISMATCH`

## Evidence
F069 showed that observed RNA PCA amplifies train-real calibration mismatch relative to true synthetic `z_bio`. The next step should identify whether a train-only denoising transform or oracle clean RNA can reduce that mismatch before training another representation model.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F070 is diagnostic and cannot promote.

## New Diagnostic Branch
`F070_DENOISED_REPRESENTATION_CEILING`

## Implementation Tasks
- Reuse the F068 split-contract benchmarks and seeds.
- Compare observed RNA PCA, observed-count PCA, train-only batch-residualized RNA PCA, clean RNA PCA, and true `z_bio`.
- Fit PCA/ridge/batch residualization from train rows only.
- Use held-out perturbation rows only for scoring.
- Report train-inner versus real-heldout optimism, latent rank, perturbation probe, and batch probe.
- Do not train JEPA and do not promote.

## Decision Use
If a train-only denoised candidate reduces mismatch, design a small representation repair branch around that target. If only clean RNA or true `z_bio` helps, prioritize learned biological denoising rather than more transition residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F070 denoised representation ceiling diagnostic
