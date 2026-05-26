# Session Amendment 098: Representation Noise Amplification Audit

## Trigger
`F068_TRAIN_REAL_MISMATCH_PERSISTS_ACROSS_SPLIT_CONTRACTS`

## Evidence
F068 found that extrapolative, random, and stratified split contracts all retain train-real calibration mismatch. The policy table suggests observed RNA PCA may amplify mismatch relative to true biological latents.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F069 is diagnostic and cannot promote.

## New Diagnostic Branch
`F069_REPRESENTATION_NOISE_AUDIT`

## Implementation Tasks
- Read F068 policy summary.
- Compare true `z_bio` versus observed RNA PCA calibration optimism for each split contract.
- Quantify transition and delta optimism amplification.
- Decide whether the next family should be representation denoising/latent recovery rather than another split or residual selector.
- Do not train JEPA and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F069 representation-noise amplification audit
