# Session Amendment 051: F022 RNA Representation Ceiling Audit

## Trigger
`F021_CELLJEPA_WARMSTART_DISCARDED_NO_SAFE_REPRESENTATION_GAIN`

## Evidence
F021 implemented a JEPA-dominant Cell-JEPA RNA warmstart, but it did not improve representation probes or preserve the protected transition floor. The next step must not be training longer by default or reopening residual/risk-gate transition operators.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F022_RNA_REPRESENTATION_CEILING_AUDIT`

## Implementation Tasks
- Compare the protected old BioFlow teacher transition floor with train-only floors built from true synthetic `z_bio`, clean RNA PCA, observed RNA PCA, and observed count PCA.
- Fit PCA bases on synthetic train rows only.
- Use eval target rows only for scoring.
- Do not use `condition_key`, `biological_key`, held-out target means, or perturbation ID one-hot shortcuts as model inputs.

## Decision Use
If true synthetic `z_bio` has transition signal but RNA projections lose it, pivot to representation extraction. If true `z_bio` also fails, pivot to benchmark/action descriptor redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F022 RNA representation ceiling audit
