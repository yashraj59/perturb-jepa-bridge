# Session Amendment 083: Fresh Descriptor Margin-Gated Rerun

## Trigger
`F053_DESCRIPTOR_MARGIN_GATE_ZERO_FALLBACK_INSUFFICIENT_TRAIN_DIAGNOSTICS`

## Evidence
F053 repaired safety by falling back to the exact floor, but that was not a scientific improvement. The missing piece is a fresh run that logs train near-tie lower-tail diagnostics before held-out scoring, using a smaller residual scale grid.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F054 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F054_DESCRIPTOR_MARGIN_RERUN`

## Implementation Tasks
- Use fresh seeds 13, 14, and 15.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Train the same low-compute real ProgramBootstrapJEPA path.
- Select scale only from train rows using near-tie lower-tail diagnostics on `(0, 0.005, 0.01, 0.0125, 0.025, 0.05)`.
- Score held-out rows only after train-only scale selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F054 fresh descriptor margin-gated rerun
