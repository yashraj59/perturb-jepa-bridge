# Session Amendment 055: Descriptor-Aligned Synthetic Benchmark Audit

## Trigger
`F025_NON_EXACT_PROGRAM_ACTION_DESCRIPTOR_PRESERVES_OLD_FLOOR`

## Evidence
F022 showed that true synthetic `z_bio` does not reproduce the current positive transition floor. F024/F025 localized a descriptor-support problem: exact held-out perturbation dimensions are unsupported, and pure program descriptors fail. Inspecting the synthetic generator shows why: perturbation directions are independent random vectors, so the program descriptor is not causally tied to the held-out target direction.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F026_DESCRIPTOR_ALIGNED_SYNTHETIC_BENCHMARK_AUDIT`

## Implementation Tasks
- Add a new named synthetic config that leaves the old benchmark locked.
- Generate perturbation directions from shared program factors plus small perturbation-specific noise.
- Keep genetic-guide dose fixed and ignore chemical dose for this benchmark.
- Run Step 0 floors across seeds 0-4 using true synthetic `z_bio`, clean RNA PCA, and observed RNA PCA.
- Compare source-as-target, mean delta, source-only ridge, full action ridge, supported-action ridge, and non-exact program-action ridge.
- Use train rows only for PCA/ridge fitting and eval rows only for scoring.

## Decision Use
Reopen JEPA architecture only if the new true `z_bio` plus non-exact program action descriptors passes the active gates and the direction geometry audit proves held-out program support. Otherwise keep architecture cooled and continue metric/data redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F026 descriptor-aligned synthetic benchmark audit
