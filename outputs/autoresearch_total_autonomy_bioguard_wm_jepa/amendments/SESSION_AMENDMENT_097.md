# Session Amendment 097: Data-Contract Calibration Redirect

## Trigger
`F067_CLOSE_RESIDUAL_SELECTOR_FAMILY_PIVOT_TO_DATA_CONTRACT`

## Evidence
F067 closed residual selectors under the current descriptor-aligned split contract. The next smallest evidence-based step is to test whether train-internal calibration can become predictive under a different synthetic perturbation holdout contract.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F068 is diagnostic and cannot promote.

## New Diagnostic Branch
`F068_DATA_CONTRACT_CALIBRATION_REDIRECT`

## Implementation Tasks
- Keep `synth_program_aligned_genetic_lite` locked.
- Add/use separate named synthetic split-contract configs.
- Compare extrapolative index, random perturbation, and stratified program holdout contracts.
- Use true `z_bio` and train-only observed RNA PCA representations.
- Fit PCA/ridge only on train rows for each split contract.
- Score heldout perturbations only after fitting.
- Report train-inner versus real-heldout transition/delta/recall optimism.
- Do not train JEPA and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F068 data-contract calibration redirect
