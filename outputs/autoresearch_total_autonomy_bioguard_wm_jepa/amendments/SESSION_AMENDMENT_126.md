# Session Amendment 126: Strict scRNA Imaging Fresh Dataset Preflight

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from `F101_CPG0003_ROSETTA_SMALL_SCALE_CALIBRATION`

## Evidence
Rosetta repairs did not produce a registered pass. F099 diagnosed a source-state
contract and validator mismatch. F100 zero-signature source failed. F101
train-only small-scale calibration fixed delta cosine but left transition
improvement slightly negative on all splits.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`

## Implementation Tasks
- Do not promote F097, F098, F100, or F101.
- Do not redesign the model from the Rosetta failure.
- Treat cpg0003 Rosetta as an auxiliary L1000 plus Cell Painting stress test,
  not as strict paired scRNA plus imaging validation.
- Search for or recover a strict paired scRNA plus imaging fresh validation
  protocol, preferably scGeneScope with an unused sealed split or another public
  paired dataset.
- Run only metadata, obs-only, backed, or manifest-level checks first.
- Keep raw data and checkpoints outside git.
- Use GPU for model runs unless unavailable or occupied.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
If no strict fresh paired scRNA plus imaging dataset is available, document that
as a validation blocker rather than promoting from Rosetta.
