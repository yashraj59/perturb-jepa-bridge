# Debate Council 126

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from F101 cpg0003 Rosetta small-scale calibration

## Evidence Summary
F099 was an artifact-only source-geometry audit. It showed that the F098 Rosetta
same-condition replicate protocol has source-to-target cosine near 0.949 with a
rank-one repeated source. The uncalibrated JEPA transition slightly improves
absolute target cosine but has strongly negative delta cosine. Full train-only
delta calibration improves delta cosine but damages absolute target cosine.

F100 tested a zero-signature source-state contract. It still failed, with
negative transition improvement and extremely large magnitude ratios because the
source-target deltas are tiny under that contract.

F101 tested train-fold selected small-scale delta calibration. It fixed delta
cosine and preserved recall at the floor, but transition improvement stayed just
below zero on every split.

## Independent Proposals
- Methodologist: stop treating Rosetta as a promotion validator; its L1000 plus
  Cell Painting geometry conflicts with the scGeneScope transition contract.
- Architect: do not redesign the model from the Rosetta failure; no architecture
  evidence was isolated.
- Skeptic: do not relax the pass gate after seeing F101's near miss; promotion
  still requires a fresh strict scRNA plus imaging Tier 3 pass.
- Monitor: push the stopped stage and resume with strict scRNA plus imaging data
  discovery/preflight.

## Decision
`COUNCIL_STOP_ROSETTA_PROMOTION_LOOP`

## Exact Next Amendment
Execute `F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`.
