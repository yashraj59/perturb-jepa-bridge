# Session Amendment 124: Frozen Selector Confirmation Or Gate Contract Audit

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from `F095_NON_EXACT_PUBCHEM_FINGERPRINT_DESCRIPTOR_RERUN`

## Evidence
F095 showed that non-exact PubChem fingerprints materially improve the scGeneScope external floor comparison. The official split-safe gate still failed by a small alternate-test recall gap. The calibrated fingerprint row cleared all floor gaps, but it was not the predeclared selected candidate and cannot be promoted post hoc.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real Tier 3 pass explicitly supersedes it.

## New Branch
`F096_FROZEN_SELECTOR_CONFIRMATION_OR_GATE_CONTRACT_AUDIT`

## Implementation Tasks
- Do not promote F095.
- Freeze the useful candidate lesson: PubChem fingerprint descriptors plus train-only JEPA calibration are the next candidate family.
- Before any promotion language, either validate a frozen selector on a genuinely new external protocol or run a non-promoting gate-contract audit.
- The gate audit must explain whether action-heldout train CV is the wrong selector for same-treatment replicate/round external validation.
- Keep PLS/full-ridge as audit floor only.
- Continue to run model work on GPU unless unavailable or occupied.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F096 frozen selector confirmation or gate contract audit
