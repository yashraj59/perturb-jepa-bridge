# Session Amendment 125: Rosetta Source Geometry Audit

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from `F098_CPG0003_ROSETTA_REPLICATE_HOLDOUT`

## Evidence
F098 used the frozen F096/F082 ProgramBootstrapJEPA path with non-exact
SMILES/dose descriptors and train-only delta calibration on a fresh cpg0003
Rosetta same-condition replicate-holdout protocol. It failed promotion gates:
test recall was below the audit floor and absolute transition improvement was
negative on every split.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F099_ROSETTA_SOURCE_GEOMETRY_AUDIT`

## Implementation Tasks
- Do not train a new model.
- Do not promote F097 or F098.
- Read only F097/F098 saved artifacts and compact metadata.
- Quantify source-as-target, target-target, and control-source geometry.
- Determine whether cpg0003 L1000/Cell Painting profiles are already centered
  signatures where the current control source definition is mismatched.
- Report whether the failure is a model failure, a source-state contract failure,
  or a validator-mismatch failure.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.
