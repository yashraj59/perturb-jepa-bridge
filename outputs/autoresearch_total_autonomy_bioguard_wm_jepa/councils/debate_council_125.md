# Debate Council 125

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from F098 cpg0003 Rosetta replicate-holdout confirmation

## Evidence Summary
F097 implemented a fresh cpg0003 Rosetta confirmation runner for the frozen F096
path. The first run used compound-holdout splits and failed. Because that split
was stricter than the scGeneScope replicate-heldout contract, F098 reran with a
same-condition replicate-holdout split.

F098 passed preflight with 1,469 train condition pairs and held-out replicate
pairs across validation, test, and alternate_test. Identity and leakage flags
were zero. It improved over the protected full-ridge audit floor on transition
and delta cosine, but absolute transition improvement stayed negative and the
test recall floor gap was negative.

## Independent Proposals
- Methodologist: run an artifact-only Rosetta geometry audit before changing the
  model, because the source-as-target and transition-improvement semantics may
  differ from scGeneScope on centered L1000/Cell Painting profiles.
- Architect: try a Rosetta-specific source-state definition, but only after the
  audit shows the current control source is invalid for these profiles.
- Skeptic: do not use cpg0003 for promotion because it is L1000 rather than
  scRNA, and both F097/F098 failed.
- Monitor: no promotion; keep PLS/full-ridge as audit floor only; push the
  stopped stage for reproducibility.

## Decision
`COUNCIL_EXECUTE_ARTIFACT_ONLY_DIAGNOSTIC`

## Exact Next Amendment
Execute `F099_ROSETTA_SOURCE_GEOMETRY_AUDIT`.
