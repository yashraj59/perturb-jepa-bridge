# C009 Nested Source-State Retrieval Calibration Audit

## Decision
`C009_NESTED_SOURCE_STATE_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
- Current seed nested selected condition recall@1: `0.480159`.
- Current seed endpoint-only constrained recall@1: `0.336640`.
- Fresh seed nested selected condition recall@1: `0.447090`.
- Protected condition recall floor: `0.481481`.

## Interpretation
This audit removes the C008 oracle choice by selecting the source-state constrained endpoint/delta weight from inner train-action folds. The source-state constraint remains an evaluation constraint only; it is not a model input and it does not alter promotion gates.
