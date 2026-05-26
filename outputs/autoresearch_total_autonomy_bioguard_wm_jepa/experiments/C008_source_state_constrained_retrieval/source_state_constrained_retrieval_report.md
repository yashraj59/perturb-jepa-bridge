# C008 Source-State Constrained Retrieval Audit

## Decision
`C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_REACHES_RECALL_FLOOR_DIAGNOSTIC`

## Evidence
- Current seed best constrained condition recall@1: `0.486111` at delta weight `1.00`.
- Current seed constrained endpoint-only recall@1: `0.336640`.
- Protected condition recall floor: `0.481481`.

## Interpretation
This audit uses cell-line/source-state metadata only as an evaluation constraint, not as a model input and not for promotion. It tests whether retrieval failure mainly comes from cross-source-state gallery competition. A pass would justify a source-state-aware objective; a below-floor result means perturbation-level separation is still insufficient even after restricting the gallery.
