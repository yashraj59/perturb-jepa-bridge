# C010 Source-Latent Neighborhood Retrieval Audit

## Decision
`C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
- Current seed nested source-neighborhood recall@1: `0.398148`.
- Current seed full-gallery endpoint recall@1: `0.265873`.
- Fresh seed nested source-neighborhood recall@1: `0.414021`.
- Protected condition recall floor: `0.481481`.
- Mean selected source-neighborhood fraction: `0.354167`.
- Mean selected delta weight: `0.506250`.

## Interpretation
This replaces exact cell-line/source-state metadata with a retrieval mask computed from source latent nearest neighbors. It is still diagnostic scoring, not a model input and not a promotion path. A robust pass would justify a JEPA source-neighborhood retrieval objective; a below-floor result means source-state structure alone is not yet enough.
