# C013 Online-Source-Neighborhood Retrieval Audit

## Decision
`C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
- Current seed online-source-neighborhood recall@1: `0.457672`.
- Current seed full-gallery endpoint recall@1: `0.265873`.
- Fresh seed online-source-neighborhood recall@1: `0.418651`.
- Protected condition recall floor: `0.481481`.
- Mean selected source-neighborhood fraction: `0.333333`.
- Mean selected delta weight: `0.573214`.

## Interpretation
C012 showed online z_bio has much cleaner source-state neighborhoods than teacher z_bio. This audit tests whether that cleaner neighborhood geometry translates into retrieval repair under nested train-only calibration.
