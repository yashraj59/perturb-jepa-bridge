# G002 Multi-Seed Fresh-Cache Replication

## Decision
`G002_FRESH_REPLICATION_CONFIRMS_SOURCE_PROXY_INSTABILITY_BELOW_FLOOR`

## Evidence
- Protected condition recall floor: `0.481481`
- Completed fresh seeds: `[1, 2, 3, 4]`
- Mean online-source-neighborhood recall: `0.425926`
- Minimum online-source-neighborhood recall: `0.383598`
- Fraction of fresh seeds at/above floor: `0.000`
- Mean best train-only composite recall: `0.406085`
- Mean floor condition recall: `0.296296`

## Interpretation
This diagnostic extends the seed-1 fresh-cache audit to multiple synthetic seeds. It uses train-only calibration and audit labels only; no source-state metadata is used as model input and no model is promoted.

If online-source-neighborhood recall remains below floor across fresh seeds, source-state preservation training stays cooled and the next council should favor metric/data diagnostics or a benchmark redesign rather than a source-state preserving JEPA objective.
