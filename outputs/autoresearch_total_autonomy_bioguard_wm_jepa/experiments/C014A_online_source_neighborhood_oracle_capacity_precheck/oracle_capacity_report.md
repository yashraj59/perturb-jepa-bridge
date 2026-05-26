# C014 Online-Source-Neighborhood Oracle Capacity Audit

## Decision
`C014_ORACLE_CAPACITY_CURRENT_ONLY_FRESH_BELOW_FLOOR`

## Evidence
- Protected condition recall floor: `0.481481`.
- Current seed selected recall: `0.457672`.
- Current seed oracle recall: `0.499339`.
- Fresh seed selected recall: `0.418651`.
- Fresh seed oracle recall: `0.451058`.
- Current selected-to-oracle gap: `0.041667`.
- Fresh selected-to-oracle gap: `0.032407`.

## Interpretation
The current seed has oracle capacity above the protected recall floor, but the fresh seed does not. This means C013 is not only an inner-calibration failure; the online-source-neighborhood repair is seed-unstable. A source-state-preserving JEPA objective remains risky unless the next step addresses seed stability or benchmark/data geometry rather than simply training toward source neighborhoods.

## Leakage And Promotion Status
This audit uses the C013 train-only outer-grid artifacts as diagnostic evidence. Oracle-best rules are not deployable and must not be used for model selection or promotion. No model is promoted.
