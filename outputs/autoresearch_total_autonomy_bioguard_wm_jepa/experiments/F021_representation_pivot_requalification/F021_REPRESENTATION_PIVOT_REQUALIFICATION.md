# F021 Representation Pivot Requalification

## Decision
`F021_REOPEN_REPRESENTATION_REPAIR_UNDER_ACTIVE_MULTISEED_GATES`

## Active Multiseed Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- condition recall@1 >= `0.255644`

## Evidence Table
| source | status | transition | delta | recall | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| F020 | F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK | 0.011281 | 0.451515 | 0.333333 | environment/risk branch selected floor fallback |
| G002 | G002_MULTI_SEED_SOURCE_STATE_SIGNAL_UNSTABLE_NO_TRAINING_REOPEN | 0.009311 | 0.440083 | 0.430886 | multi-seed source-state stability diagnostic rechecked under active gates |
| C012 | C012_SOURCE_STATE_SIGNAL_STRONGER_OUTSIDE_Z_BIO | 0.005662 | 0.397963 | 0.911229 | source-state signal stronger outside z_bio |
| C013 | C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR | 0.005662 | 0.397963 | 0.457672 | online-source-neighborhood retrieval partial repair |

## Interpretation
The environment-blend/risk branch is cooled after F020. Prior representation diagnostics were originally judged against the old seed0 recall floor, but the active named multiseed gates are now lower and explicitly registered in F014. This audit decides whether representation repair can be reopened without treating a Tier 1 diagnostic as promotion.
