# F017 Environment Blend Oracle Capacity

## Decision
`F017_ORACLE_SAFE_ENVIRONMENT_BLEND_CAPACITY_EXISTS`

## Best Oracle Rule
- candidate: `pooled_source_only_ridge`
- alpha: `1.000000`
- mean transition gap: `0.010501`
- mean delta gap: `0.089692`
- mean recall gap: `0.051852`

## Top Oracle Rules
| candidate | alpha | active fraction | transition gap | delta gap | recall gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| pooled_source_only_ridge | 1.000 | 0.474074 | 0.010501 | 0.089692 | 0.051852 |
| pooled_full_ridge | 1.000 | 0.496296 | 0.010263 | 0.083586 | 0.037037 |
| pooled_source_only_ridge | 0.750 | 0.555556 | 0.009360 | 0.072097 | 0.051852 |
| pooled_full_ridge | 0.750 | 0.548148 | 0.009018 | 0.072045 | 0.044444 |
| pooled_environment_centered_ridge | 1.000 | 0.503704 | 0.007969 | 0.052593 | 0.059259 |
| pooled_source_only_ridge | 0.500 | 0.600000 | 0.007464 | 0.054999 | 0.051852 |
| pooled_full_ridge | 0.500 | 0.622222 | 0.007321 | 0.055081 | 0.059259 |
| pooled_environment_centered_ridge | 0.750 | 0.562963 | 0.007131 | 0.047250 | 0.066667 |
| pooled_source_only_ridge | 0.350 | 0.644444 | 0.005656 | 0.039640 | 0.044444 |
| pooled_full_ridge | 0.350 | 0.659259 | 0.005556 | 0.039388 | 0.037037 |

## Interpretation
This is an eval-oracle capacity diagnostic after F016 selected floor fallback. It asks whether a row-level safe blend exists in principle if an ideal risk gate were available. Because the mask uses held-out scoring labels, this result cannot promote a candidate and cannot be used as a calibration rule.
