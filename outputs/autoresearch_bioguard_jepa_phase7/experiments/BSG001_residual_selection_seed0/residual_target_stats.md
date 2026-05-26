# Residual Target Stats

- teacher delta rank: `13.5627`
- floor delta rank: `11.6312`
- residual target rank: `14.0659`
- residual target magnitude: `0.2143`
- residual target action-probe accuracy: `0.0000`
- residual target batch-probe accuracy: `0.3333`
- residual near-zero fraction: `0.0000`
- residual batch dominated: `False`

## Split Summary

| fold_id | fit_rows | calibration_rows | fit_actions | calibration_actions | fallback_reason |
| ------- | -------- | ---------------- | ----------- | ------------------- | --------------- |
| 0       | 54       | 18               | 1,2,4,5,7,8 | 3,6                 |                 |
| 1       | 54       | 18               | 2,3,4,6,7,8 | 1,5                 |                 |
| 2       | 54       | 18               | 1,3,5,6,7,8 | 2,4                 |                 |
| 3       | 54       | 18               | 1,2,3,4,5,6 | 7,8                 |                 |

## Decision

`BSG001_PASS_RESIDUAL_TARGET_VALID`
