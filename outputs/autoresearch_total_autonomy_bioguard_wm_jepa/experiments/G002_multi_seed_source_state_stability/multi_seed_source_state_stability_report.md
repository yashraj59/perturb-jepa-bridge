# G002 Multi-Seed Source-State Stability Audit

## Decision
`G002_MULTI_SEED_SOURCE_STATE_SIGNAL_UNSTABLE_NO_TRAINING_REOPEN`

## Evidence
Protected condition recall floor: `0.481481`.

| seed | floor recall@1 | exact source-state recall@1 | teacher-neighborhood recall@1 | online-neighborhood recall@1 | online floor gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.481481 | 0.480159 | 0.398148 | 0.457672 | -0.023810 |
| 1 | 0.333333 | 0.447090 | 0.414021 | 0.418651 | -0.062831 |
| 2 | 0.296296 | 0.485450 | 0.458995 | 0.463624 | -0.017857 |
| 3 | 0.259259 | 0.396825 | 0.378968 | 0.383598 | -0.097884 |

## Interpretation
C015 did not approve source-state preservation training. G002 tests whether that decision was seed-specific by repeating the same train-only nested source-state and source-neighborhood diagnostics on additional tiny synthetic latent caches. The result is diagnostic only: source-state metadata and condition labels are used only to score retrieval geometry, not as model inputs or as a promotion path.
