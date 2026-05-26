# BioSpectral-JEPA Phase 6 Final Report

## Decision label
BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR

## Model of record
Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless Tier 3 pass supersedes it.

## What was tested
- Stage A floor reproduction: passed. Full action ridge and analytical low-rank ridge were reproduced from Phase 4 cached train-only teacher latents.
- Stage B rank bottleneck audit: passed reopening. BOJ002 failed due to both an implementation/factorization gap and a real fixed-rank bottleneck.
- BSJ001 neural low-rank equivalence: passed; neural rank-8 reduced-rank ridge matched analytical low-rank ridge.
- BSJ002 floor wrapper: passed; zero residual preserved the full-ridge floor exactly.
- BSJ003 rank ladder: passed; full-ridge expert preserved the floor and router usage was logged.
- BSJ004 spectral residual: failed; train-only residual improved train fit but dropped held-out transition improvement and recall below the full-ridge floor.
- BSJ005 kernel residual: not run because BSJ004 fired a stop condition.
- BSJ006 full BioSpectral-JEPA: not run because BSJ004 fired a stop condition.
- BSJ007 Norman RNA-only: not run because synthetic BioSpectral-JEPA did not pass.

## Key metrics
| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | floor gap | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| BSJ000 full-ridge audit | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | 0.0000 | PHASE6_REOPEN_BIOSPECTRAL_APPROVED |
| BSJ001 neural low-rank | 0.0046 | 0.3877 | 0.4074 | 6.7681 | 0.7585 | -0.0011 | BSJ001_KEEP_NEURAL_LOWRANK_MATCHES_ANALYTIC |
| BSJ002 floor wrapper | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | -0.0000 | BSJ002_KEEP_FLOOR_WRAPPER_MATCHES_FULL_RIDGE |
| BSJ003 rank ladder | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | -0.0000 | BSJ003_KEEP_RANK_LADDER_PRESERVES_FLOOR |
| BSJ004 spectral residual | 0.0050 | 0.4059 | 0.4074 | 10.4574 | 0.8000 | -0.0007 | BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR |

## Main interpretation
BOJ002 was both a bug and a real bottleneck. The Phase 6 audit showed the old BOJ002 factorization did not match analytical low-rank ridge, and the rank sweep showed rank-8 itself loses held-out recall/rank relative to the full-ridge floor.

The new BioSpectral components fixed the exact-equivalence and floor-preservation contracts. The failure moved to the residual branch: the spectral residual learned train residual structure but did not generalize to held-out perturbations. It increased delta cosine and rank, but it harmed transition improvement and recall@1, so it cannot be kept.

## JEPA identity status
No full end-to-end BioSpectral-JEPA candidate was run. Operator probes BSJ001-BSJ004 passed identity/leakage flags, but operator-only probes are not full JEPA promotions.

## Leakage status
No forbidden feature/statistic usage was detected. Each BSJ001-BSJ004 artifact includes `leakage_report.md`; fitting used train rows only, eval rows were scoring-only, `condition_key` was label-only, and no eval target means or pooled train+test target statistics were used.

## Recommendation
Close Phase 6 without promoting any model. The next amendment should redesign the residual mechanism before reopening BSJ004, with a train-only validation split or stronger residual-selection contract so residuals cannot improve train fit while reducing held-out transition/retrieval.
