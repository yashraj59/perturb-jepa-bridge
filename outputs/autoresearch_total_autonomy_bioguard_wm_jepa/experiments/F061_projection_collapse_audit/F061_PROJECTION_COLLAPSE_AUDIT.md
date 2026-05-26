# F061 Projection Collapse Audit

## Decision
`F061_PROJECTION_KILLS_DELTA_WITHOUT_STABILIZING_TRANSITION`

## Purpose
F060 tested whether positive floor-direction projection could stabilize transition-source cosine while preserving the row-wise residual signal. F061 closes or reopens that branch using only prior artifacts.

## Key Comparisons
- F058 unprojected transition gap: `0.000086`
- F058 unprojected min transition gap: `-0.000499`
- F058 unprojected delta gap: `0.001408`
- F058 held-out retrieval breaks: `0`
- F060 projected transition gap: `0.000022`
- F060 projected min transition gap: `-0.000020`
- F060 projected delta gap: `0.000000`
- F060 negative transition seeds: `2`
- F060 held-out retrieval breaks: `0`
- F059 train/held-out transition-gap correlation: `-0.885610`
- F059 train/held-out delta-gap correlation: `0.948223`

## Interpretation
Pure positive floor-direction projection is too restrictive: it removes the delta-cosine improvement that made row-wise abstention promising, while still failing to give a nonnegative transition gap on every seed.

## Next Recommendation
Do not continue pure floor-direction projection. The next candidate should either run an oracle diagnostic over a residual cone between unprojected and projected residuals, or pivot back to representation/data contract work if the oracle cone lacks capacity.

## Promotion Status
No model is promoted. Pure floor-direction projection is not continued as a promoted candidate.
