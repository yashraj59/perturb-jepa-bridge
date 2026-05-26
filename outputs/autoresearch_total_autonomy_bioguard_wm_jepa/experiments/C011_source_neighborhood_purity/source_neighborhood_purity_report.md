# C011 Source-Neighborhood Purity Audit

## Decision
`C011_SOURCE_LATENT_NEIGHBORHOOD_IMPURITY_EXPLAINS_PROXY_FAILURE`

## Evidence
- Current top-third source-neighborhood same-cell-line purity: `0.715682`.
- Current top-third source-neighborhood same-cell-line coverage: `0.715682`.
- Exact source-state nested recall@1: `0.480159`.
- Source-latent neighborhood nested recall@1: `0.398148`.
- Exact-minus-source-latent recall gap: `0.082011`.

## Interpretation
The source-latent proxy is useful but does not reproduce exact source-state constrained retrieval. This audit determines whether the gap is explained by impure source latent neighborhoods rather than by the delta retrieval objective itself.
