# F012 Benchmark Redesign Brief

## Decision
`F012_BENCHMARK_REDESIGN_REQUIRED_BEFORE_MORE_ARCHITECTURE`

## Locked Current Benchmark Facts

The current protected model of record remains the protected rank-3 train-split-only PLS raw-linear readout. The current protected floor values remain locked for the current benchmark:

| metric | locked seed0 value | fresh-seed evidence | interpretation |
|---|---:|---:|---|
| condition recall@1 | 0.481481 | mean 0.296296, max 0.333333 | seed0 outlier / unstable retrieval gate |
| transition improvement | 0.005662 | mean 0.012686 | fresh seeds are not worse |
| delta cosine | 0.397963 | mean 0.464903 | fresh seeds are stronger |
| effective rank | 10.283477 | fresh mean 10.236392 | broadly stable |

G002 also found the deployable online-source-neighborhood proxy has mean recall `0.425926`, minimum `0.383598`, and floor fraction `0.000` across fresh seeds. This keeps source-state preservation training cooled.

## Why This Blocks More Architecture Search

The current recall gate is not behaving like a stable perturbation-prediction target across synthetic seeds. The architecture loop has repeatedly improved or preserved continuous geometry while failing a seed0-specific retrieval threshold. Continuing to optimize architecture against this gate risks fitting a benchmark artifact, not learning a better biological world model.

## Redesign Requirements

A future benchmark must be created under a new explicit name and must not mutate old results in place. Minimum requirements:

1. Multi-seed baseline registry with at least seeds 0-4 before any architecture run.
2. Separate primary gates for continuous transition geometry and retrieval, instead of one seed0 recall threshold dominating decisions.
3. Retain condition recall@1 as a secondary/no-regression metric until it is proven stable.
4. Report delta cosine, transition source improvement, effective rank, magnitude ratio, median rank, and condition/perturbation/cell-line retrieval together.
5. Include source-as-target, action-ridge floor, mean-delta, and seed-randomized nulls.
6. Preserve leakage controls: no eval target rows for fitting, whitening, calibration, residual selection, or model choice.
7. If split semantics change, use a new benchmark name rather than changing `test_heldout_perturbation` in place.

## Proposed Next Benchmark Name

`synthetic_genetic_anchor_lite_multiseed_v1`

This benchmark should have a Step 0 registry before any JEPA candidate is rerun. The existing benchmark remains a historical stress test, not a target to silently relax.

## Recommended Next Action

Write and run a Step 0 benchmark-registry builder for `synthetic_genetic_anchor_lite_multiseed_v1`, using the existing latent caches for seeds 0-4 where possible. Do not launch architecture until the new registry is complete and documented.
