# Phase 7 Reproduction

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Scope: BSG000 reproduction only; no Phase 7 residual selection yet.

## Full-Ridge Floor

- transition improvement: `0.0057`
- delta cosine: `0.3980`
- recall@1: `0.4815`
- delta rank: `10.2835`

## Zero Residual Wrapper

- max absolute wrapper drift from floor: `0.00000000`

## Old BSJ004-Style Residual

- transition improvement: `0.0066`
- delta cosine: `0.4112`
- recall@1: `0.4074`
- delta rank: `10.5403`
- residual cap hit fraction: `1.0000`

## Decision

`BSG000_PASS_REPRODUCED_PHASE6`
