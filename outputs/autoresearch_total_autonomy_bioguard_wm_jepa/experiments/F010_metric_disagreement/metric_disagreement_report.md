# F010 Metric Disagreement Audit

## Decision
`F010_RECALL_GATE_SEED_UNSTABLE_CONTINUOUS_METRICS_DISAGREE`

## Evidence
- Current seed composite gain at weight 0.1: `0.101190` condition recall@1.
- Fresh seed composite gain at weight 0.5: `0.095238` condition recall@1.
- Fresh seed minus protected transition improvement gap: `0.016095`.
- Fresh seed minus protected delta-cosine gap: `0.142466`.
- Fresh seed minus protected recall@1 gap: `-0.148148`.
- Train-triplet transition/recall correlation: `0.343991`.
- Train-triplet delta/recall correlation: `0.218925`.

## Interpretation
The endpoint+delta composite repair replicated on a fresh synthetic seed, but recall@1 remains below the protected floor. The fresh seed simultaneously improves continuous transition and delta-cosine metrics while lowering recall@1, so recall is not a stable proxy for transition quality under the current latent/gallery geometry.

## Constraint
This does not change promotion gates. The protected model of record and protected full-ridge floor remain active.

## Next Technical Direction
Do not reopen a heavy JEPA candidate yet. The next low-compute step should design a perturbation-centered retrieval objective and score it against both continuous transition metrics and recall, with fixed train/eval contracts and no eval-target selection.
