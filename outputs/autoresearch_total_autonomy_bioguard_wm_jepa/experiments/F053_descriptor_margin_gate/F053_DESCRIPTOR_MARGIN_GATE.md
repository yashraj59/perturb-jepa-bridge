# F053 Descriptor Margin Gate

## Decision
`F053_DESCRIPTOR_MARGIN_GATE_ZERO_FALLBACK_INSUFFICIENT_TRAIN_DIAGNOSTICS`

## Purpose
F052 showed the F051 residual failed through one near-tie retrieval margin. F053 applies a stricter train-only certificate: nonzero residuals require continuous floor preservation, no train retrieval breaks, nonnegative train margin movement, and explicit lower-tail near-tie diagnostics. Otherwise the exact floor is used.

## Selection
- selection label: `F053_ZERO_FALLBACK_INSUFFICIENT_TRAIN_MARGIN_CERTIFICATE`
- selected residual scale: `0.000000`
- certified nonzero scales: `0`
- missing lower-tail train diagnostics: `['margin_q10_change', 'near_tie_count', 'near_tie_erosions']`
- selection reason: No nonzero scale had complete train-only near-tie lower-tail diagnostics plus nonnegative continuous and retrieval-margin evidence.

## Floor-Preserving Metrics
- selected transition: `0.498373`
- floor transition: `0.498373`
- selected delta cosine: `0.897937`
- floor delta cosine: `0.897937`
- selected recall@1: `0.851852`
- floor recall@1: `0.851852`
- floor gap transition: `0.000000`
- floor gap delta cosine: `0.000000`
- floor gap recall: `0.000000`

## Candidate Scale Certificates
```tsv
scale	seed_count	all_train_safe	min_gap_transition	min_gap_delta_cosine	min_gap_recall	total_train_broken_count	min_train_mean_margin_change	has_required_lower_tail_diagnostics	certified
0.0	3	True	0.0	0.0	0.0	0.0	0.0	False	False
0.025	3	True	0.0008510207842257	0.0010460012109348	0.0	0.0	0.0	False	False
0.05	3	True	0.0016782428202257	0.002064843219912	0.0	1.0	0.0	False	False

```

## Interpretation
F053 repairs the safety contract by refusing to spend residual scale when the train-only logs cannot certify near-tie retrieval stability. This preserves the floor exactly, but it is not a scientific win because it also discards the nonzero F051 residual.

## Next Recommendation
Rerun the descriptor-aligned JEPA only with pre-logged train near-tie lower-tail diagnostics and a smaller scale grid; do not use F051 held-out row to choose the scale.

## Promotion Status
No model is promoted. F053 is a calibration diagnostic and floor fallback.
