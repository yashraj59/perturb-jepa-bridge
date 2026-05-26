# F052 Descriptor Near-Tie Retrieval Audit

## Decision
`F052_DESCRIPTOR_FAILURE_IS_NEAR_TIE_RETRIEVAL_MARGIN`

## Purpose
F051 made the descriptor-aligned contract look much healthier than the active source/covariate-dominated floor, but it still failed exact local floor preservation. F052 is a read-only audit that asks whether the failure is a broad transition error or a near-tie retrieval-margin instability.

## Source
- source experiment: `F051_descriptor_aligned_replication`
- audit scope: `Read-only diagnostic over F051 held-out retrieval rows; no training, no eval selection, no split changes.`

## Key Metrics
- held-out rows audited: `81`
- floor-correct rows: `69`
- broken-by-residual rows: `1`
- broken fraction of rows: `0.012346`
- broken fraction of floor-correct rows: `0.014493`
- broken near-tie fraction at 0.02: `1.000000`
- broken near-tie fraction at 0.05: `1.000000`
- broken rows with positive transition and delta change: `1.000000`
- F051 selected transition gap: `-0.000008` (min `-0.000373`)
- F051 selected delta gap: `0.000301` (min `-0.000120`)
- F051 selected recall gap: `-0.012346` (min `-0.037037`)
- mean selected residual scale: `0.050000`

## Broken Rows
```tsv
seed	query_index	condition_key	perturbation_id	cell_line	selected_residual_scale	flip_type	source_top1_correct	floor_top1_correct	selected_top1_correct	floor_rank	selected_rank	rank_change_selected_minus_floor	source_margin	floor_margin	selected_margin	margin_change_selected_minus_floor	floor_correct_score	floor_best_wrong_score	selected_correct_score	selected_best_wrong_score	floor_top1_label	selected_top1_label	same_top1_label_as_floor	floor_transition_gain	selected_transition_gain	transition_gain_change_selected_minus_floor	floor_delta_cosine	selected_delta_cosine	delta_cosine_change_selected_minus_floor	residual_norm	floor_near_tie_0p02	selected_near_tie_0p02	floor_near_tie_0p05	selected_near_tie_0p05
12	21	pert=9|dose=1|cell=1	9	cell_line_1	0.05	broken_by_residual	False	True	False	1.0	2.0	1.0	-0.7820880262277308	0.001086263006651	-0.0012033419149561	-0.0022896049216072	0.9198508703888644	0.9187646073822132	0.9191792211867262	0.9203825631016824	pert=9|dose=1|cell=1	pert=11|dose=1|cell=0	False	1.1941304497981562	1.1947580051515554	0.0006275553533992	0.9207027303944284	0.9212125020684974	0.000509771674069	0.0551059505452152	True	True	True	True

```

## Interpretation
The descriptor-aligned residual is not failing by broad collapse. The observed F051 failure is a floor-correct held-out query with a tiny floor margin, where the residual slightly improves continuous transition and delta metrics but moves the nearest-neighbor margin across zero.

## Next Recommendation
Implement a train-only retrieval-margin safety gate for the descriptor-aligned contract: residual scale must default to zero unless train-internal near-tie margin diagnostics prove no local floor flip risk.

## Promotion Status
No model is promoted. F052 is a diagnostic/pivot event under continuous Debate Council autonomy.
