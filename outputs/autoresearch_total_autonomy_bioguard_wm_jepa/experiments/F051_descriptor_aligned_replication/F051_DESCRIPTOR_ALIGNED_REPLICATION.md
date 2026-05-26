# F051 Descriptor-Aligned Replication

## Decision
`F051_DESCRIPTOR_ALIGNED_ACTION_CONTRACT_HELDOUT_BELOW_FLOOR`

## Purpose
F050 showed the active protected floor is source/covariate dominated and should not drive another residual-head search. F051 returns to the descriptor-aligned biological action-target contract approved in F026 and replicates the F042 non-exact program-action JEPA diagnostic on fresh seeds.

## Selection Rule
`F042 pre-registered tiny-cap rule: choose best train-continuous-safe scale from {0, 0.025, 0.05}; held-out rows scoring-only.`

## Mean Metrics
- selected residual scale mean: `0.050000`
- selected nonzero fraction: `1.000000`
- heldout local floor preserved: `False`
- active gates pass: `True`
- selected transition: `0.498365`
- floor transition: `0.498373`
- selected delta cosine: `0.898238`
- floor delta cosine: `0.897937`
- selected recall@1: `0.839506`
- floor recall@1: `0.851852`
- selected floor gap transition: `-0.000008` (min `-0.000373`)
- selected floor gap delta cosine: `0.000301` (min `-0.000120`)
- selected floor gap recall: `-0.012346` (min `-0.037037`)
- heldout broken rows: `1`

## Seed Rows
```tsv
seed	selected_transition_improvement	selected_delta_cosine	selected_recall_at_1	selected_delta_rank	selected_magnitude_ratio	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_delta_rank	direct_transition_improvement	direct_delta_cosine	direct_recall_at_1	direct_delta_rank	selected_gap_transition	selected_gap_delta_cosine	selected_gap_recall	selected_action_negative_gap	rna_to_image_recall_at_1	image_to_rna_recall_at_1	target_effective_rank	image_target_effective_rank	floor_init_error	identity_violation	leakage_flag	train_small_cap_selected_scale	train_small_cap_floor_transition	train_small_cap_floor_delta_cosine	train_small_cap_floor_recall_at_1	train_small_cap_selected_transition	train_small_cap_selected_delta_cosine	train_small_cap_selected_recall_at_1	train_small_cap_selected_gap_transition	train_small_cap_selected_gap_delta_cosine	train_small_cap_selected_gap_recall	train_small_cap_selected_train_broken_count_diagnostic_only	train_small_cap_selected_train_mean_margin_change_diagnostic_only	retrieval_row_count	retrieval_broken_count	retrieval_repaired_count	retrieval_preserved_correct_count	retrieval_still_wrong_count	retrieval_broken_near_tie_fraction	retrieval_broken_mean_floor_margin	retrieval_broken_mean_selected_margin	retrieval_broken_mean_margin_change	retrieval_broken_mean_transition_gain_change	retrieval_broken_mean_delta_cosine_change	retrieval_mean_floor_margin	retrieval_mean_selected_margin	retrieval_mean_margin_change	retrieval_mean_rank_change	retrieval_mean_residual_norm	retrieval_selected_median_rank	retrieval_floor_median_rank	selected_residual_scale
10	0.44706717073352825	0.8730262897090926	0.7037037037037037	6.695430032498728	1.081785291358679	0.4469103177162808	0.8728070115030019	0.7037037037037037	6.461749231903911	0.43643896623972417	0.8607322738319985	0.5555555555555556	9.04214296154892	0.00015685301724743805	0.0002192782060906895	0.0	0.022559506379201566	0.7037037037037037	0.7037037037037037	8.485393849699653	7.267879006787531	5.589253340154698e-07	0.0	0.0	0.05	0.5922423185021983	0.9474093185173393	0.5	0.5941465663716315	0.9504686861416989	0.5	0.0019042478694332177	0.0030593676243595436	0.0	1.0	3.919511670487683e-05	27.0	0.0	0.0	19.0	8.0	0.0	0.0	0.0	0.0	0.0	0.0	0.08714160365190943	0.08656703967946172	-0.00057456397244771	0.0	0.05525759488181291	1.0	1.0	0.05
11	0.6020459388556995	0.8939768631917322	0.8888888888888888	7.816207179033211	1.0300871529943667	0.6024186857949821	0.8940968252118728	0.8888888888888888	7.483875381671284	0.5787818786470841	0.8744259219071394	0.8518518518518519	11.123159348233258	-0.00037274693928268654	-0.00011996202014064039	0.0	0.03957157328026528	0.7407407407407407	0.7777777777777778	10.29325091855501	10.312921280916642	5.762042052381844e-07	0.0	0.0	0.05	0.6611106143937406	0.9538829046617275	0.5	0.6633331730028625	0.9559477478816396	0.5	0.002222558609121905	0.0020648432199120714	0.0	0.0	0.0	27.0	0.0	0.0	24.0	3.0	0.0	0.0	0.0	0.0	0.0	0.0	0.12214131218614009	0.12049647895271372	-0.0016448332334263543	0.0	0.051285167417687696	1.0	1.0	0.05
12	0.44598148101869844	0.9277117470781816	0.9259259259259259	7.443680483021622	1.029564402764195	0.44578978509615197	0.9269080989865179	0.9629629629629629	7.165357691487652	0.4358870281403164	0.9259721246697088	0.8888888888888888	10.186238905672658	0.00019169592254647538	0.0008036480916636668	-0.03703703703703698	0.021263049432229786	0.6666666666666666	0.8888888888888888	8.461674988184464	8.669737495678774	7.021583749988736e-07	0.0	0.0	0.05	0.5211663418159606	0.9547687881481026	0.5	0.5228445846361863	0.9568997573286288	0.5	0.0016782428202257726	0.002130969180526132	0.0	0.0	0.00017250489509966184	27.0	1.0	0.0	25.0	1.0	1.0	0.0010862630066510448	-0.0012033419149561908	-0.0022896049216072356	0.0006275553533992007	0.0005097716740690705	0.06526367183624476	0.06546347039712988	0.00019979856088512025	0.037037037037037035	0.049047910293445844	1.0	1.0	0.05

```

## Train Selection Grid
```tsv
seed	scale	safe	transition_improvement	delta_cosine	recall_at_1	gap_transition	gap_delta_cosine	gap_recall	train_broken_count_diagnostic_only	train_mean_margin_change_diagnostic_only
10	0.0	True	0.5922423185021983	0.9474093185173393	0.5	0.0	0.0	0.0	0.0	0.0
10	0.025	True	0.5932097748102613	0.9489587649251554	0.5	0.0009674563080629417	0.0015494464078160686	0.0	0.0	1.96714989253089e-05
10	0.05	True	0.5941465663716315	0.9504686861416989	0.5	0.0019042478694332177	0.0030593676243595436	0.0	1.0	3.919511670487683e-05
11	0.0	True	0.6611106143937406	0.9538829046617275	0.5	0.0	0.0	0.0	0.0	0.0
11	0.025	True	0.6622385618177709	0.9549289058726624	0.5	0.0011279474240302179	0.0010460012109348993	0.0	0.0	0.0
11	0.05	True	0.6633331730028625	0.9559477478816396	0.5	0.002222558609121905	0.0020648432199120714	0.0	0.0	0.0
12	0.0	True	0.5211663418159606	0.9547687881481026	0.5	0.0	0.0	0.0	0.0	0.0
12	0.025	True	0.5220173626001863	0.9558480625321342	0.5	0.00085102078422572	0.0010792743840315477	0.0	0.0	8.608000252684674e-05
12	0.05	True	0.5228445846361863	0.9568997573286288	0.5	0.0016782428202257726	0.002130969180526132	0.0	0.0	0.00017250489509966184

```

## Held-Out Broken Rows
```tsv
seed	query_index	condition_key	perturbation_id	cell_line	selected_residual_scale	flip_type	source_top1_correct	floor_top1_correct	selected_top1_correct	floor_rank	selected_rank	rank_change_selected_minus_floor	source_margin	floor_margin	selected_margin	margin_change_selected_minus_floor	floor_correct_score	floor_best_wrong_score	selected_correct_score	selected_best_wrong_score	floor_top1_label	selected_top1_label	same_top1_label_as_floor	floor_transition_gain	selected_transition_gain	transition_gain_change_selected_minus_floor	floor_delta_cosine	selected_delta_cosine	delta_cosine_change_selected_minus_floor	residual_norm	floor_near_tie_0p02	selected_near_tie_0p02	floor_near_tie_0p05	selected_near_tie_0p05
12	21	pert=9|dose=1|cell=1	9	cell_line_1	0.05	broken_by_residual	False	True	False	1.0	2.0	1.0	-0.7820880262277308	0.0010862630066510448	-0.0012033419149561908	-0.0022896049216072356	0.9198508703888644	0.9187646073822133	0.9191792211867262	0.9203825631016824	pert=9|dose=1|cell=1	pert=11|dose=1|cell=0	False	1.1941304497981562	1.1947580051515554	0.0006275553533992007	0.9207027303944283	0.9212125020684974	0.0005097716740690705	0.05510595054521528	True	True	True	True

```

## Promotion Status
No model is promoted. This is a fresh-seed diagnostic validation branch for the descriptor-aligned action-target contract.
