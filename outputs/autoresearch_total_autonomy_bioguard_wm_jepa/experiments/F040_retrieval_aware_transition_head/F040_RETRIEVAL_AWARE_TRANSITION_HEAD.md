# F040 Retrieval-Aware Transition Head

## Decision
`F040_RETRIEVAL_AWARE_ZERO_FALLBACK`

## Purpose
F039 selected zero residual because post-hoc train-margin gates were too conservative. F040 changes training: the floor-initialized residual head receives a train-only paired-target margin preservation loss before the same train-only scale proxy is applied.

## Selection Rule
`retrieval-aware residual head trained on train-only paired-target margin preservation, then train-only safe-scale proxy chooses deployment scale`

## Mean Metrics
- selected residual scale mean: `0.000000`
- selected nonzero fraction: `0.000000`
- heldout local floor preserved: `True`
- selected transition: `0.333054`
- floor transition: `0.333054`
- selected delta cosine: `0.912222`
- floor delta cosine: `0.912222`
- selected recall@1: `0.859259`
- floor recall@1: `0.859259`
- selected floor gap transition: `0.000000` (min `0.000000`)
- selected floor gap delta cosine: `0.000000` (min `0.000000`)
- selected floor gap recall: `0.000000` (min `0.000000`)
- heldout broken rows: `0`
- train proxy near-tie erosions mean: `0.000000`
- train proxy q10 margin change mean: `0.000000`

## Seed Rows
```tsv
seed	selected_transition_improvement	selected_delta_cosine	selected_recall_at_1	selected_delta_rank	selected_magnitude_ratio	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_delta_rank	direct_transition_improvement	direct_delta_cosine	direct_recall_at_1	direct_delta_rank	selected_gap_transition	selected_gap_delta_cosine	selected_gap_recall	selected_action_negative_gap	rna_to_image_recall_at_1	image_to_rna_recall_at_1	target_effective_rank	image_target_effective_rank	floor_init_error	identity_violation	leakage_flag	train_proxy_selected_scale	train_proxy_floor_transition	train_proxy_floor_delta_cosine	train_proxy_floor_recall_at_1	train_proxy_selected_transition	train_proxy_selected_delta_cosine	train_proxy_selected_recall_at_1	train_proxy_selected_gap_transition	train_proxy_selected_gap_delta_cosine	train_proxy_selected_gap_recall	train_proxy_near_tie_erosions	train_proxy_margin_q10_change	train_proxy_broken_count	retrieval_row_count	retrieval_broken_count	retrieval_repaired_count	retrieval_preserved_correct_count	retrieval_still_wrong_count	retrieval_broken_near_tie_fraction	retrieval_broken_mean_floor_margin	retrieval_broken_mean_selected_margin	retrieval_broken_mean_margin_change	retrieval_broken_mean_transition_gain_change	retrieval_broken_mean_delta_cosine_change	retrieval_mean_floor_margin	retrieval_mean_selected_margin	retrieval_mean_margin_change	retrieval_mean_rank_change	retrieval_mean_residual_norm	retrieval_selected_median_rank	retrieval_floor_median_rank	selected_residual_scale
0	0.14366915588425286	0.9348091301779191	0.8888888888888888	7.653265271940026	1.032848813148596	0.14366915588425286	0.9348091301779191	0.8888888888888888	7.653265271940026	0.14787307436842131	0.951865428515925	0.8518518518518519	10.730915473128652	0.0	0.0	0.0	0.022385561620677466	0.8148148148148148	0.7777777777777778	4.9867390041693875	5.965925413995415	6.146275541851765e-07	0.0	0.0	0.0	0.14541942888185247	0.9490979256830906	0.4583333333333333	0.14541942888185247	0.9490979256830906	0.4583333333333333	0.0	0.0	0.0	0.0	0.0	0.0	27.0	0.0	0.0	24.0	3.0	0.0	0.0	0.0	0.0	0.0	0.0	0.04787862585862564	0.04787862585862564	0.0	0.0	0.0	1.0	1.0	0.0
1	0.1707350373131547	0.8418910341147743	0.5925925925925926	7.628453394528662	0.9980487481514073	0.1707350373131547	0.8418910341147743	0.5925925925925926	7.628453394528662	0.1899567815070071	0.9086305815723855	0.7037037037037037	10.60217926191635	0.0	0.0	0.0	0.029399823212959364	0.6296296296296297	0.8888888888888888	5.943674980225257	8.031676607695829	9.431691827543887e-07	0.0	0.0	0.0	0.23016913206959908	0.9223779129863063	0.4861111111111111	0.23016913206959908	0.9223779129863063	0.4861111111111111	0.0	0.0	0.0	0.0	0.0	0.0	27.0	0.0	0.0	16.0	11.0	0.0	0.0	0.0	0.0	0.0	0.0	-0.0017944427586545578	-0.0017944427586545578	0.0	0.0	0.0	1.0	1.0	0.0
2	0.2724394655216964	0.932261923188985	1.0	6.230242065351273	0.989959751361646	0.2724394655216964	0.932261923188985	1.0	6.230242065351273	0.2732000993428611	0.9447704625011399	1.0	8.325130562717217	0.0	0.0	0.0	0.011421118009169173	0.7037037037037037	0.7037037037037037	5.545190988462882	7.238290478159652	1.1600628440078253e-06	0.0	0.0	0.0	0.31538051651863425	0.9472656765128912	0.5	0.31538051651863425	0.9472656765128912	0.5	0.0	0.0	0.0	0.0	0.0	0.0	27.0	0.0	0.0	27.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.025340978206050177	0.025340978206050177	0.0	0.0	0.0	1.0	1.0	0.0
3	0.5212980055525284	0.9320907531731707	1.0	7.051391799370344	1.0111265488257464	0.5212980055525284	0.9320907531731707	1.0	7.051391799370344	0.5180918218114406	0.9400557462842232	1.0	10.608024035795088	0.0	0.0	0.0	0.06773297839884662	0.8518518518518519	1.0	7.845789275370145	7.65144492648573	6.175035700550779e-07	0.0	0.0	0.0	0.5964334664363728	0.9479116135821665	0.5	0.5964334664363728	0.9479116135821665	0.5	0.0	0.0	0.0	0.0	0.0	0.0	27.0	0.0	0.0	27.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.15187170315005852	0.15187170315005852	0.0	0.0	0.0	1.0	1.0	0.0
4	0.5571293539076213	0.9200586293997617	0.8148148148148148	6.089894692766619	1.0546942332990834	0.5571293539076213	0.9200586293997617	0.8148148148148148	6.089894692766619	0.5671167045169277	0.9474360366162773	0.9629629629629629	9.013227010319271	0.0	0.0	0.0	0.09750004547557162	0.7777777777777778	0.7777777777777778	7.17675460536759	6.623638804543242	4.288532871044026e-07	0.0	0.0	0.0	0.6985857906455684	0.9562826450260533	0.5	0.6985857906455684	0.9562826450260533	0.5	0.0	0.0	0.0	0.0	0.0	0.0	27.0	0.0	0.0	22.0	5.0	0.0	0.0	0.0	0.0	0.0	0.0	0.06913614809181384	0.06913614809181384	0.0	0.0	0.0	1.0	1.0	0.0

```

## Train-Safe Scale Rows
```tsv
seed	scale	safe	transition_improvement	delta_cosine	recall_at_1	gap_transition	gap_delta_cosine	gap_recall	broken_count	near_tie_erosions	margin_q10_change	mean_margin_change	near_tie_count
0	0.0	True	0.14541942888185247	0.9490979256830906	0.4583333333333333	0.0	0.0	0.0	0.0	0.0	0.0	0.0	33.0
1	0.0	True	0.23016913206959908	0.9223779129863063	0.4861111111111111	0.0	0.0	0.0	0.0	0.0	0.0	0.0	35.0
2	0.0	True	0.31538051651863425	0.9472656765128912	0.5	0.0	0.0	0.0	0.0	0.0	0.0	0.0	36.0
3	0.0	True	0.5964334664363728	0.9479116135821665	0.5	0.0	0.0	0.0	0.0	0.0	0.0	0.0	35.0
4	0.0	True	0.6985857906455684	0.9562826450260533	0.5	0.0	0.0	0.0	0.0	0.0	0.0	0.0	36.0

```

## Held-Out Broken Rows
```tsv
No broken-by-residual rows.

```

## Promotion Status
No model is promoted. F040 is still Tier 2 calibration; a safe nonzero result can only justify future Tier 3/no-regression design.
