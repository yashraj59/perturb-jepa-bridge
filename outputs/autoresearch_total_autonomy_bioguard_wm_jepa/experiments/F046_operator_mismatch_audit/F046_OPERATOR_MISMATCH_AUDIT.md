# F046 Operator Mismatch Audit

## Decision
`F046_OPERATOR_TRAIN_HELDOUT_MISMATCH_DOMINATES`

## Purpose
F045 selected the tiny residual on all active multiseed latent seeds using train-only metrics, but held-out scoring fell below the protected floor. F046 checks whether this is a systematic train-heldout residual direction mismatch.

## Findings
- selected nonzero fraction: `1.000000`
- any mismatch fraction: `1.000000`
- train-positive / eval-negative transition fraction: `0.800000`
- train-positive / eval-negative delta fraction: `0.800000`
- train-nonnegative / eval-negative recall fraction: `0.200000`
- retrieval-broken seed fraction: `0.200000`
- mean train transition gap: `0.000702`
- mean eval transition gap: `-0.000171`
- mean train delta gap: `0.007193`
- mean eval delta gap: `-0.001023`

## Interpretation
The active latent operator wrapper selected a nonzero tiny residual on all seeds because train metrics were positive, but held-out delta/transition/retrieval frequently moved below the exact floor. This points to train-heldout residual direction mismatch rather than predictor class alone.

## Seed Mismatch Rows
```tsv
cache_dir	cache_status	floor_delta_cosine	floor_delta_rank	floor_init_error	floor_magnitude_ratio	floor_recall_at_1	floor_transition_improvement	identity_violation	leakage_flag	retrieval_broken_count	retrieval_mean_margin_change	retrieval_repaired_count	retrieval_selected_median_rank	seed	selected_action_negative_gap	selected_delta_cosine	selected_delta_rank	selected_gap_delta_cosine	selected_gap_recall	selected_gap_transition	selected_magnitude_ratio	selected_recall_at_1	selected_residual_scale	selected_transition_improvement	train_small_cap_selected_gap_delta_cosine	train_small_cap_selected_gap_recall	train_small_cap_selected_gap_transition	train_small_cap_selected_train_broken_count_diagnostic_only	transition_mismatch	delta_mismatch	recall_mismatch	retrieval_broken_seed	any_mismatch
outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit	current_phase4_cache	0.39796251669473043	10.283476977293052	1.2150936186394734e-07	0.7744371112179363	0.48148148148148145	0.005661906516194076	0.0	0.0	0.0	0.00020021272972092774	0.0	2.0	0	0.007523315925562053	0.39720227262243785	10.316817928409325	-0.0007602440722925818	0.0	-9.575479188705504e-05	0.7779397220308291	0.48148148148148145	0.05	0.005566151724307021	0.005759931821920583	0.01388888888888884	0.0007604525279941016	0.0	True	True	False	False	True
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G001_fresh_synthetic_seed_cache/seed1_latent_cache	loaded	0.5404280277268804	9.937518308977577	1.7799167917331715e-07	0.7933582487536444	0.3333333333333333	0.021756478285427833	0.0	0.0	0.0	-6.987137209073184e-05	1.0	2.0	1	-7.414507676253498e-05	0.5433684708199755	9.96494976532515	0.0029404430930951575	0.037037037037037035	-2.511006554795081e-06	0.798212476637268	0.37037037037037035	0.05	0.02175396727887304	0.005055838763573073	0.02777777777777779	0.0004415181595667922	0.0	True	False	False	False	True
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed2_latent_cache	loaded	0.4018851082029823	9.632914633777684	8.837127465216588e-08	0.5918627929859949	0.2962962962962963	0.009402064646795659	0.0	0.0	0.0	-0.00015282146819768278	0.0	5.0	2	0.004287600061286865	0.40003301639633404	9.64884509944786	-0.0018520918066482484	0.0	-0.00026208066524511034	0.5941102227026462	0.2962962962962963	0.05	0.009139983981550548	0.006780482885493044	0.0	0.0005370029530789508	0.0	True	True	False	False	True
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed3_latent_cache	loaded	0.4200548145191377	10.486444270464679	9.459221469998447e-08	0.8778340372856198	0.25925925925925924	0.0004243423315500173	0.0	0.0	0.0	3.171783712174336e-05	0.0	3.0	3	-0.002135209695620043	0.4197255774601498	10.546916310445019	-0.0003292370589878524	0.0	4.466289690451919e-06	0.8756921560643275	0.25925925925925924	0.05	0.0004288086212404692	0.007456374462903148	0.01388888888888884	0.0004942525462058103	0.0	False	True	False	False	True
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_fresh_cache_replication/seed4_latent_cache	loaded	0.4972452146646043	10.888689749023888	9.158843345541401e-08	0.7407240625696372	0.2962962962962963	0.019162402606056148	0.0	0.0	1.0	-0.0005211891152532041	0.0	2.0	4	0.006845377633614433	0.49213313643351536	10.903721698447217	-0.005112078231088946	-0.037037037037037035	-0.0004992156554017094	0.7462547585948256	0.25925925925925924	0.05	0.01866318695065444	0.010913107675538436	0.01388888888888884	0.0012757593128367647	1.0	True	True	True	True	True

```

## Promotion Status
No model is promoted. The result is a diagnostic pivot for target/representation redesign.
