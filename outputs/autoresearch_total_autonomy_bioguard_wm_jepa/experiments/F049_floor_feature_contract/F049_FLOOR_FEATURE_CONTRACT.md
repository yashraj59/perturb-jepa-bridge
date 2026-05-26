# F049 Floor Feature Contract

## Decision
`F049_FLOOR_FEATURE_CONTRACT_AMBIGUOUS`

## Purpose
F048 showed that removing exact perturbation one-hot action columns makes the candidate-legal program-action floor fall below the protected full floor. F049 decomposes the protected full-ridge prediction to determine whether train exact-action columns provide a centering/extrapolation contribution on held-out perturbations.

## Key Metrics
- mean full transition: `0.011281`
- mean full delta cosine: `0.451515`
- mean full recall@1: `0.333333`
- mean no-train-exact transition: `0.011281`
- mean no-train-exact delta cosine: `0.451515`
- mean no-train-exact recall@1: `0.333333`
- no-train-exact transition gap: `0.000000`
- no-train-exact delta gap: `0.000000`
- no-train-exact recall gap: `0.000000`
- exact train-action contribution norm fraction: `0.000000`
- program contribution norm fraction: `0.337415`
- exact train-action contribution delta cosine: `-0.000000`
- eval unseen exact-action contribution norm: `0.000000`

## Interpretation
This is a diagnostic-only audit of the protected floor. It does not replace the protected model of record or protected floor, but it documents whether the protected floor depends on feature behavior that candidate JEPA paths are forbidden to use as their main representation/action path.

## Seed Rows
```tsv
seed	cache_status	cache_dir	train_supported_exact_action_cols	eval_active_unseen_exact_action_cols	program_action_cols	full_transition_improvement	full_delta_cosine	full_recall_at_1	full_delta_rank	full_magnitude_ratio	no_train_exact_transition_improvement	no_train_exact_delta_cosine	no_train_exact_recall_at_1	no_program_transition_improvement	no_program_delta_cosine	no_program_recall_at_1	no_action_transition_improvement	no_action_delta_cosine	no_action_recall_at_1	no_train_exact_gap_transition	no_train_exact_gap_delta_cosine	no_train_exact_gap_recall	exact_train_contrib_norm_mean	program_contrib_norm_mean	full_delta_norm_mean	true_delta_norm_mean	exact_train_contrib_norm_fraction	program_contrib_norm_fraction	exact_train_contrib_delta_cosine_mean	program_contrib_delta_cosine_mean	exact_train_contrib_full_delta_cosine_mean	program_contrib_full_delta_cosine_mean	eval_unseen_exact_contrib_norm_mean	identity_violation	leakage_flag
0	current_phase4_cache	outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit	1,2,3,4,5,6,7,8	9,10,11	12,13,14,15,16,17,18,19	0.005661906516194076	0.39796251669473043	0.48148148148148145	10.283476977293052	0.7744371112179363	0.005661906516210322	0.3979625166948039	0.48148148148148145	0.01048805160234058	0.41175162103907637	0.25925925925925924	0.010488051602355426	0.41175162103913415	0.25925925925925924	1.6246552714260787e-14	7.349676423018536e-14	0.0	1.9427841321876237e-13	0.10985610536305034	0.30997363358887636	0.4251684870814574	6.267578663688452e-13	0.3544046765885723	1.6078427998065433e-07	0.03800703527723833	1.5010998708362492e-06	0.41149345078451716	0.0	0.0	0.0
1	loaded	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G001_fresh_synthetic_seed_cache/seed1_latent_cache	1,2,3,4,5,6,7,8	9,10,11	12,13,14,15,16,17,18,19	0.021756478285427833	0.5404280277268804	0.3333333333333333	9.937518308977577	0.7933582487536444	0.02175647828542761	0.5404280277268888	0.3333333333333333	0.023096262158386924	0.5213083833816944	0.3333333333333333	0.02309626215838794	0.521308383381721	0.3333333333333333	-2.220446049250313e-16	8.43769498715119e-15	0.0	1.0434917017904062e-13	0.10372685086097196	0.2530448015012562	0.34597669066648745	4.12374289295655e-13	0.4099149646449347	-1.8994349029984294e-07	0.09290064831891914	-2.20507411640366e-07	0.38067344610761933	0.0	0.0	0.0
2	loaded	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed2_latent_cache	1,2,3,4,5,6,7,8	9,10,11	12,13,14,15,16,17,18,19	0.009402064646795659	0.4018851082029823	0.2962962962962963	9.632914633777684	0.5918627929859949	0.00940206464679889	0.4018851082030364	0.2962962962962963	0.00458376379057947	0.3279908686212952	0.3333333333333333	0.004583763790582747	0.3279908686213446	0.3333333333333333	3.2317898357447916e-15	5.412337245047638e-14	0.0	1.0640638218156761e-13	0.05473439664459727	0.1972943272972581	0.34214753396737857	5.39328138012037e-13	0.2774250906977696	-2.3519421522858475e-07	0.2963900771315147	1.1148718777544621e-07	0.25659068179229927	0.0	0.0	0.0
3	loaded	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed3_latent_cache	1,2,3,4,5,6,7,8	9,10,11	12,13,14,15,16,17,18,19	0.0004243423315500173	0.4200548145191377	0.25925925925925924	10.486444270464679	0.8778340372856198	0.0004243423315436396	0.4200548145190885	0.25925925925925924	0.005174837353515233	0.4437640242265373	0.2962962962962963	0.005174837353508659	0.4437640242264746	0.2962962962962963	-6.377656649320107e-15	-4.9182879990894435e-14	0.0	1.2308842062960182e-13	0.08851913826357055	0.24406570261041677	0.30882456057482865	5.043249391991728e-13	0.3626856920772142	3.8380679248356847e-07	0.12920914194751848	-2.2949210454081964e-07	0.5339947087425215	0.0	0.0	0.0
4	loaded	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_fresh_cache_replication/seed4_latent_cache	1,2,3,4,5,6,7,8	9,10,11	12,13,14,15,16,17,18,19	0.019162402606056148	0.4972452146646043	0.2962962962962963	10.888689749023888	0.7407240625696372	0.019162402606050708	0.4972452146645916	0.2962962962962963	0.017161424718914842	0.5079634747680908	0.2222222222222222	0.017161424718909916	0.5079634747680729	0.2222222222222222	-5.440092820663267e-15	-1.2712053631958042e-14	0.0	1.6738918821424985e-13	0.07614246907700721	0.26939290920037023	0.3865740804977513	6.21357068050178e-13	0.2826446668660222	-1.6043257506373252e-07	0.04909133648538484	-7.032748701678516e-07	0.2496393947211983	0.0	0.0	0.0

```

## Promotion Status
No model is promoted.
