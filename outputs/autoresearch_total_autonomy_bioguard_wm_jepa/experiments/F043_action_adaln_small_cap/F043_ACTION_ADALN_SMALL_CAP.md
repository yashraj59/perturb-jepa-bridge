# F043 Action-AdaLN Small-Cap Calibration

## Decision
`F043_ACTION_ADALN_SMALL_CAP_HELDOUT_BELOW_FLOOR`

## Purpose
F042 validated a tiny residual cap in the operator-style path on fresh synthetic seeds. F043 tests whether the same calibration idea works in the named BioGuard-WM Action-AdaLN + RoPE frozen-latent path.

## Selection Rule
`Action-AdaLN + RoPE frozen-latent residual with small_cap_continuous calibration; candidate scales <= 0.05 selected from train-only crossfit continuous metrics.`

## Mean Metrics
- nonzero residual seeds: `1` / `5`
- selected residual scale mean: `0.010000`
- active gates pass: `True`
- heldout local floor preserved: `False`
- selected transition: `0.011263`
- floor transition: `0.011281`
- selected delta cosine: `0.451403`
- floor delta cosine: `0.451515`
- selected recall@1: `0.333333`
- floor recall@1: `0.333333`
- selected floor gap transition: `-0.000019` (min `-0.000094`)
- selected floor gap delta cosine: `-0.000112` (min `-0.000562`)
- selected floor gap recall: `0.000000` (min `0.000000`)

## Seed Rows
```tsv
seed	cache_dir	decision_label	selection_status	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_gap_transition	floor_gap_delta_cosine	floor_gap_recall	residual_scale	cv_lcb_transition_gap	cv_lcb_delta_cosine_gap	cv_lcb_recall_gap	mean_transition_gap	mean_recall_gap	action_negative_gap	artifact_dir
0	outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	CALIBRATION_DEFAULT_TO_FLOOR	0.005661906516194076	0.39796251669473043	0.48148148148148145	10.283476977293052	0.7744371112179363	0.005661906516194076	0.39796251669473043	0.48148148148148145	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed0
1	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G001_fresh_synthetic_seed_cache/seed1_latent_cache	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	CALIBRATION_DEFAULT_TO_FLOOR	0.021756478285427833	0.5404280277268804	0.3333333333333333	9.937518308977577	0.7933582487536444	0.021756478285427833	0.5404280277268804	0.3333333333333333	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed1
2	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed2_latent_cache	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	CALIBRATION_DEFAULT_TO_FLOOR	0.009402064646795659	0.4018851082029823	0.2962962962962963	9.632914633777684	0.5918627929859949	0.009402064646795659	0.4018851082029823	0.2962962962962963	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed2
3	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed3_latent_cache	BGWM_SMALL_CAP_KEEP_TINY_ACTION_ADALN_RESIDUAL	CALIBRATION_SELECTED_SMALL_CAP_CONTINUOUS_RESIDUAL	0.0003300710729142607	0.4194926829006113	0.25925925925925924	10.486444270464679	0.879302078152805	0.0004243423315500173	0.4200548145191377	0.25925925925925924	-9.427125863575658e-05	-0.0005621316185263625	0.0	0.05	-3.8043931238201306e-05	-0.0002061220895775066	0.0	1.108782673891074e-06	0.0	0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed3
4	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_fresh_cache_replication/seed4_latent_cache	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	CALIBRATION_DEFAULT_TO_FLOOR	0.019162402606056148	0.4972452146646043	0.2962962962962963	10.888689749023888	0.7407240625696372	0.019162402606056148	0.4972452146646043	0.2962962962962963	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed4

```

## Promotion Status
No model is promoted. F043 is a Tier 1 frozen-latent diagnostic of the Action-AdaLN + RoPE path.
