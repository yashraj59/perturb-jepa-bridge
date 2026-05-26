# F044 Action-AdaLN Failure Audit

## Decision
`F044_ACTION_ADALN_FAILURE_LOCALIZED_TO_UNSTABLE_NO_ACTION_SIGNAL`

## Purpose
F043 applied the F041/F042 tiny-cap idea to the named Action-AdaLN + RoPE BioGuard-WM frozen-latent path. It selected one nonzero residual but fell below the exact protected floor. F044 localizes whether the failure is calibration noise, action-signal absence, or a reporting issue.

## Findings
- nonzero seed count: `1`
- below-floor nonzero seed count: `1`
- strict LCB selector would select nonzero count: `0`
- action-negative zero fraction: `1.000000`
- mean floor gap transition: `-0.000019`
- min floor gap transition: `-0.000094`
- mean floor gap delta cosine: `-0.000112`
- min floor gap delta cosine: `-0.000562`

## Interpretation
The tiny-cap operator signal does not transfer to the Action-AdaLN/RoPE residual path because train crossfit evidence is marginal, action-negative separation is zero, and the only nonzero seed regresses exact held-out floor transition/delta.

## Below-Floor Nonzero Rows
```tsv
action_negative_gap	artifact_dir	cache_dir	cv_lcb_delta_cosine_gap	cv_lcb_recall_gap	cv_lcb_transition_gap	decision_label	delta_cosine	delta_rank	floor_delta_cosine	floor_gap_delta_cosine	floor_gap_recall	floor_gap_transition	floor_recall_at_1	floor_transition_improvement	magnitude_ratio	mean_recall_gap	mean_transition_gap	recall_at_1	residual_scale	seed	selection_status	transition_improvement
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed3	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed3_latent_cache	-0.0002061220895775066	0.0	-3.8043931238201306e-05	BGWM_SMALL_CAP_KEEP_TINY_ACTION_ADALN_RESIDUAL	0.4194926829006113	10.486444270464679	0.4200548145191377	-0.0005621316185263625	0.0	-9.427125863575658e-05	0.25925925925925924	0.0004243423315500173	0.879302078152805	0.0	1.108782673891074e-06	0.25925925925925924	0.05	3	CALIBRATION_SELECTED_SMALL_CAP_CONTINUOUS_RESIDUAL	0.0003300710729142607

```

## F043 Seed Rows
```tsv
action_negative_gap	artifact_dir	cache_dir	cv_lcb_delta_cosine_gap	cv_lcb_recall_gap	cv_lcb_transition_gap	decision_label	delta_cosine	delta_rank	floor_delta_cosine	floor_gap_delta_cosine	floor_gap_recall	floor_gap_transition	floor_recall_at_1	floor_transition_improvement	magnitude_ratio	mean_recall_gap	mean_transition_gap	recall_at_1	residual_scale	seed	selection_status	transition_improvement
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed0	outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit	0.0	0.0	0.0	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	0.39796251669473043	10.283476977293052	0.39796251669473043	0.0	0.0	0.0	0.48148148148148145	0.005661906516194076	0.7744371112179363	0.0	0.0	0.48148148148148145	0.0	0	CALIBRATION_DEFAULT_TO_FLOOR	0.005661906516194076
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed1	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G001_fresh_synthetic_seed_cache/seed1_latent_cache	0.0	0.0	0.0	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	0.5404280277268804	9.937518308977577	0.5404280277268804	0.0	0.0	0.0	0.3333333333333333	0.021756478285427833	0.7933582487536444	0.0	0.0	0.3333333333333333	0.0	1	CALIBRATION_DEFAULT_TO_FLOOR	0.021756478285427833
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed2	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed2_latent_cache	0.0	0.0	0.0	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	0.4018851082029823	9.632914633777684	0.4018851082029823	0.0	0.0	0.0	0.2962962962962963	0.009402064646795659	0.5918627929859949	0.0	0.0	0.2962962962962963	0.0	2	CALIBRATION_DEFAULT_TO_FLOOR	0.009402064646795659
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed3	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_source_state_stability/seed3_latent_cache	-0.0002061220895775066	0.0	-3.8043931238201306e-05	BGWM_SMALL_CAP_KEEP_TINY_ACTION_ADALN_RESIDUAL	0.4194926829006113	10.486444270464679	0.4200548145191377	-0.0005621316185263625	0.0	-9.427125863575658e-05	0.25925925925925924	0.0004243423315500173	0.879302078152805	0.0	1.108782673891074e-06	0.25925925925925924	0.05	3	CALIBRATION_SELECTED_SMALL_CAP_CONTINUOUS_RESIDUAL	0.0003300710729142607
0.0	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F043_action_adaln_small_cap/seed4	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/G002_multi_seed_fresh_cache_replication/seed4_latent_cache	0.0	0.0	0.0	BGWM_SMALL_CAP_ZERO_FLOOR_FALLBACK	0.4972452146646043	10.888689749023888	0.4972452146646043	0.0	0.0	0.0	0.2962962962962963	0.019162402606056148	0.7407240625696372	0.0	0.0	0.2962962962962963	0.0	4	CALIBRATION_DEFAULT_TO_FLOOR	0.019162402606056148

```

## Promotion Status
No model is promoted. The protected floor and protected PLS model of record remain active.
