# F083 Tier 3 Preflight

## Decision
`F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS`

## Purpose
F082 produced a stable non-promoting Tier 2-style pass. F083 checks whether the candidate is actually eligible for Tier 3 under the locked model-of-record and no-regression rules.

## Gate Summary
```tsv
gate	floor	mean	min_or_mean	pass	gap
transition	0.0057	0.20781648143800757	0.12491412882339287	True	0.20211648143800756
delta_cosine	0.398	0.9344029333900138	0.9047365628809659	True	0.5364029333900138
recall_at_1	0.4815	1.0	1.0	True	0.5185
delta_rank	10.2835	7.0239482850671004	6.595168576592012	False	-3.2595517149328996
magnitude_ratio	0.7744	1.009602448752875	1.009602448752875	True	0.235202448752875

```

## Cross-Modal And Validator Gates
- RNA->image recall@1: `0.771605`
- image->RNA recall@1: `0.878601`
- cross-modal gate pass: `True`
- paired external validator available: `False`
- Norman path exists: `True`
- Norman status: `RNA-only diagnostic possible; not a paired scRNA+imaging Tier 3 validator.`

## Candidate Code Lock
`scripts/run_bioguard_wm_total_autonomy.py::_run_delta_calibrated_jepa_seed_grid + perturb_jepa.models.program_bootstrap_jepa.ProgramBootstrapJEPA`

## Required Next Branch
`rank_preserving_delta_calibrator_or_rank_ladder_before_Tier3`

## Summary Row
```tsv
source_experiment	candidate_code_path	protected_model_of_record	transition_floor	delta_cosine_floor	recall_floor	delta_rank_floor	magnitude_ratio_floor	mean_transition	min_transition	mean_delta_cosine	min_delta_cosine	mean_recall_at_1	min_recall_at_1	mean_delta_rank	min_delta_rank	mean_magnitude_ratio	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	max_identity_violation	max_leakage_flag	paired_external_validator_available	norman_rna_only_path_exists	norman_validator_status	model_promoted	transition_floor_pass	delta_floor_pass	recall_floor_pass	rank_floor_pass	magnitude_floor_pass	cross_modal_gate_pass	transition_floor_gap	delta_floor_gap	recall_floor_gap	rank_floor_gap	magnitude_floor_gap	required_next_branch	diagnostic_label
F082	scripts/run_bioguard_wm_total_autonomy.py::_run_delta_calibrated_jepa_seed_grid + perturb_jepa.models.program_bootstrap_jepa.ProgramBootstrapJEPA	rank-3 train-split-only PLS raw-linear readout	0.0057	0.398	0.4815	10.2835	0.7744	0.20781648143800757	0.12491412882339287	0.9344029333900138	0.9047365628809659	1.0	1.0	7.0239482850671004	6.595168576592012	1.009602448752875	0.7716049382716049	0.8786008230452675	0.0	0.0	False	True	RNA-only diagnostic possible; not a paired scRNA+imaging Tier 3 validator.	False	True	True	True	False	True	True	0.20211648143800756	0.5364029333900138	0.5185	-3.2595517149328996	0.235202448752875	rank_preserving_delta_calibrator_or_rank_ladder_before_Tier3	F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS

```

## Promotion Status
No model is promoted. F083 is a preflight audit and explicitly blocks promotion unless every Tier 3 gate is satisfied.
