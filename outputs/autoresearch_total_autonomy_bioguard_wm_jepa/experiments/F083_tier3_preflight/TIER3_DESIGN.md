# Tier 3 Design Draft After F082

## Candidate
Delta-calibrated source/delta/rank ProgramBootstrapJEPA from F081/F082.

## Locked Code Path
- `perturb_jepa/models/program_bootstrap_jepa.py`
- `scripts/run_bioguard_wm_total_autonomy.py`
- candidate functions: `_train_source_delta_rank_program_jepa`, `_evaluate_delta_calibrated_source_delta_rank_jepa`, `_run_delta_calibrated_jepa_seed_grid`

## Must-Pass Gates Before Promotion
- transition improvement >= `0.005700`
- delta cosine >= `0.398000`
- recall@1 >= `0.481500`
- delta rank >= `10.283500`
- magnitude ratio >= `0.774400`
- RNA->image and image->RNA recall@1 >= `0.700000`
- no identity violation
- no leakage flag
- at least one paired scRNA+image external validator or a documented valid substitute

## Current F082 State
- transition: `0.207816`
- delta cosine: `0.934403`
- recall@1: `1.000000`
- delta rank: `7.023948`
- paired external validator available: `False`

## Current Blockers
- rank floor pass: `False`
- paired external validator available: `False`

## Decision
`F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS`

## Next Action
`rank_preserving_delta_calibrator_or_rank_ladder_before_Tier3`
