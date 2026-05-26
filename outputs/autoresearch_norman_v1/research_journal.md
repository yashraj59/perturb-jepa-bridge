# Norman v1 Research Journal

## Experiment 0: Step 0 Baselines

**Hypothesis**: Before any architecture search, the Norman split must be anchored by simple recomputed baselines and immutable published comparators.

**Family**: Step0 / Stage A.

**Implementation**: Added `perturb_jepa/data/norman2019.py`, `scripts/run_norman_step0.py`, and `tests/test_norman2019_split.py`. No locked evaluator, gene set, DE definition, or published number was modified.

**Initialization / identity preservation**: GEARS published numbers and Family N remain the active model-of-record references. The bridge architecture is unchanged.

**Tier result**: BASELINE_COMPLETE. Best recomputed exact-train-combo all-gene Pearson delta baseline: `single_perturbation_additive` = `0.8981`.

**Diagnostics**: per-condition metrics, split counts, prediction-variance ratio, direction-aware DE agreement, and published-number caveats are written under `outputs/autoresearch_norman_v1/step0_baselines/`.

**Decision**: Step 0 review gate reached. Family A and Family B were not started.

**Learning**: Published GEARS numbers are aggregate Pearson DE/MSE values. They are not directly subset-specific and do not include top-20 DE overlap in Supplementary Table 6.

Runtime seconds: `23.45`.

## Experiment 7: Feature-conditioned perturbation bridge

**Hypothesis**: Family A can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.

**Family**: Family A.

**Implementation**: `A1_feature_bridge_mlp` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.

**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.

**Tier result**: TIER1_DISCARD_NO_SIGNAL. Exact-train-combo Pearson delta all genes = `0.6919`; Tier 1 gate = `0.9181`.

**Diagnostics**: exact top20 overlap = `0.4357`; unseen-single direction accuracy = `0.6351`; variance ratio = `0.7740`; hard fail = `primary 0.6919 <= gate 0.9181`; metadata = `{"description": "Feature-conditioned perturbation bridge", "device": "cuda", "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "operator_rank": 0}, "operator_gate_mean": 1.0, "operator_to_additive_ratio_final": 0.0, "operator_to_additive_ratio_post_gate": 0.0, "operator_to_additive_ratio_raw": 0.0, "train_conditions": 68, "train_loss_final": 0.0008159104618243873, "train_loss_initial": 0.038459520787000656}`.

**Learning**: Runtime `4.20` seconds. No model-of-record change.

## Experiment 8: Feature-conditioned low-rank operator bridge

**Hypothesis**: Family A can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.

**Family**: Family A.

**Implementation**: `A2_feature_bridge_rank2_operator` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.

**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.

**Tier result**: TIER1_DISCARD_NO_SIGNAL. Exact-train-combo Pearson delta all genes = `0.6566`; Tier 1 gate = `0.9181`.

**Diagnostics**: exact top20 overlap = `0.4000`; unseen-single direction accuracy = `0.6689`; variance ratio = `0.7442`; hard fail = `primary 0.6566 <= gate 0.9181`; metadata = `{"description": "Feature-conditioned low-rank operator bridge", "device": "cuda", "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "operator_rank": 2}, "operator_gate_mean": 0.006902818568050861, "operator_to_additive_ratio_final": 0.49196958541870117, "operator_to_additive_ratio_post_gate": 0.49196958541870117, "operator_to_additive_ratio_raw": 0.49196958541870117, "train_conditions": 68, "train_loss_final": 0.0010613586055114865, "train_loss_initial": 0.12292309105396271}`.

**Learning**: Runtime `3.16` seconds. No model-of-record change.

## Experiment 9: Pure additive composition architecture

**Hypothesis**: Family B can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.

**Family**: Family B.

**Implementation**: `B1_pure_additive_architecture` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.

**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.

**Tier result**: TIER1_DISCARD_NO_SIGNAL. Exact-train-combo Pearson delta all genes = `0.8981`; Tier 1 gate = `0.9181`.

**Diagnostics**: exact top20 overlap = `0.5750`; unseen-single direction accuracy = `0.0000`; variance ratio = `1.4616`; hard fail = `primary 0.8981 <= gate 0.9181`; metadata = `{"description": "Pure additive architecture using train single perturbation deltas.", "interaction_to_additive_ratio": 0.0, "train_single_genes": 70}`.

**Learning**: Runtime `0.15` seconds. No model-of-record change.

## Experiment 10: Additive plus bounded MLP interaction

**Hypothesis**: Family B can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.

**Family**: Family B.

**Implementation**: `B2_additive_bounded_interaction_mlp` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.

**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.

**Tier result**: TIER1_DISCARD_NO_SIGNAL. Exact-train-combo Pearson delta all genes = `0.8922`; Tier 1 gate = `0.9181`.

**Diagnostics**: exact top20 overlap = `0.5893`; unseen-single direction accuracy = `0.3730`; variance ratio = `1.2790`; hard fail = `primary 0.8922 <= gate 0.9181`; metadata = `{"description": "Additive plus bounded MLP interaction", "device": "cuda", "eval_all_interaction_to_additive_ratio_max": 749261120.0, "eval_all_interaction_to_additive_ratio_mean": 93979648.0, "interaction_dominating": false, "interaction_scale": 0.047957565635442734, "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "rank": 0}, "train_conditions": 68, "train_interaction_ratio_final": 0.0749145895242691, "train_loss_final": 0.0001007603423204273, "train_loss_initial": 0.00032221697620116174}`.

**Learning**: Runtime `2.59` seconds. No model-of-record change.

## Experiment 11: Additive plus low-rank interaction

**Hypothesis**: Family B can improve exact-combo pseudobulk delta prediction beyond the protected additive and Family N baselines.

**Family**: Family B.

**Implementation**: `B3_additive_low_rank_interaction` in `perturb_jepa/models/norman_compositional.py` via `scripts/run_norman_tier1_architecture.py`.

**Initialization / identity preservation**: Existing bridge and Step 0 baselines are unchanged. Feature-conditioned lookup support is opt-in and zero-init residual mode is tested separately.

**Tier result**: TIER1_DISCARD_NO_SIGNAL. Exact-train-combo Pearson delta all genes = `0.8940`; Tier 1 gate = `0.9181`.

**Diagnostics**: exact top20 overlap = `0.6036`; unseen-single direction accuracy = `0.4095`; variance ratio = `1.4435`; hard fail = `primary 0.8940 <= gate 0.9181`; metadata = `{"description": "Additive plus low-rank interaction", "device": "cuda", "eval_all_interaction_to_additive_ratio_max": 1363667200.0, "eval_all_interaction_to_additive_ratio_mean": 89777784.0, "interaction_dominating": false, "interaction_scale": 0.04768238216638565, "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "rank": 4}, "train_conditions": 68, "train_interaction_ratio_final": 0.08870905637741089, "train_loss_final": 9.525285713607445e-05, "train_loss_initial": 0.0003245782572776079}`.

**Learning**: Runtime `2.64` seconds. No model-of-record change.
