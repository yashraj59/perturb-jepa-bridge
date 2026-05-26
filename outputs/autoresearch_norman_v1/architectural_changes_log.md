# Architectural Changes Log

## Step 0

- Architecture experiments run: `0`.
- Bridge, perturbation encoder, evaluator code, Norman split definition after creation, gene set/DE definitions, and published baseline values were not modified during search.
- New non-architecture infrastructure: Norman condition loader, split fidelity test, and Step 0 baseline script.
- Family A and Family B are locked until Step 0 review.

## Experiment 7: A1_feature_bridge_mlp

- Family: `Family A`
- Status: `TIER1_DISCARD_NO_SIGNAL`
- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`
- Config/diagnostics: `{"description": "Feature-conditioned perturbation bridge", "device": "cuda", "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "operator_rank": 0}, "operator_gate_mean": 1.0, "operator_to_additive_ratio_final": 0.0, "operator_to_additive_ratio_post_gate": 0.0, "operator_to_additive_ratio_raw": 0.0, "train_conditions": 68, "train_loss_final": 0.0008159104618243873, "train_loss_initial": 0.038459520787000656}`

## Experiment 8: A2_feature_bridge_rank2_operator

- Family: `Family A`
- Status: `TIER1_DISCARD_NO_SIGNAL`
- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`
- Config/diagnostics: `{"description": "Feature-conditioned low-rank operator bridge", "device": "cuda", "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "operator_rank": 2}, "operator_gate_mean": 0.006902818568050861, "operator_to_additive_ratio_final": 0.49196958541870117, "operator_to_additive_ratio_post_gate": 0.49196958541870117, "operator_to_additive_ratio_raw": 0.49196958541870117, "train_conditions": 68, "train_loss_final": 0.0010613586055114865, "train_loss_initial": 0.12292309105396271}`

## Experiment 9: B1_pure_additive_architecture

- Family: `Family B`
- Status: `TIER1_DISCARD_NO_SIGNAL`
- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`
- Config/diagnostics: `{"description": "Pure additive architecture using train single perturbation deltas.", "interaction_to_additive_ratio": 0.0, "train_single_genes": 70}`

## Experiment 10: B2_additive_bounded_interaction_mlp

- Family: `Family B`
- Status: `TIER1_DISCARD_NO_SIGNAL`
- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`
- Config/diagnostics: `{"description": "Additive plus bounded MLP interaction", "device": "cuda", "eval_all_interaction_to_additive_ratio_max": 749261120.0, "eval_all_interaction_to_additive_ratio_mean": 93979648.0, "interaction_dominating": false, "interaction_scale": 0.047957565635442734, "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "rank": 0}, "train_conditions": 68, "train_interaction_ratio_final": 0.0749145895242691, "train_loss_final": 0.0001007603423204273, "train_loss_initial": 0.00032221697620116174}`

## Experiment 11: B3_additive_low_rank_interaction

- Family: `Family B`
- Status: `TIER1_DISCARD_NO_SIGNAL`
- Code: `perturb_jepa/models/norman_compositional.py`, `scripts/run_norman_tier1_architecture.py`
- Config/diagnostics: `{"description": "Additive plus low-rank interaction", "device": "cuda", "eval_all_interaction_to_additive_ratio_max": 1363667200.0, "eval_all_interaction_to_additive_ratio_mean": 89777784.0, "interaction_dominating": false, "interaction_scale": 0.04768238216638565, "model_config": {"dropout": 0.0, "feature_dim": 140, "gene_dim": 5045, "hidden_dim": 64, "rank": 4}, "train_conditions": 68, "train_interaction_ratio_final": 0.08870905637741089, "train_loss_final": 9.525285713607445e-05, "train_loss_initial": 0.0003245782572776079}`
