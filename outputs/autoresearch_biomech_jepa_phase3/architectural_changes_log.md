# Architectural Changes Log

No BioMechanistic-JEPA architecture has been implemented at Phase 3 start.

Only diagnostic scripts and non-leaky action descriptor utilities may be added before the Stage A diagnostics are complete.

## Stage A Diagnostic Code

- `scripts/audit_biotech_image_branch.py`
- `scripts/audit_biotech_transition_targets.py`
- `scripts/audit_action_descriptors.py`
- `perturb_jepa/training/action_descriptors.py`
- `tests/test_action_descriptors.py`

Stage A decision: `PHASE3_DIAGNOSTICS_COMPLETE_PROCEED`.

Architecture work is now permitted because no stop condition fired:

- no image data/loader issue was found;
- delta targets have headroom;
- held-out action descriptors are valid and non-leaky.

## BioMechanistic-JEPA Code Additions

- `perturb_jepa/models/biomech_jepa.py`
  - `BioMechanisticJEPAConfig`
  - `ProgramActionEncoder`
  - `PopulationTransitionPredictor`
  - `BioMechanisticJEPA`
- `perturb_jepa/training/biomech_losses.py`
  - delta JEPA, target transition JEPA, prototype losses, anti-collapse losses, and auxiliary/main loss ratio diagnostics.
- `perturb_jepa/training/biomech_trainer.py`
- `perturb_jepa/evaluation/biomech_metrics.py`
- `scripts/train_biomech_jepa.py`
- `scripts/evaluate_biomech_jepa.py`
- `tests/test_biomech_jepa_model.py`
- `tests/test_biomech_transition_targets.py`

Focused tests:

```text
pytest tests/test_biomech_jepa_model.py tests/test_biomech_transition_targets.py tests/test_action_descriptors.py tests/test_norman_biotech_batches.py tests/test_synthetic_biology_lite.py::test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates
```

Result: `8 passed`.

BMJ001 outcome: `TIER1_DISCARD_NO_SIGNAL`. Stop condition fired before action-token, population, cross-modal repair, Norman, or Tier 2 runs.
