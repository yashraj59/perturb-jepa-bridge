# Architectural Changes Log

## Phase 8 v3 Implementation

- file changed: `perturb_jepa/models/jepawm_rope.py`
  - new functions/classes: `RotaryEmbedding`, `rotate_half`, `apply_rope`
  - initialization: deterministic sinusoidal RoPE buffers; no trainable parameters
  - observed effect: focused RoPE tests pass, including norm preservation and odd-head-dim rejection
  - identity risk: low; utility module only

- file changed: `perturb_jepa/models/action_adaln_predictor.py`
  - new functions/classes: `RoPESelfAttention`, `ActionAdaLNBlock`, `ActionAdaLNRoPEPredictor`
  - initialization: AdaLN-zero modulation and zero output head
  - observed effect: starts as zero residual and enforces fixed context-token contract
  - cost/memory impact: tiny CPU frozen-latent probe scale

- file changed: `perturb_jepa/models/bioguard_wm_jepa.py`
  - new functions/classes: `BioGuardWMJEPAConfig`, `RidgeFloorHead`, `FloorPreservingJEPAWMTransitionHead`
  - initialization: residual scale initialized to zero; protected floor exact when residual scale is zero
  - observed effect: floor preservation tests pass below `1e-7`
  - identity risk: guarded by JEPA identity tests and leakage reports

- file changed: `perturb_jepa/evaluation/bioguard_wm_metrics.py`
  - new functions/classes: `bioguard_wm_transition_metrics`, `collapse_diagnostics`, `write_metric_artifacts`
  - expected effect: centralize transition/floor-gap metrics for future total-autonomy families

- file changed: `scripts/run_bioguard_wm_total_autonomy.py`
  - new behavior: initializes total-autonomy logs, locks prior facts, writes paper notes, runs focused tests, runs BGWM000-BGWM002, writes Debate Council/amendment on ordinary failure, and continues into Family F diagnostics
  - observed effect: BGWM002 closed with zero residual on seeds 0/1/2; F001 and F002 diagnostics completed

- tests added/updated: `tests/test_total_autonomy_council.py`, `tests/test_jepa_identity_contract.py`, plus Phase 8 v3 checks in RoPE, predictor, and floor-preservation tests
  - observed effect: focused suite passed, `25 passed`
