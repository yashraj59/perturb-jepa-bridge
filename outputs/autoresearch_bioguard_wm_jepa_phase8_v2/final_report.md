# BioGuard-WM-JEPA Phase 8 v2 Final Report

## Decision label
PHASE8V2_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA

## Model of record
Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass supersedes it.

## Phase 7 status integration
- Phase 7 decision: `PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA`
- Floor values: transition `0.0057`, delta cosine `0.3980`, recall@1 `0.4815`, rank `10.2835`
- Residual candidates locked as failed: spectral, kernel, program

## Paper integration
- Paper path: `papers/2512.24497v3.pdf`
- Lessons used: latent dynamics predictor, action-AdaLN + RoPE, fixed context contract, L2/cosine endpoint costs, deterministic predictor risk control.
- What was implemented from the paper: action-AdaLN + RoPE predictor primitives and floor-preserving JEPA-WM transition head.
- What was deliberately not implemented: stochastic/diffusion heads, planning optimizer, full JEPA wrapper unless BGWM002 passes.

## What was implemented
- Files changed: `perturb_jepa/models/jepawm_predictor.py`, `perturb_jepa/training/bioguard_wm_losses.py`, `perturb_jepa/training/bioguard_wm_calibration.py`, `perturb_jepa/training/bioguard_wm_rollouts.py`, `perturb_jepa/training/bioguard_wm_status.py`, `scripts/train_bioguard_wm_jepa.py`, `scripts/evaluate_bioguard_wm_jepa.py`, `scripts/run_bioguard_wm_phase8_v2.py`
- New classes: `BioJEPAWMContextConfig`, `RotaryEmbedding`, `RotarySelfAttention`, `ActionAdaLNBlock`, `ActionAdaLNPredictor`, `FloorPreservingJEPAWMTransitionHead`
- New tests: focused Phase 8 v2 tests

## What was tested
- Focused tests before BGWM runs: `18 passed`; see `test_results.md`
- BGWM000: audit/floor/paper/Phase 7 lock
- BGWM001: zero-residual action-AdaLN smoke
- BGWM002: frozen-latent action-AdaLN + RoPE residual with train-only calibration
- BGWM003-BGWM006: not run unless BGWM002 keeps a positive residual

## Key metrics
| experiment | transition improvement | delta cosine | recall@1 | delta rank | floor gap transition | floor gap recall | residual scale | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BGWM000 | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.000000 | 0.000000 | 0.0000 | BGWM000_KEEP_AUDIT_REOPEN_PREDICTOR_ASSAY |
| BGWM001 | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.000000 | 0.000000 | 0.0000 | BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT |
| BGWM002 | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.000000 | 0.000000 | 0.0000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |

## JEPA identity
- online/context encoders: operator-only unless full wrapper is opened
- EMA target encoders: operator-only unless full wrapper is opened
- stop-gradient target latents: yes for frozen teacher latents
- latent transition prediction: yes
- cross-modal JEPA: not run because BGWM002 did not keep a residual
- raw-linear PLS main path used: no

## Leakage status
- eval target rows used for fitting: no
- condition_key input: no
- biological_key input: no
- eval target means: no
- pooled train+test stats: no

## Main interpretation
Phase 8 v2 answered the predictor question under current data. If BGWM002 selected zero residual, the JEPA-WM predictor recipe did not provide train-internal evidence for a safe residual above the protected full-ridge floor.

## Recommendation
BGWM002 train-only calibration selected residual_scale=0. BSG/BSJ residual failures remain locked.
