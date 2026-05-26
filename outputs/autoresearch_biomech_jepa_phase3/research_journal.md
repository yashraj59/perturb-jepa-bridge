# BioMechanistic-JEPA Phase 3 Research Journal

## Protocol

Controlling prompt: `prompt/biomech_jepa_phase3_amendment_prompt.md`

Phase 3 starts from the Phase 2 conclusion: BioTech-JEPA is a real JEPA diagnostic success, but it is not promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

## Phase 3 Research Question

Can a real JEPA biological action world model improve held-out genetic perturbation transition prediction by making the main target a biological state delta, using non-leaky gene/program action descriptors, predicting population prototype transitions, retaining `z_bio`/`z_tech` separation, and repairing RNA/image cross-modal prediction?

## Diagnostic-First Gate

Before architecture work, Phase 3 must complete:

1. image branch health audit;
2. transition target audit;
3. action descriptor audit.

Architecture changes are allowed only if the diagnostics do not trigger a stop condition.

## Stage A Diagnostics

### A1 Image Branch Health

Command:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/audit_biotech_image_branch.py --device cuda --eval-steps 4 --batch-size 4 --bag-size 3 --output-dir outputs/autoresearch_biomech_jepa_phase3/diagnostics/image_branch_health
```

Decision label: `IMAGE_BRANCH_AUDIT_HEALTHY`

Key findings:

- Trained online image->RNA recall@1: `0.0000`
- Trained teacher image->RNA recall@1: `0.1250`
- Random online image->RNA recall@1: `0.1875`
- Image teacher `z_bio` effective rank: `5.9059`
- Image teacher `z_bio` std mean: `0.0858`
- Image/RNA branch gradient ratio: `0.9884`
- Cross-modal weighted/total loss ratio: `0.3436`

Interpretation: no loader failure or image latent collapse was found. The failure is directional training/objective behavior: the trained online image->RNA path is worse than the teacher and worse than a random checkpoint on this tiny sample.

### A2 Transition Target Audit

Command:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/audit_biotech_transition_targets.py --device cuda --steps 4 --batch-size 4 --bag-size 3 --output-dir outputs/autoresearch_biomech_jepa_phase3/diagnostics/transition_target_audit
```

Decision label: `DELTA_TARGET_HAS_HEADROOM`

Key held-out metrics:

- Source->target cosine mean: `0.9091`
- Source->delta cosine mean: `-0.2092`
- Delta teacher effective rank: `10.1424`
- Delta teacher std mean: `0.0819`
- Absolute target NN recall@1: `0.3750`
- Delta target NN recall@1: `1.0000`
- Delta target batch-probe accuracy: `0.3125` vs majority `0.4375`

Interpretation: delta targets are not collapsed and are not more batch-predictive than majority in this audit. Delta-state JEPA is permitted.

### A3 Action Descriptor Audit

Command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/audit_action_descriptors.py --output-dir outputs/autoresearch_biomech_jepa_phase3/diagnostics/action_descriptor_audit
```

Decision label: `ACTION_DESCRIPTOR_VALID`

Key findings:

- Synthetic held-out descriptor coverage: `1.0000`
- Synthetic held-out nonzero descriptor fraction: `1.0000`
- Synthetic descriptor rank: `11.0000`
- Norman action genes: `105`
- Norman descriptor rank: `105`
- Norman chemical dose used: `0`
- Norman batch metadata used: `0`

Stage A decision: `PHASE3_DIAGNOSTICS_COMPLETE_PROCEED`.

## BMJ001: Delta-Target Synthetic Only

Hypothesis: delta-state JEPA should improve action-conditioned transition prediction because the transition target audit showed non-collapsed, discriminative delta targets.

Command:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/train_biomech_jepa.py --dataset synth_genetic_anchor_lite --eval-split test_heldout_perturbation --seed 0 --device cuda --steps 80 --eval-steps 8 --batch-size 2 --bag-size 4 --shared-dim 32 --bio-dim 24 --tech-dim 8 --predictor-dim 64 --num-condition-prototypes 4 --enable-delta-jepa --output-dir outputs/autoresearch_biomech_jepa_phase3/experiments/BMJ001_delta_synth_seed0 --save-checkpoint
```

Result:

- `encoder_path_used = 1.0`
- `pls_raw_linear_main_path_used = 0.0`
- `condition_key_feature_present = 0.0`
- `heldout_action_descriptor_valid = 1.0`
- transition source cosine improvement: `-0.1695`
- transition-to-target recall@1: `0.0968`
- transition-to-target median rank: `7.0`
- delta cosine: `-0.0332`
- absolute target cosine: `0.6891`
- delta teacher effective rank: `12.4196`
- delta prediction effective rank: `4.7813`
- target `z_bio` effective rank: `8.1391`
- image->RNA recall@1: `0.1290`
- RNA->image recall@1: `0.0968`
- batch allocation gap: `0.0645`

Decision label: `TIER1_DISCARD_NO_SIGNAL`.

Stop condition fired: BMJ001 transition improvement is below `+0.0200`. Per prompt, no BMJ002-BMJ006 runs were launched.

Interpretation: the diagnostic delta target had headroom, but the first learned delta predictor moved the predicted target away from the teacher target. The failure is in the learned transition operator, not in JEPA identity or descriptor availability.
