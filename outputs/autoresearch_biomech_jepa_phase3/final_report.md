# BioMechanistic-JEPA Phase 3 Final Report

## Model-Of-Record Decision

No Phase 3 candidate is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

Decision label: `SEARCH_CLOSED_NO_NEW_BASELINE`.

## JEPA Identity

BMJ001 remained a real JEPA candidate by identity checks:

- encoder path used: `1.0`
- raw-linear PLS main path used: `0.0`
- condition-key feature present: `0.0`
- teacher stop-gradient verified: `1.0`
- separate `z_bio`/`z_tech` latents present: `1.0`
- held-out action descriptor valid: `1.0`

## Diagnostics

Stage A completed with:

- image branch: `IMAGE_BRANCH_AUDIT_HEALTHY`
- transition target: `DELTA_TARGET_HAS_HEADROOM`
- action descriptor: `ACTION_DESCRIPTOR_VALID`
- overall: `PHASE3_DIAGNOSTICS_COMPLETE_PROCEED`

The transition target audit showed delta targets had real headroom:

- held-out delta teacher effective rank: `10.1424`
- held-out delta teacher std mean: `0.0819`
- held-out delta target NN recall@1: `1.0000`
- held-out delta target batch-probe accuracy: `0.3125` vs majority `0.4375`

## BMJ001 Result

BMJ001 implemented delta-state JEPA only.

Key metrics:

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

Stop condition fired because BMJ001 transition improvement was below `+0.0200`. BMJ002-BMJ006 were not launched.

## Required Questions

- Did delta targets improve over absolute targets? Diagnostic targets had headroom, but the learned BMJ001 delta predictor did not improve transition prediction.
- Did action descriptors help held-out perturbation generalization? Not tested in BMJ002 because BMJ001 stopped the loop. The descriptor audit passed.
- Did population prototype transition help? Not tested because BMJ001 stopped the loop.
- Did image-to-RNA retrieval recover from zero? BMJ001 image->RNA recall@1 was `0.1290`, but cross-modal repair BMJ004 was not run, so this is not a gated repair success.
- Did batch allocation remain correct? Weakly directionally correct but below the BMJ004 gate: `z_tech - z_bio` batch-probe gap was `0.0645`.
- Norman diagnostic result and limitations: BMJ005 was not run because BMJ001 fired a hard stop. Norman remains RNA-only, A549-only, with no exposed batch metadata and fixed guide-presence dose.

## Recommendation

Close Phase 3 as currently specified. The next amendment should target the delta predictor optimization itself before adding action-token or population complexity. The failure pattern is: delta targets are informative, but the first learned delta operator anti-aligns with the teacher delta and degrades target prediction.
