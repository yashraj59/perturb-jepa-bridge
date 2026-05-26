# BioFlow-JEPA Phase 4 Research Journal

## Protocol

Controlling prompt: `prompt/bioflow_jepa_phase4_delta_operator_prompt.md`

## Research Question

Can a real cross-modal, action-conditioned JEPA learn a biological transition operator by fitting an action-conditioned latent vector field in `z_bio` space, such that predicted transitions move control states toward held-out perturbed teacher states rather than anti-aligning with teacher deltas?

## Current Constraints

- Do not continue BMJ001 by training longer.
- Do not launch BMJ002-BMJ006.
- Run the mandatory delta-operator audit before any BioFlow model implementation.
- Reopen model implementation only if `REOPENING_DECISION.md` says `PHASE4_DELTA_OPERATOR_REOPEN_APPROVED`.
- Protected model of record remains the rank-3 train-split-only PLS raw-linear readout.

## BFJ000: Delta Operator Audit

**Hypothesis**: BMJ001 failed because the transition operator optimization anti-aligned with informative teacher deltas; audit cached teacher latents and tested simple train-only transition operators before reopening architecture work.

**Implementation**: `scripts/run_delta_operator_audit.py` and existing `scripts/cache_bioflow_teacher_latents.py`; no BioFlow model code implemented before the decision.

**Result**: eval teacher delta effective rank `11.7819`, eval delta std mean `0.0832`, best eval baseline `action_ridge_delta` transition improvement `0.0057`, best train frozen-latent optimization `endpoint_cosine` improvement `0.0607`.

**Decision label**: `PHASE4_DELTA_OPERATOR_REOPEN_APPROVED`.

**Stop-condition check**: if denied, stop and write final report; if approved, only then implement BioFlow-JEPA BFJ001.

## BFJ000: Delta Operator Audit

**Hypothesis**: BMJ001 failed because the transition operator optimization anti-aligned with informative teacher deltas; audit cached teacher latents and tested simple train-only transition operators before reopening architecture work.

**Implementation**: `scripts/run_delta_operator_audit.py` and existing `scripts/cache_bioflow_teacher_latents.py`; no BioFlow model code implemented before the decision.

**Result**: eval teacher delta effective rank `11.7819`, eval delta std mean `0.0832`, best eval baseline `action_ridge_delta` transition improvement `0.0057`, best train frozen-latent optimization `endpoint_cosine` improvement `0.0607`.

**Decision label**: `PHASE4_DELTA_OPERATOR_REOPEN_APPROVED`.

**Stop-condition check**: if denied, stop and write final report; if approved, only then implement BioFlow-JEPA BFJ001.

## BFJ001: Frozen-Encoder BioFlow Transition Probe

**Hypothesis**: A controlled vector field trained with train-only delta whitening, delta direction loss, and source-improvement hinge will stop the BMJ001 anti-aligned delta failure while preserving JEPA identity.

**Implementation files changed**: `perturb_jepa/models/bioflow_jepa.py`, `perturb_jepa/training/bioflow_losses.py`, `perturb_jepa/training/bioflow_trainer.py`, `perturb_jepa/evaluation/bioflow_metrics.py`, `scripts/train_bioflow_jepa.py`.

**Initialization / identity preservation**: base BioTech checkpoint `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt`; encoders frozen `True`; PLS main path `0.0`; condition-key feature `0.0`; teacher stop-gradient `1.0`.

**Tier result**: transition improvement `-0.0104`, delta cosine `-0.1054`, pred delta rank `7.6852`, recall@1 `0.0500`.

**Decision label**: `BFJ_TIER1_DISCARD_NO_SIGNAL`.

**Learning**: BFJ001 is a transition-operator diagnostic only; protected PLS remains the model of record and no Phase 4 result can promote.
