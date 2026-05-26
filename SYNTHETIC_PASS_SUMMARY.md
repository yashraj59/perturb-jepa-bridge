# Synthetic Pass Summary

## Bottom Line
The best working synthetic/current-registry candidate is the delta-calibrated real JEPA wrapper from F082:

```text
F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN
```

It is a real JEPA path, not a renamed autoencoder or PLS clone. It uses online/context encoders, EMA target encoders, stop-gradient target latents, query/predictor paths, cross-modal RNA/image prediction, and an action-conditioned transition path. A train-only delta calibrator is applied after the JEPA wrapper to recover held-out delta direction without fitting on eval targets.

This candidate passed synthetic/current-registry gates strongly enough for Tier 3 design, but it is not promoted because no real paired scRNA plus imaging Tier 3 validator has run.

## Best Metrics
From `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F082_delta_calibrated_tier2_validation/F082_DELTA_CALIBRATED_TIER2_VALIDATION.md`:

```text
seeds = 37,38,39,40,41,42
seed/policy rows = 18
mean calibrated transition improvement = 0.207816
std calibrated transition improvement = 0.039553
min calibrated transition improvement = 0.124914
mean calibrated delta cosine = 0.934403
min calibrated delta cosine = 0.904737
mean calibrated recall@1 = 1.000000
min calibrated recall@1 = 1.000000
mean calibrated delta rank = 7.023948
mean magnitude ratio = 1.009602
mean RNA -> image recall@1 = 0.771605
mean image -> RNA recall@1 = 0.878601
max identity violation = 0
max leakage flag = 0
```

F086 then showed:

```text
synthetic_current_registry_gates_pass = True
cross_modal_gate_pass = True
candidate_preserves_current_floor = True
current_registry_rank_supported = True
```

## What Finally Made Synthetic Work
The key change was not simply training longer. The useful path was:

1. Redesign the synthetic benchmark around descriptor-aligned program actions so non-exact action descriptors had learnable biological structure.
2. Preserve source-state structure rather than forcing a direct endpoint-only transition predictor.
3. Add explicit delta-direction and rank repair before wrapping the mechanism in a full JEPA.
4. Wrap the source/delta/rank repair into `ProgramBootstrapJEPA`, with real JEPA identity components and cross-modal RNA/image prediction.
5. Add a train-only delta calibrator to correct the JEPA predicted deltas into the image-teacher latent delta geometry.
6. Validate on fresh seeds and multiple split policies before calling it a synthetic pass.

In short:

```text
descriptor-aligned synthetic benchmark
-> source/program image-teacher repair
-> source/delta/rank repair
-> real ProgramBootstrapJEPA wrapper
-> train-only delta calibration
-> fresh-seed Tier 2 style validation
```

## Main Milestones
Important late-stage experiments:

```text
F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN
F028_PCA_DISTILLED_RNA_ENCODER_PRESERVES_PROGRAM_FLOOR
F029_REPRESENTATION_FLOOR_GOOD_DIRECT_JEPA_UNDER_FLOOR
F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA_TIER1_PASS
F031_PROTECTED_TIER2_PASS_LOCAL_RIDGE_FLOOR_NOT_PRESERVED
F079_SOURCE_DELTA_RANK_REPAIR_READY_FOR_WRAPPER
F080_FULL_JEPA_WRAPPER_MIXED_OR_INCONCLUSIVE
F081_DELTA_CALIBRATED_JEPA_TIER1_PASS_NONPROMOTING
F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN
F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE
F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED
```

## How Many Things Were Tried
The total-autonomy result registry currently has:

```text
128 experiment/result rows
128 unique experiment IDs
121 debate-council continuation cycles
```

Major families tried:

```text
Representation repair: 33 rows
Metric and data redesign: 23 rows
Descriptor-aligned action contract: 17 rows
Program and graph action priors: 10 rows
External validator: 6 rows
Tier 3 design: 4 rows
```

Other families included action-AdaLN/RoPE residuals, floor-preserving real JEPA, exact floor-initialized operator wrappers, data-contract calibration, retrieval-margin gates, delta-direction JEPA, operator-initialized JEPA, population transport, PCA-bootstrap JEPA, and fresh synthetic seed geometry.

## What Failed Before This
Repeated failures were useful:

- residual branches often improved train fit but fell below the protected floor on held-out transition/retrieval;
- row-wise abstention produced safe but too-weak gains;
- train-only selectors were optimistic relative to real held-out perturbations;
- vanilla Cell-JEPA-style RNA warmstart did not become a safe transition solution;
- direct JEPA transitions could improve delta direction but did not preserve the same-representation ridge floor;
- source/image teacher paths often lost perturbation structure until source/delta/rank repair was added.

## Current Real-Data Blocker
The next required proof is real paired scRNA plus imaging validation. scGeneScope is the best candidate found, but F092 hit effective workspace quota:

```text
HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED
```

So the synthetic candidate is ready for another cluster, but real validation must begin with the quota-safe recovery plan:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
```

## Promotion Status
No model is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record until a Tier 3 pass supersedes it.
