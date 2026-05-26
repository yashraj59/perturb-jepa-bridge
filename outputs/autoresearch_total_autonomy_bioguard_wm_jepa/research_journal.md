# Research Journal

## F001_BOOTSTRAP_FLOOR_NOISE_AUDIT: Family F diagnostic

Bootstrap 95% interval for floor transition improvement: [-0.014673, 0.026343].

## F002_HELDOUT_DIFFICULTY_STRATIFICATION: Family F diagnostic

Stratified held-out perturbations by teacher delta norm without changing split semantics.
## F003_ACTION_CONDITION_HETEROGENEITY_AUDIT: Family F diagnostic

Per-action audit found 2 positive and 1 non-positive perturbation groups under the floor.

## F004_ACTION_SUPPORT_DISTANCE_AUDIT: Family F diagnostic

Compared held-out perturbation action descriptors against train action support and linked support distance to floor performance.

## F005_TRAIN_ONLY_LEAVE_ACTION_OUT_FLOOR_AUDIT: Family F diagnostic

Ran train-only leave-action-out floor scoring to estimate action-ridge generalization failures without eval selection.

## F006_RIDGE_ENSEMBLE_UNCERTAINTY_AUDIT: Family F diagnostic

Used train-only bootstrap ridge ensemble disagreement as a candidate abstention signal and scored it against held-out action heterogeneity.

## F007_REPLICATE_CEILING_AUDIT: Family F diagnostic

Measured condition-level target/delta replicate consistency and compared it with floor-vs-source gains.

## C001_REPRESENTATION_ERROR_AUDIT: Family C diagnostic

Audited online/teacher consistency, latent rank, floor-error direction, and z_tech coupling by held-out action.

## C002_LATENT_DELTA_MANIFOLD_SUPPORT_AUDIT: Family C diagnostic

Compared held-out action response deltas to train action delta manifolds using only train delta support as the reference.

## C003_TRAIN_ONLY_DELTA_SUPPORT_AUDIT: Family C diagnostic

Checked whether train-only leave-action-out floor failures align with weak latent delta support from remaining train actions.

## E001_ACTION_PRIOR_SUFFICIENCY_AUDIT: Family E diagnostic

Compared source-only, action-only, and source+action ridge floors under train-only leave-action-out and held-out scoring.

## E002_TRAIN_ONLY_SOURCE_ACTION_BLEND_AUDIT: Family E diagnostic

Train-only calibrated source/action floor blend selected alpha=0.10; decision=E002_BLEND_SELECTED_DIAGNOSTIC_ONLY.

## D001_PROTOTYPE_TRANSITION_DIAGNOSTIC: Family D diagnostic

Fit a train-only source-prototype plus action ridge transition to test whether population/prototype state helps the held-out failure mode.

## E003_CONDITION_LEVEL_SOURCE_ACTION_BLEND_AUDIT: Family E diagnostic

Condition-level train-only blend calibration selected alpha=0.000; decision=E003_CONDITION_BLEND_REJECTED_KEEP_FULL_FLOOR.

## E004_SUPPORT_AWARE_SOURCE_ACTION_BLEND_AUDIT: Family E diagnostic

Support-aware train-only blend calibration selected alpha=0.100, threshold=0.4934; decision=E004_SUPPORT_AWARE_BLEND_SELECTED_DIAGNOSTIC_ONLY.

## E005_PREDICTED_SUPPORT_SCORE_AUDIT: Family E diagnostic

Audited whether deployable predicted-delta support separates held-out action floor failures from safe actions.

## E006_HELDOUT_SUPPORT_THRESHOLD_ORACLE_AUDIT: Family E diagnostic

Held-out oracle sweep for support-gated source/action blending found 31 floor-preserving candidates. Audit only; no selection.

## E007_HIGH_SUPPORT_ABSTENTION_CALIBRATION_DIAGNOSTIC: Family E diagnostic

High-support train-only calibration selected alpha=0.100, threshold=0.8863; decision=E007_HIGH_SUPPORT_CALIBRATION_DISCARDED_HELDOUT_BELOW_FLOOR.

## E008_NESTED_TRAIN_ONLY_HIGH_SUPPORT_CALIBRATION_AUDIT: Family E diagnostic

Nested train-only high-support calibration mean transition gap=-0.000016, min recall gap=0.000000; decision=E008_NESTED_HIGH_SUPPORT_CALIBRATION_FAILED_TRAIN_ONLY.

## F008_TRAIN_ONLY_HELDOUT_SET_GEOMETRY_AUDIT: Family F diagnostic

Train-only triplet geometry audit found 4 negative transition triplets and 55 recall-below-floor triplets across 56 triplets.

## F009_SUPPORT_THRESHOLD_SPLIT_REDESIGN_AUDIT: Family F diagnostic

Support-threshold redesign found 22 rules with no negative transition but 0 rules preserving recall floor.

## R001_RETRIEVAL_LABEL_GRANULARITY_AUDIT: Retrieval metric diagnostic

Condition recall=0.2659, perturbation recall=0.3201, cell-line recall=0.8313; decision=R001_RETRIEVAL_DOMINATED_BY_CELL_LINE_NOT_PERTURBATION.

## C004_CELL_LINE_SOURCE_DOMINANCE_AUDIT: Representation repair diagnostic

Pred endpoint eta(cell)=0.5015, eta(pert)=0.0416; decision=C004_CELL_LINE_SOURCE_DOMINANCE_CONFIRMED_PERTURBATION_DELTA_UNDERREPRESENTED.

## C005_PERTURBATION_CENTERED_DELTA_RETRIEVAL_AUDIT: Representation repair diagnostic

Delta perturbation recall=0.4279 vs endpoint perturbation recall=0.3201; decision=C005_DELTA_SPACE_PARTIALLY_REPAIRS_PERTURBATION_RECALL_BUT_BELOW_FLOOR.

## C006_ENDPOINT_DELTA_COMPOSITE_RETRIEVAL_AUDIT: Representation repair diagnostic

Best composite condition recall=0.3671 at delta_weight=0.10 after correcting rank-position accounting; decision=C006_COMPOSITE_RETRIEVAL_PARTIALLY_REPAIRS_CONDITION_RECALL_BELOW_FLOOR.
## C007_NESTED_COMPOSITE_RETRIEVAL_CALIBRATION_AUDIT: Representation repair diagnostic

Nested train-only composite selection achieved condition recall=0.3651 with mean selected delta_weight=0.148; decision=C007_NESTED_COMPOSITE_CALIBRATION_PARTIAL_GAIN_BELOW_FLOOR.

## G001_FRESH_SYNTHETIC_SEED_LATENT_CACHE_AUDIT: Fresh-seed geometry diagnostic

Fresh seed-1 floor recall=0.3333; best composite train-only condition recall=0.3717 at delta_weight=0.50; decision=G001_FRESH_SYNTHETIC_SEED_CONFIRMS_COMPOSITE_PARTIAL_REPAIR_BELOW_FLOOR.

## F010_METRIC_DISAGREEMENT_AUDIT: Metric and data redesign diagnostic

Fresh seed improved transition/delta but recall fell by -0.1481; current/fresh composite gains were 0.1012/0.0952; decision=F010_RECALL_GATE_SEED_UNSTABLE_CONTINUOUS_METRICS_DISAGREE.

## C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_AUDIT: Representation repair diagnostic

Source-state constrained retrieval achieved condition recall=0.4861 at delta_weight=1.00; decision=C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_REACHES_RECALL_FLOOR_DIAGNOSTIC.

## C009_NESTED_SOURCE_STATE_RETRIEVAL_CALIBRATION_AUDIT: Representation repair diagnostic

Nested source-state retrieval current recall=0.4802, fresh recall=0.4471; decision=C009_NESTED_SOURCE_STATE_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR.

## C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_AUDIT: Representation repair diagnostic

Nested source-latent neighborhood retrieval current recall=0.3981, fresh recall=0.4140; decision=C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR.

## C011_SOURCE_NEIGHBORHOOD_PURITY_AUDIT: Representation repair diagnostic

Top-third same-cell-line purity=0.7157, exact-vs-latent recall gap=0.0820; decision=C011_SOURCE_LATENT_NEIGHBORHOOD_IMPURITY_EXPLAINS_PROXY_FAILURE.

## C012_SOURCE_STATE_LATENT_SPACE_AUDIT: Representation repair diagnostic

Best top-third source-state purity=0.9112 in z_bio_online; z_bio purity=0.7157; decision=C012_SOURCE_STATE_SIGNAL_STRONGER_OUTSIDE_Z_BIO.

## C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_AUDIT: Representation repair diagnostic

Online-source-neighborhood retrieval current recall=0.4577, fresh recall=0.4187; decision=C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR.

## C014A_ONLINE_SOURCE_NEIGHBORHOOD_ORACLE_CAPACITY_AUDIT: Representation repair diagnostic

Current seed oracle recall=0.4993, fresh seed oracle recall=0.4511; decision=C014_ORACLE_CAPACITY_CURRENT_ONLY_FRESH_BELOW_FLOOR. This separates inner-rule calibration from source-neighborhood capacity and shows current-only oracle capacity with fresh-seed instability.

## C014_ONLINE_TEACHER_SOURCE_GEOMETRY_AUDIT: Representation repair diagnostic

Online-vs-teacher purity gain=0.1955, eta gain=0.2628, residual eta=0.5674; decision=C014_TEACHER_TARGET_ATTENUATES_SOURCE_STATE_STRUCTURE.

## C015_SOURCE_STATE_PRESERVATION_DECISION_AUDIT: Representation repair diagnostic

Current online recall=0.4577, fresh online recall=0.4187, floor=0.4815; decision=C015_DO_NOT_REOPEN_TRAINING_SOURCE_STATE_PRESERVATION_INSUFFICIENT.


## G002_MULTI_SEED_FRESH_CACHE_REPLICATION: Fresh-seed geometry diagnostic

Fresh seeds [1, 2, 3, 4] online-source recall mean=0.4259, min=0.3836, floor=0.4815; decision=G002_FRESH_REPLICATION_CONFIRMS_SOURCE_PROXY_INSTABILITY_BELOW_FLOOR.

## F011_PROTECTED_FLOOR_SEED_STABILITY_AUDIT: Metric/data redesign diagnostic

Seed0 recall=0.4815, fresh mean recall=0.2963, fresh max recall=0.3333; transition seed0=0.0057, fresh mean transition=0.0127; decision=F011_RECALL_FLOOR_SEED_SPECIFIC_CONTINUOUS_METRICS_STABLE.

## F012_BENCHMARK_REDESIGN_BRIEF: Metric/data redesign diagnostic

Decision=F012_BENCHMARK_REDESIGN_REQUIRED_BEFORE_MORE_ARCHITECTURE. The old seed0 protected benchmark remains locked, but future architecture work should move to a new named multi-seed benchmark registry rather than silently changing gates.
## G002_MULTI_SEED_SOURCE_STATE_STABILITY_AUDIT: Fresh synthetic seed geometry diagnostic

Completed 4 seeds; online pass fraction=0.000, mean online floor gap=-0.0506; decision=G002_MULTI_SEED_SOURCE_STATE_SIGNAL_UNSTABLE_NO_TRAINING_REOPEN.


## F013_MULTISEED_BENCHMARK_REGISTRY: Metric/data redesign diagnostic

Built synthetic_genetic_anchor_lite_multiseed_v1 registry across seeds 0-4. Full action-ridge transition mean=0.0113, delta cosine mean=0.4515, recall mean=0.3333. Proposed gates are not active.

## G002A_BOOKKEEPING_NOTE: Fresh-cache replication row preserved

Added G002A to preserve the multi-seed fresh-cache replication result because G002 is also used by a source-state stability diagnostic artifact in the current worktree. No metric, gate, or model status changed.
## F013_MULTISEED_STEP0_BASELINE_REGISTRY: Metric/data redesign diagnostic

Built Step 0 registry for synthetic_genetic_anchor_lite_multiseed_v1 across 5 seeds; decision=F013_MULTISEED_STEP0_REGISTRY_COMPLETE_ARCHITECTURE_STILL_LOCKED.


## F014_MULTISEED_V1_BENCHMARK_ACTIVATION: Metric/data redesign diagnostic

Activated synthetic_genetic_anchor_lite_multiseed_v1 for search with transition gate=0.0032, delta gate=0.3944, secondary recall gate=0.2556. Old benchmark unchanged.
## BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY: Phase 8 v3 Tier 1 sanity

Ran Action-AdaLN floor-preserving residual sanity across 5 seeds; mean transition=0.0113, mean delta=0.4515, mean recall=0.3333; decision=BGWM003_PASS_FLOOR_PRESERVING_SANITY_ZERO_RESIDUAL.


## BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY: Action-AdaLN residual candidate

Ran existing action-AdaLN residual candidate across seeds 0-4 on synthetic_genetic_anchor_lite_multiseed_v1. Transition mean=0.0113, delta mean=0.4515, recall mean=0.3333, nonzero residual fraction=0.000; decision=BGWM003_NO_IMPROVEMENT_ZERO_RESIDUAL_FLOOR_FALLBACK.
## F015_MULTI_ENVIRONMENT_TRANSITION_AUDIT: Metric/data redesign diagnostic

Best non-floor baseline=pooled_source_only_ridge transition gap=0.0028, delta gap=-0.0010; decision=F015_POOLED_ENVIRONMENT_TRANSITION_PASSES_GATES_BELOW_PER_SEED_FLOOR.

## F016_ENVIRONMENT_BLEND_CALIBRATION: Metric/data redesign diagnostic

Train-only selected candidate=floor_fallback alpha=0.000; heldout transition gap=0.0000, recall gap=0.0000; decision=F016_TRAIN_ONLY_BLEND_SELECTS_FLOOR_FALLBACK.

## F017_ENVIRONMENT_BLEND_ORACLE_CAPACITY: Metric/data redesign diagnostic

Oracle best candidate=pooled_source_only_ridge alpha=1.000; transition gap=0.0105, recall gap=0.0519; decision=F017_ORACLE_SAFE_ENVIRONMENT_BLEND_CAPACITY_EXISTS.

## F018_ENVIRONMENT_BLEND_RISK_PROXY: Metric/data redesign diagnostic

Selected rule=pooled_full_ridge alpha=0.350 feature=support_gain; heldout transition gap=0.0034, mean recall gap=0.0000, but seed0 recall gap=-0.1111; decision=F018_TRAIN_ONLY_RISK_PROXY_DISCARDED_HELDOUT_BELOW_FLOOR.
## F019_RISK_PROXY_FAILURE_LOCALIZATION: Metric/data redesign diagnostic

F018 unsafe seed0 failure audited; best second gate=action_support ge 0.5000; decision=F019_NO_SIMPLE_SECOND_GATE_REPAIRS_F018_FAILURE.

## F020_LEARNED_ENVIRONMENT_RISK_GATE: Metric/data redesign diagnostic

Selected rule=floor_fallback alpha=0.000; heldout transition gap=0.0000, min recall gap=0.0000; decision=F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK.

## F021_REPRESENTATION_PIVOT_REQUALIFICATION: Metric/data redesign diagnostic

Environment branch cooled=True; G002 active-gate pass=True; decision=F021_REOPEN_REPRESENTATION_REPAIR_UNDER_ACTIVE_MULTISEED_GATES.

## C016_ONLINE_LATENT_TRANSITION_AUDIT: Representation repair diagnostic

Online passes active gates=True; recall gain vs teacher=0.0296; transition gain vs teacher=-0.0041; decision=C016_ONLINE_LATENT_REPRESENTATION_HAS_REPAIR_SIGNAL.

## F021_CELLJEPA_RNA_REPRESENTATION_WARMSTART: Representation warmstart diagnostic

Decision=F021_CELLJEPA_WARMSTART_DISCARDED_NO_SAFE_REPRESENTATION_GAIN; mean new transition=0.000025; mean old transition=0.0113; mean same-condition gain=-0.0667; weighted JEPA/reconstruction ratio=29.61.
## F022_RNA_REPRESENTATION_CEILING_AUDIT: Representation ceiling diagnostic

Decision=F022_SYNTHETIC_Z_BIO_CEILING_DOES_NOT_SUPPORT_TRANSITION_SEARCH; best_non_old=true_synthetic_z_bio; true_z_bio_transition=-0.0629.

## F023_OLD_LATENT_ACTION_CONTRACT_AUDIT: Old latent action/source contract diagnostic

Decision=F023_OLD_LATENT_ACTION_CONTRACT_AMBIGUOUS; full-source transition gap=0.001415; full-wrong-action gap=0.003291; action_only_pass=False.

## F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_AUDIT: Heldout action descriptor support diagnostic

Decision=F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_LIMITS_ACTION_GENERALIZATION; mean support fraction=0.5000; mean full-wrong transition gain=0.003291.

## F025_PROGRAM_ONLY_ACTION_DESCRIPTOR_AUDIT: Program-only action descriptor diagnostic

Decision=F025_NON_EXACT_PROGRAM_ACTION_DESCRIPTOR_PRESERVES_OLD_FLOOR; supported-source transition gap=0.001415; program-source transition gap=-0.007482.

## F026_DESCRIPTOR_ALIGNED_SYNTHETIC_BENCHMARK_AUDIT: Benchmark/action-target redesign diagnostic

Decision=F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN; new true-z program transition=0.155562; new true-z program recall=1.000000.

## F027_CELLJEPA_PROGRAM_ALIGNED_WARMSTART: Representation warmstart on descriptor-aligned benchmark

Decision=F027_CELLJEPA_BELOW_PCA_FLOOR_USE_REPRESENTATION_REPAIR; CellJEPA transition=0.000071; PCA transition=0.280793.

## F028_TRAIN_ONLY_PCA_DISTILLED_RNA_ENCODER: Representation repair with train-only PCA teacher

Decision=F028_PCA_DISTILLED_RNA_ENCODER_PRESERVES_PROGRAM_FLOOR; distilled transition=0.251776; PCA transition=0.280793.

## F029_PCA_BOOTSTRAP_CROSS_MODAL_ACTION_JEPA: Real JEPA Tier 1 probe on descriptor-aligned benchmark

Decision=F029_REPRESENTATION_FLOOR_GOOD_DIRECT_JEPA_UNDER_FLOOR; direct transition=0.066379; ridge floor=0.090390; cross RNA->image=0.7407.

## F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA: Delta-direction repair for real cross-modal/action JEPA

Decision=F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA_TIER1_PASS; direct delta cosine=0.875045; gain vs F029=0.643023; transition=0.244565.

## F031_DELTA_DIRECTION_TIER2_VALIDATION: Tier 2 validation for F030 delta-direction JEPA

Decision=F031_PROTECTED_TIER2_PASS_LOCAL_RIDGE_FLOOR_NOT_PRESERVED; protected Tier2 pass=True; local ridge floor preserved=False; transition=0.330428.

## F032_FLOOR_PRESERVING_JEPA_RESIDUAL_CALIBRATION: Train-only floor-preserving residual calibration for F030/F031

Decision=F032_NONZERO_RESIDUAL_DISCARDED_HELDOUT_BELOW_LOCAL_FLOOR; selected scale mean=0.190000; selected transition=0.336264; floor transition=0.333054.

## F033_CONSERVATIVE_FLOOR_GATE: Conservative train-only gate for JEPA residual above local floor

Decision=F033_CONSERVATIVE_GATE_ZERO_FALLBACK; nonzero fraction=0.000; heldout preserved=True; selected transition=0.333054.

## F034_OPERATOR_INITIALIZED_TRANSITION_HEAD: Floor-initialized residual transition predictor

Decision=F034_OPERATOR_INITIALIZED_RESIDUAL_DISCARDED_HELDOUT_BELOW_FLOOR; nonzero fraction=1.000; selected transition=0.339932; floor transition=0.333054.

## F035_INNER_VAL_OPERATOR_GATE: Inner train-validation gate for floor-initialized operator head

Decision=F035_INNER_VAL_GATE_STILL_HELDOUT_BELOW_FLOOR; nonzero fraction=1.000; heldout preserved=False; selected transition=0.340385.

## F036_RETRIEVAL_FAILURE_LOCALIZATION: Retrieval margin and nearest-neighbor failure audit after F035

Decision=F036_RECALL_FAILURE_PRIMARILY_MARGIN_INSTABILITY; broken rows=7.0; broken near-tie fraction=1.000; mean transition gap=0.007331.

## F037_RETRIEVAL_MARGIN_GATE: Train-only retrieval-margin gate for operator residuals

Decision=F037_RETRIEVAL_MARGIN_GATE_STILL_HELDOUT_BELOW_FLOOR; nonzero fraction=0.800; heldout preserved=False; broken rows=5.

## F038_ORACLE_SAFE_SCALE_CAPACITY: Held-out oracle residual safe-scale capacity audit

Decision=F038_ORACLE_SAFE_NONZERO_SCALE_CAPACITY_EXISTS; nonzero fraction=1.000; mean transition gap=0.004624.

## F039_TRAIN_PROXY_SAFE_SCALE: Train-only proxy for oracle safe residual scale

Decision=F039_TRAIN_PROXY_ZERO_FALLBACK; nonzero fraction=0.000; heldout preserved=True; transition gap=0.000000.

## F040_RETRIEVAL_AWARE_TRANSITION_HEAD: Retrieval-aware floor-initialized transition head

Decision=F040_RETRIEVAL_AWARE_ZERO_FALLBACK; nonzero fraction=0.000; heldout preserved=True; transition gap=0.000000.

## F041_CONSTRAINT_ABLATION: Train-only residual scale constraint ablation

Decision=F041_SIMPLE_TRAIN_PROXY_ABLATION_RECOVERS_SAFE_NONZERO_DIAGNOSTIC; selected rule=small_cap_0p05_continuous; nonzero fraction=1.000; heldout safe all=True.

## F042_FRESH_SMALL_CAP_VALIDATION: Fresh-seed validation of tiny residual cap

Decision=F042_FRESH_SMALL_CAP_VALIDATION_PASS_DIAGNOSTIC; nonzero fraction=1.000; heldout preserved=True; transition gap=0.001529.

## F043_ACTION_ADALN_SMALL_CAP: Action-AdaLN tiny-cap calibration

Decision=F043_ACTION_ADALN_SMALL_CAP_TIER1_PASS_DIAGNOSTIC; nonzero=1/5; floor gap transition=-0.000019; recall gap=0.000000.

## F044_ACTION_ADALN_FAILURE_AUDIT: Action-AdaLN tiny-cap failure localization

Decision=F044_ACTION_ADALN_FAILURE_LOCALIZED_TO_UNSTABLE_NO_ACTION_SIGNAL; nonzero=1; below-floor nonzero=1; strict LCB nonzero=0.

## F045_ACTIVE_OPERATOR_WRAPPER: Exact floor-initialized operator wrapper on active multiseed latents

Decision=F045_ACTIVE_OPERATOR_WRAPPER_HELDOUT_BELOW_FLOOR; nonzero fraction=1.000; floor preserved=False; transition gap=-0.000171.

## F046_OPERATOR_MISMATCH_AUDIT: Active operator train-heldout mismatch audit

Decision=F046_OPERATOR_TRAIN_HELDOUT_MISMATCH_DOMINATES; any mismatch=1.000; train/eval transition gap=0.000702/-0.000171.

## F047_TARGET_GEOMETRY_AUDIT: Target/action support geometry audit after operator mismatch

Decision=F047_HELDOUT_EXACT_ACTION_SUPPORT_GAP_CONFIRMED; unseen perturbations=1.000; exact action match=0.000; residual support=0.734.

## F048_NON_EXACT_ACTION_CONTRACT: Non-exact program action contract diagnostic

Decision=F048_NON_EXACT_ACTION_OPERATOR_STILL_BELOW_FLOOR; nonzero fraction=1.000; local floor preserved=False; selected gap transition=0.000071.

## F049_FLOOR_FEATURE_CONTRACT: Protected full-ridge floor feature contribution audit

Decision=F049_FLOOR_FEATURE_CONTRACT_AMBIGUOUS; no-exact transition gap=0.000000; exact contribution fraction=0.000.

## F050_SOURCE_ACTION_CONTRACT_SYNTHESIS: Synthesis of F047-F049 source/action contract evidence

Decision=F050_PROTECTED_FLOOR_IS_COVARIATE_ADJUSTED_SOURCE_DOMINATED; program gap=-0.008897; adjusted-source gap=0.000819.

## F051_DESCRIPTOR_ALIGNED_REPLICATION: Fresh descriptor-aligned action contract replication

Decision=F051_DESCRIPTOR_ALIGNED_ACTION_CONTRACT_HELDOUT_BELOW_FLOOR; nonzero fraction=1.000; heldout preserved=False; transition gap=-0.000008.

## F052_DESCRIPTOR_NEAR_TIE_AUDIT: Descriptor-aligned near-tie retrieval audit

Decision=F052_DESCRIPTOR_FAILURE_IS_NEAR_TIE_RETRIEVAL_MARGIN; broken rows=1/81; near-tie fraction=1.000; recommendation=Implement a train-only retrieval-margin safety gate for the descriptor-aligned contract: residual scale must default to zero unless train-internal near-tie margin diagnostics prove no local floor flip risk.

## F053_DESCRIPTOR_MARGIN_GATE: Train-only descriptor margin gate

Decision=F053_DESCRIPTOR_MARGIN_GATE_ZERO_FALLBACK_INSUFFICIENT_TRAIN_DIAGNOSTICS; selected_scale=0.000000; certified_nonzero=0; reason=No nonzero scale had complete train-only near-tie lower-tail diagnostics plus nonnegative continuous and retrieval-margin evidence.

## F054_DESCRIPTOR_MARGIN_RERUN: Fresh descriptor margin-gated JEPA rerun

Decision=F054_DESCRIPTOR_MARGIN_RERUN_ZERO_FALLBACK; nonzero fraction=0.000; heldout preserved=True; transition gap=0.000000.

## F055_ROWWISE_ABSTENTION: Fresh row-wise residual abstention diagnostic

Decision=F055_ROWWISE_ABSTENTION_SAFE_NONZERO_DIAGNOSTIC; active fraction=0.296; transition gap=0.001195; heldout preserved=True.

## F056_ROWWISE_ABSTENTION_REPLICATION: Independent row-wise residual abstention replication

Decision=F056_ROWWISE_ABSTENTION_REPLICATES_SAFE_NONZERO; active fraction=0.309; transition gap=0.000626; heldout preserved=True.

## F057_ROWWISE_SYNTHESIS: Six-seed row-wise abstention synthesis

Decision=F057_ROWWISE_ABSTENTION_READY_FOR_TIER2_DESIGN; seeds=6; transition gap=0.000910; delta gap=0.001660; broken=0.

## F058_ROWWISE_TIER2: Promotion-ineligible row-wise abstention Tier 2 validation

Decision=F058_ROWWISE_TIER2_SAFE_BUT_WEAK; seeds=6; transition gap=0.000086; delta gap=0.001408; broken=0.

## F059_ROWWISE_STABILITY_AUDIT: Read-only row-wise Tier 2 transition stability audit

Decision=F059_TIER2_WEAKNESS_IS_TRANSITION_GENERALIZATION_NOISE; negative transition seeds=1; transition gap mean=0.000086; std=0.000274.

## F060_FLOOR_DIRECTION_PROJECTION: Floor-direction-projected row-wise residual diagnostic

Decision=F060_FLOOR_DIRECTION_PROJECTION_SAFE_BUT_WEAK; transition gap=0.000022; min transition gap=-0.000020; broken=0.

## F061_PROJECTION_COLLAPSE_AUDIT: Read-only floor-direction projection collapse audit

Decision=F061_PROJECTION_KILLS_DELTA_WITHOUT_STABILIZING_TRANSITION; F060 transition gap=0.000022; F060 delta gap=0.000000.

## F062_ORACLE_RESIDUAL_CONE: Held-out oracle residual-cone capacity diagnostic

Decision=F062_ORACLE_CONE_CAPACITY_EXISTS_ALL_SEEDS; nonzero fraction=1.000; transition gap=0.000323; delta gap=0.000906.

## F063_TRAIN_CONE_SELECTOR: Train-only residual-cone selector diagnostic

Decision=F063_TRAIN_CONE_SELECTOR_HELDOUT_BELOW_FLOOR; cone_mix=0.083; transition gap=-0.000882; delta gap=-0.000082.

## F064_TRAIN_HELDOUT_SELECTOR_AUDIT: Train-versus-held-out selector mismatch audit

Decision=F064_TRAIN_SELECTOR_OVERFITS_CONTINUOUS_TRANSITION_GAINS; train transition gap=0.001412; held-out transition gap=-0.000882; train-positive/heldout-negative transition seeds=1.

## F065_ACTION_HELDOUT_CALIBRATION: Train-only action-heldout calibration gate

Decision=F065_ACTION_HELDOUT_GATE_HELDOUT_BELOW_FLOOR; inner_safe_nonzero_fraction=1.000; heldout transition gap=-0.000873; residual scale=0.020988.

## F066_ACTION_HELDOUT_MISMATCH_AUDIT: Action-heldout calibration mismatch audit

Decision=F066_TRAIN_ACTION_HELDOUT_STILL_OPTIMISTIC_FOR_REAL_HELDOUT; inner transition gap=0.003880; real transition gap=-0.000873; support gap=-0.013426.

## F067_RESIDUAL_SELECTOR_CLOSURE: Residual-selector family closure synthesis

Decision=F067_CLOSE_RESIDUAL_SELECTOR_FAMILY_PIVOT_TO_DATA_CONTRACT; oracle_capacity=1.0; train-only below-floor count=2.0; calibration mismatch count=2.0.

## F068_DATA_CONTRACT_CALIBRATION: Data-contract calibration redirect diagnostic

Decision=F068_TRAIN_REAL_MISMATCH_PERSISTS_ACROSS_SPLIT_CONTRACTS; current abs optimism=0.026180; best alternative=stratified_program (0.064983).

## F069_REPRESENTATION_NOISE_AUDIT: Representation noise amplification audit

Decision=F069_OBSERVED_RNA_REPRESENTATION_AMPLIFIES_CALIBRATION_MISMATCH; observed-true delta optimism=0.037552; ratio=5.168.

## F070_DENOISED_REPRESENTATION_CEILING: Denoised representation ceiling diagnostic

Decision=F070_TRUE_ZBIO_ONLY_REDUCES_MISMATCH_RNA_OBSERVATION_LIMIT; best_candidate=observed_counts_log_train_pca; observed_delta_optimism=0.088751; candidate_delta_optimism=0.088751.

## F071_ZBIO_OBSERVABILITY_AUDIT: Train-only z_bio observability diagnostic

Decision=F071_CROSS_MODAL_ZBIO_RECOVERY_REDUCES_MISMATCH; best=image_to_zbio_ridge; best_z_cos=0.9992; best_delta_optimism=0.052551.

## F072_IMAGE_LATENT_CEILING_AUDIT: Non-oracle image latent ceiling diagnostic

Decision=F072_IMAGE_LATENT_NONORACLE_REDUCES_MISMATCH; image_delta_optimism=0.052534; rna_to_image_cos=0.5886.

## F073_IMAGE_TEACHER_RNA_STUDENT_JEPA: Tiny image-teacher RNA-student representation repair

Decision=F073_RNA_STUDENT_IMAGE_TEACHER_WEAK_OR_UNSTABLE; image_cos=0.7021; candidate_delta_optimism=0.074547.

## F074_IMAGE_TEACHER_FAILURE_LOCALIZATION: Image-teacher RNA-student failure localization

Decision=F074_FAILURE_IS_PERTURBATION_STRUCTURE_LOSS; image_cos=0.7021; perturbation_probe_gap=-0.3457.

## F075_RELATIONAL_IMAGE_TEACHER_RNA_STUDENT: Relational image-teacher RNA-student representation repair

Decision=F075_RELATIONAL_IMAGE_TEACHER_STILL_LOSES_STRUCTURE; image_cos=0.6962; recall_gap_vs_image=-0.2881; probe_gap_vs_f073=0.0077.

## F076_CONDITION_CENTROID_IMAGE_TEACHER: Train-only condition-centroid image-teacher RNA-student repair

Decision=F076_CONDITION_CENTROID_TEACHER_STILL_LOSES_STRUCTURE; image_cos=0.7019; recall_gap_vs_image=-0.3169; probe_gap_vs_f075=-0.0077.

## F077_SOURCE_PROGRAM_IMAGE_TEACHER: Source-state plus coarse program-action image-teacher probe

Decision=F077_SOURCE_PROGRAM_ACTION_REPAIRS_STRUCTURE_BUT_WEAK; image_cos=0.8078; recall_gap_vs_f076=0.3169; probe_gap_vs_f076=-0.1034.

## F078_SOURCE_PROGRAM_FAILURE_LOCALIZATION: Source-program image-teacher failure localization

Decision=F078_SOURCE_PROGRAM_RECALL_WITH_LOW_TRANSITION_AND_ACTION_SIGNAL; recall_gap=0.0000; transition_gap=-0.2261; probe_gap=-0.4491.

## F079_SOURCE_DELTA_RANK_REPAIR: Source-program delta/rank repair

Decision=F079_SOURCE_DELTA_RANK_REPAIR_READY_FOR_WRAPPER; transition=0.1375; delta=0.9914; rank=6.6257.

## F080_SOURCE_DELTA_RANK_JEPA_WRAPPER: Full real-JEPA wrapper around F079 source/delta/rank repair

Decision=F080_FULL_JEPA_WRAPPER_MIXED_OR_INCONCLUSIVE; image_teacher_transition=0.2091; delta=0.7821; cross_modal_min=0.5267.

## F081_DELTA_CALIBRATED_JEPA_WRAPPER: Train-only delta-calibrated full JEPA wrapper

Decision=F081_DELTA_CALIBRATED_JEPA_TIER1_PASS_NONPROMOTING; calibrated_transition=0.2296; calibrated_delta=0.9539; raw_delta=0.7699.

## F082_DELTA_CALIBRATED_TIER2_VALIDATION: Fresh-seed validation of F081 delta-calibrated JEPA

Decision=F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN; mean_transition=0.2078; mean_delta=0.9344; min_delta=0.9047.

## F083_TIER3_PREFLIGHT: Tier 3 no-regression preflight for delta-calibrated JEPA

Decision=F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS; transition_gap=0.2021; delta_gap=0.5364; rank_gap=-3.2596; paired_external_validator=False.

## F084_RANK_FLOOR_COMPARABILITY: Rank-floor comparability audit

Decision=F084_LOCKED_RANK_FLOOR_EXCEEDS_IMAGE_TEACHER_CEILING; calibrated_rank=7.0239; target_image_rank=7.2667; locked_floor=10.2835.

## F085_CURRENT_LATENT_FLOOR_REGISTRY: Current image-teacher latent floor registry

Decision=F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE; current_floor_transition=0.2093; candidate_transition=0.2078; true_delta_rank=7.5992.

## F086_CURRENT_REGISTRY_TIER3_DESIGN: Current-registry Tier 3 design and validator preflight

Decision=F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED; synthetic_current_registry_gates_pass=True; local_paired_validator_available=False; future_paired_validator_candidate_found=True.

## F087_SCGENESCOPE_ADAPTER_PREFLIGHT: scGeneScope metadata adapter preflight

Decision=F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL; root_exists=False; manifest_valid=False; paired_condition_count=0.

## F088_SCGENESCOPE_REMOTE_DISCOVERY: Remote scGeneScope metadata discovery

Decision=F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE; entries=47; light_metadata=0; smallest_paired_feature_gb=13.590; download_attempted=False.

## F089_SCGENESCOPE_SUPPLEMENT_HARVEST: Official scGeneScope supplement metadata harvest

Decision=F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED; supplement_bytes=2056507; croissant_found=True; split_contract=True; payload_download_attempted=False.

## F090_SCGENESCOPE_CROISSANT_CONTRACT: scGeneScope Croissant adapter contract validation

Decision=F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE; manifest_rows=40; ready_pairs=16/16; payload_download_attempted=False.

## F091_SCGENESCOPE_FEATURE_PREFLIGHT: scGeneScope feature-size, storage, RAM, and backed-IO preflight

Decision=F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN; paired_feature_gb=13.590; storage_gate_pass=True; backed_io_gate_pass=True; payload_download_attempted=False.

## F092_SCGENESCOPE_OBS_ONLY_DRY_RUN: scGeneScope smallest paired feature backed obs dry run

Decision=HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED; approved payloads downloaded but writing status failed with disk quota exceeded; removed `data/raw/scgenescope`; no training, fitting, or model promotion occurred.

Post-escalation safeguard: F092 retry is now blocked while `HARD_ESCALATION_REPORT.md` exists unless `--acknowledge-hard-escalation-retry` is supplied, and future F092 attempts reserve status-write space before payload download.

Recovery artifact written: `SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md`; the loop remains halted until a smaller manifest-backed subset or explicit quota-safe retry evidence exists.
