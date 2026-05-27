# Total-Autonomy BioGuard-WM-JEPA Autoresearch

AUTONOMY_MODE = CONTINUOUS_DEBATE_COUNCIL

## Session Amendments



# Session Amendment 001: Metric/Data Redesign Before More Residuals

## Trigger
`BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN`

## Evidence
Action-AdaLN + RoPE preserved the floor but train-only calibration selected zero residual.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only spectral, kernel, program, and action-AdaLN predictors are cooled until a diagnostic identifies residual headroom.

## New Or Reopened Family
Family F: Metric And Data Redesign.

## Exact Next Experiment
`F001_BOOTSTRAP_FLOOR_NOISE_AUDIT`

Continuation already executed under this amendment:
`F002_HELDOUT_DIFFICULTY_STRATIFICATION`

## Implementation Tasks
- Bootstrap the protected floor metrics on held-out perturbation scoring.
- Estimate whether claimed residual effects are below evaluation noise.
- Do not modify locked splits or use eval targets for selection.

## Gates
Diagnostic only; no model promotion. Leakage and identity reports remain required.

## Do-Not-Run List
Do not force nonzero residual scale. Do not launch full JEPA wrapper until a frozen-latent residual passes calibration.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: Family F: bootstrap floor/noise audit


# Session Amendment 002: Action/Condition Heterogeneity Audit

## Trigger
`F001_F002_METRIC_HETEROGENEITY_AFTER_PHASE8_FAILURE`

## Evidence
F001 showed the protected floor transition estimate is noisy, and F002 showed floor performance differs sharply by effect-size tertile.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force a nonzero residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, action/condition heterogeneity branch.

## Exact Next Experiment
`F003_ACTION_CONDITION_HETEROGENEITY_AUDIT`

## Implementation Tasks
- Stratify held-out scoring by perturbation, condition, cell line, and batch labels.
- Use labels only for diagnostics, never as model inputs.
- Identify whether failures concentrate in specific perturbations or support cells.

## Gates
Diagnostic only; no model promotion. Leakage and identity reports remain required.

## Do-Not-Run List
Do not launch more residual architecture until heterogeneity is explained.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F003 action/condition heterogeneity audit



# Session Amendment 003: Action Support Distance Audit

## Trigger
`F003_ACTION_HETEROGENEITY_IDENTIFIED`

## Evidence
F003 showed one held-out perturbation group is negative under the protected floor while two held-out groups are positive.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force a nonzero residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, action descriptor support branch.

## Exact Next Experiment
`F004_ACTION_SUPPORT_DISTANCE_AUDIT`

## Implementation Tasks
- Compute nearest train action-descriptor support for each held-out perturbation.
- Relate train-support distance to floor transition performance.
- Use action labels only for diagnostics.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch GEARS-style descriptors or per-action gates until support distance is measured.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F004 action support distance audit



# Session Amendment 004: Train-Only Leave-Action-Out Floor Audit

## Trigger
`F004_SUPPORT_DISTANCE_DOES_NOT_EXPLAIN_NEGATIVE_ACTION`

## Evidence
F004 showed descriptor support distance is identical for held-out perturbations and does not explain the negative perturbation group.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Action descriptor engineering is deferred until train-only generalization failures are quantified.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only action generalization audit.

## Exact Next Experiment
`F005_TRAIN_ONLY_LEAVE_ACTION_OUT_FLOOR_AUDIT`

## Implementation Tasks
- Leave each train perturbation group out.
- Fit the ridge floor on remaining train rows only.
- Score the left-out train action group.
- Use no eval/test target rows for fitting or selection.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch new residual or graph-action model until this audit is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F005 train-only leave-action-out floor audit



# Session Amendment 005: Ridge Ensemble Uncertainty Audit

## Trigger
`F005_TRAIN_ONLY_ACTION_FLOOR_FAILURES_CONFIRMED`

## Evidence
F005 showed train-only leave-action-out ridge floors can be negative for some action groups, so a safe candidate needs a train-only way to detect floor failure risk.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, uncertainty/abstention feasibility branch.

## Exact Next Experiment
`F006_RIDGE_ENSEMBLE_UNCERTAINTY_AUDIT`

## Implementation Tasks
- Fit ridge floor bootstrap ensemble on train rows only.
- Compute held-out prediction disagreement as a diagnostic uncertainty signal.
- Compare uncertainty to held-out floor failures for analysis only.

## Gates
Diagnostic only; no model promotion and no eval-driven gate deployment.

## Do-Not-Run List
Do not deploy an abstention gate unless train-only calibration can select it.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F006 ridge ensemble uncertainty audit



# Session Amendment 006: Replicate Ceiling Audit

## Trigger
`F006_UNCERTAINTY_DOES_NOT_SEPARATE_FLOOR_FAILURE`

## Evidence
F006 showed bootstrap ridge ensemble uncertainty does not isolate the negative held-out perturbation, so simple uncertainty abstention is not yet justified.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and uncertainty-gated deployment remain cooled.

## New Or Reopened Family
Family F: Metric And Data Redesign, replicate/technical ceiling branch.

## Exact Next Experiment
`F007_REPLICATE_CEILING_AUDIT`

## Implementation Tasks
- Measure condition-level target and delta replicate consistency.
- Compare replicate ceilings with action-level floor failures.
- Use eval rows only for diagnostic scoring, not fitting or selection.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch representation repair until replicate ceiling is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F007 replicate ceiling audit



# Session Amendment 007: Representation/Error Audit

## Trigger
`F007_REPLICATE_CEILING_DOES_NOT_EXPLAIN_ACTION_FAILURE`

## Evidence
F001-F007 show the residual failure is action-specific and not explained by action descriptor support, bootstrap uncertainty, or replicate ceiling.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and action-support-only fixes remain cooled.

## New Or Reopened Family
Family C: Representation Repair Before Operator Learning.

## Exact Next Experiment
`C001_REPRESENTATION_ERROR_AUDIT`

## Implementation Tasks
- Audit online/teacher consistency.
- Audit z_bio target/delta/floor-error ranks.
- Decompose floor error by held-out action.
- Check whether floor error couples to z_tech movement.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not retrain residuals until representation error modes are localized.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C001 representation/error audit



# Session Amendment 008: Latent Delta Manifold Support Audit

## Trigger
`C001_ACTION9_HIGH_FLOOR_ERROR_WITH_HEALTHY_RANK`

## Evidence
C001 showed the worst action has high floor error even though online/teacher consistency and latent rank are healthy.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and simple support-distance fixes remain cooled.

## New Or Reopened Family
Family C: Representation Repair, latent response manifold branch.

## Exact Next Experiment
`C002_LATENT_DELTA_MANIFOLD_SUPPORT_AUDIT`

## Implementation Tasks
- Compute train action mean teacher deltas.
- Compare held-out action mean deltas to train delta manifold.
- Relate delta-manifold support to action-level floor failure.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not train new architecture until response support is measured.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C002 latent delta manifold support audit



# Session Amendment 009: Train-Only Delta Support Audit

## Trigger
`C002_WEAK_DELTA_MANIFOLD_SUPPORT_FOR_BAD_ACTION`

## Evidence
C002 showed the weakest held-out action also had the weakest cosine support from train action mean deltas.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and simple uncertainty gates remain cooled.

## New Or Reopened Family
Family C: Representation Repair, train-only delta support validation.

## Exact Next Experiment
`C003_TRAIN_ONLY_DELTA_SUPPORT_AUDIT`

## Implementation Tasks
- Leave out each train perturbation action.
- Compute nearest remaining train action delta support.
- Fit ridge on remaining train rows only and score the left-out train action.
- Test whether weak delta support predicts floor failure.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch graph/action descriptor architecture until this train-only validation is logged.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C003 train-only delta support audit



# Session Amendment 010: Action Prior Sufficiency Audit

## Trigger
`C003_TRAIN_ONLY_DELTA_SUPPORT_PREDICTS_FAILURE`

## Evidence
C003 validated train-only that weak latent delta support predicts action-ridge floor failures.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and simple support-distance fixes remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, diagnostic baseline branch.

## Exact Next Experiment
`E001_ACTION_PRIOR_SUFFICIENCY_AUDIT`

## Implementation Tasks
- Compare source-only, action-only, and source+action ridge baselines.
- Use train-only leave-action-out scoring for model-choice evidence.
- Also score held-out perturbations as diagnostics only.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not implement graph/program descriptors until this audit shows action-prior insufficiency or source-only dominance.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E001 action-prior sufficiency audit



# Session Amendment 011: Train-Only Source/Action Blend Audit

## Trigger
`E001_SOURCE_ONLY_TRANSITION_IMPROVES_BUT_RECALL_COLLAPSES`

## Evidence
E001 showed source-only improves transition and delta cosine but collapses retrieval, while source+action preserves recall but has lower transition gain.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and action-only descriptor fixes remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, safe floor-composition diagnostic.

## Exact Next Experiment
`E002_TRAIN_ONLY_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Blend source-only and source+action ridge floors.
- Select blend weight by train-only leave-action-out calibration.
- Evaluate held-out perturbations only after selection.

## Gates
Diagnostic only; no model promotion and no replacement of protected floor.

## Do-Not-Run List
Do not use held-out rows for selecting blend weight.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E002 train-only source/action floor blend calibration



# Session Amendment 012: Prototype Transition Diagnostic

## Trigger
`E002_BLEND_HELDOUT_RECALL_DROP`

## Evidence
E002 found a train-only selected source/action blend improves transition and delta cosine but still drops held-out recall below the protected floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and simple floor-blend deployment remain cooled.

## New Or Reopened Family
Family D: Population Transport JEPA, prototype diagnostic branch.

## Exact Next Experiment
`D001_PROTOTYPE_TRANSITION_DIAGNOSTIC`

## Implementation Tasks
- Build train-only source-state prototypes.
- Fit prototype+action ridge transition on train rows.
- Score held-out perturbations and action heterogeneity.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not call this a JEPA candidate or promote it.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: D001 prototype transition diagnostic



# Session Amendment 013: Condition-Level Blend Calibration

## Trigger
`D001_PROTOTYPE_TRANSITION_BELOW_FLOOR`

## Evidence
D001 prototype transition diagnostic fell below the protected full-ridge transition floor and reduced delta rank, so prototype transport is not a safe continuation path under the current synthetic benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, simple source/action blend deployment, and prototype transition are cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, stricter condition-level blend calibration diagnostic.

## Exact Next Experiment
`E003_CONDITION_LEVEL_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Use train-only condition/action grouped folds for source/action blend selection.
- Require nonnegative fold-level recall preservation and near-nonnegative delta cosine preservation before selecting a nonzero source-only component.
- Evaluate held-out perturbations only after train-only selection.

## Gates
Diagnostic only; no model promotion and no replacement of the protected transition floor.

## Do-Not-Run List
Do not select blend weight from held-out perturbation rows or promote an eval-only improvement.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E003 condition-level source/action blend calibration



# Session Amendment 014: Support-Aware Blend Calibration

## Trigger
`E003_CONDITION_BLEND_REJECTED_KEEP_FULL_FLOOR`

## Evidence
E003 showed that nonzero source/action blending can improve mean train-only transition metrics but violates worst-case fold preservation. C003 showed weak latent delta-manifold support predicts floor failures, so the next test must use a deployable support score rather than held-out targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global source/action blending, and prototype transition remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, support-aware safe floor-composition diagnostic.

## Exact Next Experiment
`E004_SUPPORT_AWARE_SOURCE_ACTION_BLEND_AUDIT`

## Implementation Tasks
- Compute support from predicted deltas against train-only delta references.
- Select a support threshold and source-only blend weight by train-only leave-action-out calibration.
- Default to the protected full-ridge floor unless train-only gates preserve recall and delta-cosine worst cases.

## Gates
Diagnostic only; no model promotion and no replacement of the protected transition floor.

## Do-Not-Run List
Do not use held-out target deltas, condition keys, or exact target means for support scoring or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E004 support-aware source/action blend calibration



# Session Amendment 015: Predicted Support Score Audit

## Trigger
`E004_SUPPORT_AWARE_BLEND_HELDOUT_BELOW_FLOOR`

## Evidence
E004 selected a nonzero support-aware source/action blend by train-only calibration, but the held-out score still dropped below the protected full-ridge floor on recall. The support score may not distinguish held-out unsafe actions.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, support-aware blending, and prototype transition remain non-promotable under current evidence.

## New Or Reopened Family
Family E: Program And Graph Action Priors, predicted-support failure localization.

## Exact Next Experiment
`E005_PREDICTED_SUPPORT_SCORE_AUDIT`

## Implementation Tasks
- Compare deployable predicted-delta support against teacher-delta support as an audit only.
- Measure whether predicted support correlates with held-out action transition success.
- Use the result to choose between representation-support repair and synthetic benchmark geometry expansion.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not use teacher support for model selection or held-out gating.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E005 predicted support score failure audit



# Session Amendment 016: Support Threshold Oracle Audit

## Trigger
`E005_PREDICTED_SUPPORT_OVERCONFIDENT_ON_BAD_ACTION`

## Evidence
E005 found predicted support is correlated with held-out transition success but remains overconfident on the worst action. Before designing a new calibration mechanism, test whether this support-gated source/action family has any held-out capacity at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, and the current support-aware selected candidate remain non-promotable.

## New Or Reopened Family
Family E: Program And Graph Action Priors, support-threshold oracle capacity audit.

## Exact Next Experiment
`E006_HELDOUT_SUPPORT_THRESHOLD_ORACLE_AUDIT`

## Implementation Tasks
- Sweep support thresholds and source-only blend weights on held-out rows for audit only.
- Report whether any candidate could preserve the protected floor on transition, recall, and delta cosine.
- Do not use the oracle-selected candidate as a promoted model or calibration result.

## Gates
Diagnostic only. If no oracle candidate preserves the floor, retire this support-gated blend family.

## Do-Not-Run List
Do not use this held-out oracle for model selection, training, or promotion.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E006 held-out support-threshold oracle audit



# Session Amendment 017: High-Support Abstention Calibration

## Trigger
`E006_ORACLE_SUPPORT_GATE_HAS_CAPACITY_CALIBRATION_FAILURE`

## Evidence
E006 showed support-threshold gating has held-out oracle capacity, but only in a high-threshold, low-active-fraction regime. Because E006 used held-out labels, the next run must use train-only calibration and remain diagnostic only.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, and low-threshold support blending remain non-promotable.

## New Or Reopened Family
Family E: Program And Graph Action Priors, high-support abstention calibration diagnostic.

## Exact Next Experiment
`E007_HIGH_SUPPORT_ABSTENTION_CALIBRATION_DIAGNOSTIC`

## Implementation Tasks
- Select only high support thresholds with low active fraction using train-only leave-action-out folds.
- Require recall preservation and bounded transition/delta worst-case regressions on train folds.
- Evaluate held-out rows after selection, but mark any pass as oracle-informed diagnostic only.

## Gates
Diagnostic only; no model promotion because this family was designed after an E006 held-out oracle audit.

## Do-Not-Run List
Do not promote E007 or use it as a model of record without a fresh split/Tier 2 amendment.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E007 high-support abstention calibration



# Session Amendment 018: Nested Train-Only Calibration Audit

## Trigger
`E007_HIGH_SUPPORT_CALIBRATION_HELDOUT_RECALL_DROP`

## Evidence
E007 still dropped held-out recall, so its train-only calibration may be unstable rather than merely too conservative or too weak.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only and support-gated source/action blending remain non-promotable on the current held-out split.

## New Or Reopened Family
Family E: Program And Graph Action Priors, nested train-only calibration stability audit.

## Exact Next Experiment
`E008_NESTED_TRAIN_ONLY_HIGH_SUPPORT_CALIBRATION_AUDIT`

## Implementation Tasks
- Use outer train perturbations as pseudo-heldout groups.
- Select the high-support rule only from inner train perturbations.
- Score the selected rule on the outer train perturbation and compare to that fold's full-ridge floor.

## Gates
Diagnostic only. Failure means the support-gated blend family lacks stable train-only calibration under current data.

## Do-Not-Run List
Do not use held-out eval rows in this audit.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E008 nested train-only high-support calibration audit



# Session Amendment 019: Train-Only Heldout-Set Geometry Audit

## Trigger
`E008_NESTED_HIGH_SUPPORT_CALIBRATION_FAILED_TRAIN_ONLY`

## Evidence
E008 failed nested train-only calibration: support-gated blending selected nonzero rules in all outer folds but had negative mean outer transition gap and a negative delta-cosine worst case. This closes the current support-gated blend family under the present data geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated source/action blending, global blending, prototype transport, and residual-only predictor heads remain non-promotable on this split.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only heldout-set geometry audit.

## Exact Next Experiment
`F008_TRAIN_ONLY_HELDOUT_SET_GEOMETRY_AUDIT`

## Implementation Tasks
- Enumerate train-only pseudo-heldout triplets of perturbations.
- Refit the full action-ridge floor on remaining train perturbations.
- Score pseudo-heldout triplets and relate failures to train delta support.
- Use this only to decide whether benchmark/split geometry needs redesign before more JEPA candidates.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not use held-out eval rows in this audit and do not select a model from these pseudo-heldout sweeps.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F008 train-only heldout-set geometry audit



# Session Amendment 020: Support-Threshold Split Redesign Audit

## Trigger
`F008_SPLIT_GEOMETRY_UNSTABLE_REDESIGN_BENCHMARK_BEFORE_MORE_CANDIDATES`

## Evidence
F008 found train-only pseudoheldout geometry instability: several pseudoheldout triplets had negative transition and nearly all triplets fell below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending and residual-only operator searches remain cooled until benchmark geometry is clarified.

## New Or Reopened Family
Family F: Metric And Data Redesign, support-threshold split redesign audit.

## Exact Next Experiment
`F009_SUPPORT_THRESHOLD_SPLIT_REDESIGN_AUDIT`

## Implementation Tasks
- Sweep train-only support-score thresholds over F008 pseudoheldout triplets.
- Identify whether support thresholds can eliminate negative transition without destroying recall.
- Use the result to decide between benchmark redesign and retrieval-label metric audit.

## Gates
Diagnostic only; no model promotion and no mutation of the locked held-out split.

## Do-Not-Run List
Do not use these thresholds to select a candidate model on the existing test split.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F009 support-threshold split redesign audit



# Session Amendment 021: Retrieval Label Granularity Audit

## Trigger
`F009_SUPPORT_THRESHOLDS_REMOVE_NEGATIVE_TRANSITION_NOT_RECALL`

## Evidence
F009 showed train-only support thresholds can eliminate negative transition pseudoheldout triplets but cannot preserve the current condition-level recall floor. This suggests the retrieval metric or representation dominance may be limiting progress.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending and residual-only transition heads remain cooled.

## New Or Reopened Family
Family R: Retrieval And Metric Redesign, label granularity audit.

## Exact Next Experiment
`R001_RETRIEVAL_LABEL_GRANULARITY_AUDIT`

## Implementation Tasks
- Recompute train-only pseudoheldout retrieval under condition, condition+batch, perturbation, cell-line, and batch labels.
- Determine whether recall failure is label-granularity, perturbation discrimination, or source/cell-line dominance.
- Do not replace protected gates; document diagnostic evidence only.

## Gates
Diagnostic only; no model promotion and no benchmark mutation.

## Do-Not-Run List
Do not promote on alternative retrieval labels.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: R001 retrieval label granularity audit



# Session Amendment 022: Cell-Line Source-Dominance Audit

## Trigger
`R001_RETRIEVAL_DOMINATED_BY_CELL_LINE_NOT_PERTURBATION`

## Evidence
R001 showed pseudoheldout retrieval is much stronger for cell-line labels than perturbation or condition labels. The next diagnostic must quantify whether latent endpoints and predicted deltas are dominated by source/cell-line structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending, residual-only transition heads, and metric relabeling are not promotable.

## New Or Reopened Family
Family C: Representation Repair, cell-line/source-dominance audit.

## Exact Next Experiment
`C004_CELL_LINE_SOURCE_DOMINANCE_AUDIT`

## Implementation Tasks
- Use train-only pseudoheldout predictions.
- Measure eta-squared and nearest-centroid predictability for cell line, perturbation, and batch across source, target, true delta, predicted endpoint, and predicted delta.
- Decide whether the next mechanism should remove source/cell-line dominance or redesign retrieval targets.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not fit or select using held-out eval rows.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C004 cell-line/source-dominance representation audit



# Session Amendment 023: Perturbation-Centered Delta Retrieval Audit

## Trigger
`C004_CELL_LINE_SOURCE_DOMINANCE_CONFIRMED_PERTURBATION_DELTA_UNDERREPRESENTED`

## Evidence
C004 confirmed predicted endpoints are dominated by cell-line/source structure, while true deltas are more perturbation-structured than predicted deltas.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending, metric relabeling, and endpoint-only residual corrections remain cooled.

## New Or Reopened Family
Family C: Representation Repair, perturbation-centered delta retrieval audit.

## Exact Next Experiment
`C005_PERTURBATION_CENTERED_DELTA_RETRIEVAL_AUDIT`

## Implementation Tasks
- Compare endpoint retrieval with delta-space retrieval on train-only pseudoheldout triplets.
- Score perturbation, condition, and cell-line labels without replacing protected gates.
- Decide whether a future candidate should add a perturbation-centered delta retrieval objective.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a new JEPA head until delta-space retrieval evidence is recorded.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C005 perturbation-centered delta retrieval audit



# Session Amendment 024: Endpoint+Delta Composite Retrieval Audit

## Trigger
`C005_DELTA_SPACE_PARTIALLY_REPAIRS_PERTURBATION_RECALL_BUT_BELOW_FLOOR`

## Evidence
C005 showed delta-space retrieval improves perturbation recall relative to endpoint retrieval, but remains below the protected condition-level recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Endpoint-only, residual-only, and support-gated transition modifications remain cooled.

## New Or Reopened Family
Family C: Representation Repair, endpoint+delta composite retrieval audit.

## Exact Next Experiment
`C006_ENDPOINT_DELTA_COMPOSITE_RETRIEVAL_AUDIT`

## Implementation Tasks
- Combine endpoint similarity and delta similarity with fixed diagnostic weights.
- Score condition and perturbation retrieval on train-only pseudoheldout triplets.
- Decide whether a future dual-objective JEPA target is justified.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not use composite scoring to claim a promoted model.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C006 endpoint+delta composite retrieval audit



# Session Amendment 025: Nested Composite Retrieval Calibration Audit

## Trigger
`C006_COMPOSITE_RETRIEVAL_PARTIALLY_REPAIRS_CONDITION_RECALL_BELOW_FLOOR`

## Evidence
C006 showed that endpoint+delta composite scoring partially repairs condition-level retrieval, but the best diagnostic weight was selected on the same pseudoheldout triplets it scored and remained below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Endpoint-only, residual-only, support-gated transition, and uncalibrated composite scoring remain cooled.

## New Or Reopened Family
Family C: Representation Repair, train-only nested composite calibration.

## Exact Next Experiment
`C007_NESTED_COMPOSITE_RETRIEVAL_CALIBRATION_AUDIT`

## Implementation Tasks
- Select the endpoint/delta retrieval weight using only inner train-action folds.
- Score the selected rule on outer train-only pseudoheldout perturbation triplets.
- Compare selected composite scoring with endpoint-only and delta-only scoring without changing protected gates.

## Gates
Diagnostic only; no model promotion and no metric replacement. A future dual-objective JEPA candidate requires the nested rule to beat endpoint-only scoring and approach the protected condition-recall floor without held-out peeking.

## Do-Not-Run List
Do not launch a dual-objective JEPA candidate until nested composite calibration is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C007 nested composite retrieval calibration audit



# Session Amendment 026: Fresh Synthetic Seed Latent-Cache Audit

## Trigger
`C007_NESTED_COMPOSITE_CALIBRATION_PARTIAL_GAIN_BELOW_FLOOR`

## Evidence
C007 showed the endpoint+delta composite rule is train-internally calibratable and improves over endpoint-only retrieval, but it remains materially below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Uncalibrated composite scoring, residual-only predictors, and immediate dual-objective JEPA reopening are cooled until seed-specific geometry is checked.

## New Or Reopened Family
Family G: Fresh Synthetic Seed Geometry.

## Exact Next Experiment
`G001_FRESH_SYNTHETIC_SEED_LATENT_CACHE_AUDIT`

## Implementation Tasks
- Build a tiny CPU latent cache for synthetic seed 1 using the existing Phase 4 cache script.
- Reproduce the action-ridge floor on that fresh synthetic seed.
- Re-run the train-only endpoint+delta composite geometry audit on that fresh seed.
- Decide whether the current failure is seed-specific geometry, general representation saturation, or enough evidence for a JEPA target change.

## Gates
Diagnostic only; no model promotion and no metric replacement. The protected model of record and protected transition floor remain unchanged.

## Do-Not-Run List
Do not train a heavier model or use GPU for this audit.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F010 protected metric redesign memo



# Session Amendment 027: Protected Metric Disagreement Audit

## Trigger
`G001_FRESH_SYNTHETIC_SEED_CONFIRMS_COMPOSITE_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
G001 confirmed that endpoint+delta composite retrieval gives a repeatable partial repair on a fresh synthetic seed, but recall@1 remains below the protected floor even when continuous transition and delta-cosine metrics improve.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate JEPA architecture reopening remains cooled until the metric disagreement is explicitly documented. This does not alter protected gates.

## New Or Reopened Family
Family F: Metric And Data Redesign, protected metric disagreement audit.

## Exact Next Experiment
`F010_METRIC_DISAGREEMENT_AUDIT`

## Implementation Tasks
- Compare recall@1 with transition improvement and delta-cosine across current and fresh synthetic seeds.
- Quantify whether endpoint+delta composite gains replicate without reaching the protected recall floor.
- Write a report that preserves current gates but explains whether recall is acting as an unstable proxy.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a heavier model or relax the protected floor based on this audit alone.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F010 protected metric disagreement audit



# Session Amendment 028: Source-State Constrained Retrieval Audit

## Trigger
`F010_RECALL_GATE_SEED_UNSTABLE_CONTINUOUS_METRICS_DISAGREE`

## Evidence
F010 showed that recall@1 is unstable relative to continuous transition quality, while endpoint+delta composite gains replicate across the original and fresh synthetic seeds.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate heavy JEPA training remains cooled until retrieval failure is separated into cross-source-state competition versus perturbation-level separation failure.

## New Or Reopened Family
Family C: Representation Repair, source-state constrained perturbation retrieval audit.

## Exact Next Experiment
`C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_AUDIT`

## Implementation Tasks
- Restrict diagnostic retrieval scoring to candidates with the same source-state/cell-line metadata.
- Compare constrained endpoint, delta, and endpoint+delta composite retrieval.
- Run the same diagnostic on the fresh seed cache when available.
- Do not use the metadata constraint as a model input or promotion path.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state JEPA candidate until this diagnostic shows the reachable ceiling.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C008 source-state constrained perturbation retrieval audit



# Session Amendment 029: Nested Source-State Retrieval Calibration Audit

## Trigger
`C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_REACHES_RECALL_FLOOR_DIAGNOSTIC`

## Evidence
C008 reached the protected current-seed condition-recall floor only under same-source-state constrained scoring, while fresh seed scoring remained below floor. The C008 weight was selected oracle-style from the scored triplets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state JEPA training remains cooled until the constrained retrieval rule survives train-only nested calibration.

## New Or Reopened Family
Family C: Representation Repair, nested source-state constrained retrieval calibration.

## Exact Next Experiment
`C009_NESTED_SOURCE_STATE_RETRIEVAL_CALIBRATION_AUDIT`

## Implementation Tasks
- Select endpoint/delta source-state constrained retrieval weights using inner train-action folds.
- Score the selected rule on outer train-only heldout perturbation triplets.
- Repeat the same nested audit on the existing fresh synthetic seed cache when available.
- Keep source-state metadata as an evaluation constraint only, not as a model input.

## Gates
Diagnostic only; no model promotion and no metric replacement. A source-state JEPA candidate requires the nested current-seed result to preserve the protected recall floor and fresh-seed diagnostics to avoid clear collapse.

## Do-Not-Run List
Do not train a source-state JEPA candidate before C009 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C009 nested source-state retrieval calibration audit



# Session Amendment 030: Source-Latent Neighborhood Retrieval Audit

## Trigger
`C009_NESTED_SOURCE_STATE_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C009 showed exact source-state constrained retrieval survives nested calibration only as a near miss on the current seed and remains below floor on the fresh seed. Exact source-state metadata should not become a model input shortcut.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state metadata-conditioned JEPA training remains cooled.

## New Or Reopened Family
Family C: Representation Repair, source-latent neighborhood retrieval.

## Exact Next Experiment
`C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_AUDIT`

## Implementation Tasks
- Build retrieval masks from source latent nearest-neighborhood fractions, not cell-line metadata.
- Select neighborhood fraction and endpoint/delta score weight using inner train-action folds.
- Score selected rules on outer train-only heldout perturbation triplets.
- Repeat on the existing fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement. A future JEPA candidate requires the source-latent proxy to approach or preserve the protected recall floor without exact metadata inputs.

## Do-Not-Run List
Do not train a source-state JEPA candidate before C010 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C010 source-latent neighborhood retrieval audit



# Session Amendment 031: Source-Neighborhood Purity Audit

## Trigger
`C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C010 showed that source-latent nearest-neighborhood retrieval is a partial repair but substantially underperforms exact source-state constrained retrieval. The failure could come from impure source latent neighborhoods or from perturbation-delta insufficiency even with good neighborhoods.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-latent neighborhood JEPA training remains cooled until the source-neighborhood failure mode is localized.

## New Or Reopened Family
Family C: Representation Repair, source-neighborhood purity audit.

## Exact Next Experiment
`C011_SOURCE_NEIGHBORHOOD_PURITY_AUDIT`

## Implementation Tasks
- Measure same-cell-line/source-state purity and coverage of source latent nearest-neighborhoods.
- Compare exact source-state constrained nested recall with source-latent neighborhood nested recall.
- Run the same purity summary on the fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA objective before C011 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C011 source-neighborhood purity audit



# Session Amendment 032: Source-State Latent-Space Audit

## Trigger
`C011_SOURCE_LATENT_NEIGHBORHOOD_IMPURITY_EXPLAINS_PROXY_FAILURE`

## Evidence
C011 showed source-latent neighborhoods in z_bio are impure enough to explain the failure of source-latent neighborhood retrieval relative to exact source-state constrained retrieval.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preserving JEPA training remains cooled until the current latent allocation of source-state information is audited.

## New Or Reopened Family
Family C: Representation Repair, source-state latent-space allocation audit.

## Exact Next Experiment
`C012_SOURCE_STATE_LATENT_SPACE_AUDIT`

## Implementation Tasks
- Compare source-state/cell-line neighborhood purity in z_bio, z_tech, joint z_bio+z_tech, and online z_bio source latents.
- Run the same summary on the existing fresh synthetic seed cache when available.
- Decide whether source-state signal is weak in z_bio specifically or weak across available source spaces.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA candidate before C012 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C012 source-state latent-space allocation audit



# Session Amendment 033: Online-Source-Neighborhood Retrieval Audit

## Trigger
`C012_SOURCE_STATE_SIGNAL_STRONGER_OUTSIDE_Z_BIO`

## Evidence
C012 showed that source-state/cell-line neighborhoods are much cleaner in online/context z_bio than in teacher z_bio on the current seed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preserving JEPA training remains cooled until online-source neighborhoods prove they repair retrieval under nested calibration.

## New Or Reopened Family
Family C: Representation Repair, online-source-neighborhood retrieval.

## Exact Next Experiment
`C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_AUDIT`

## Implementation Tasks
- Use online/context z_bio source latents to construct source-neighborhood masks.
- Select neighborhood fraction and endpoint/delta score weight using inner train-action folds.
- Score selected rules on outer train-only heldout perturbation triplets.
- Repeat on the existing fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA candidate before C013 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C013 online-source-neighborhood retrieval audit

# Session Amendment 034: Online-Source-Neighborhood Oracle Capacity Audit

## Trigger
`C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C013 showed online/context z_bio source-neighborhood retrieval partially repairs condition recall but remains below the protected floor on both current and fresh seed. Before training a source-state-preserving JEPA objective, the loop must determine whether the C013 grid contains train-only oracle rules that can cross the floor, or whether source-neighborhood capacity itself is seed-unstable.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state-preserving JEPA training remains cooled until C014 distinguishes calibration failure from source-neighborhood capacity failure.

## New Or Reopened Family
Family C: Representation Repair, online-source-neighborhood oracle-capacity audit.

## Exact Next Experiment
`C014_ONLINE_SOURCE_NEIGHBORHOOD_ORACLE_CAPACITY_AUDIT`

## Implementation Tasks
- Reuse C013 current and fresh outer-grid TSVs.
- For each outer triplet, compare the inner-selected rule with the oracle-best rule over source-neighborhood fraction and delta-score weight.
- Report selected mean, oracle mean, selected-to-oracle gap, fraction of triplets above the protected recall floor, and seed stability.
- Use this only as diagnostic evidence; do not use oracle choices for model selection or promotion.

## Gates
Diagnostic only; no model promotion and no metric replacement. A source-state-preserving JEPA objective requires evidence that train-only or deployable rules can plausibly preserve the protected recall floor across seeds.

## Do-Not-Run List
Do not train source-state-preserving JEPA from C013 alone. Do not use C014 oracle choices as deployable selection rules.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C014 online-source-neighborhood oracle capacity audit



# Session Amendment 034: Online-Teacher Source Geometry Audit

## Trigger
`C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C013 showed online/context source neighborhoods improve retrieval but remain below floor. C012 showed online/context z_bio has much stronger source-state purity than teacher z_bio, so the next question is whether the teacher target representation attenuates source-state structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preserving JEPA training remains cooled until online-vs-teacher source geometry is documented.

## New Or Reopened Family
Family C: Representation Repair, online-vs-teacher source geometry audit.

## Exact Next Experiment
`C014_ONLINE_TEACHER_SOURCE_GEOMETRY_AUDIT`

## Implementation Tasks
- Compare online/context z_bio with teacher z_bio for source-state purity, eta-squared, centroid accuracy, rank, norm, and pairwise separation.
- Measure online-teacher row alignment and whether the online-minus-teacher residual carries source-state information.
- Repeat the same diagnostic on the existing fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA candidate before C014 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C014 online-vs-teacher source geometry audit



# Session Amendment 035: Source-State Preservation Decision Audit

## Trigger
`C014_TEACHER_TARGET_ATTENUATES_SOURCE_STATE_STRUCTURE`

## Evidence
C014 showed teacher target z_bio attenuates source-state structure relative to online/context z_bio. However, C013 online-source-neighborhood retrieval still stayed below the protected floor, especially on the fresh synthetic seed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preservation training remains cooled until a decision audit confirms enough calibrated and fresh-seed headroom.

## New Or Reopened Family
Family C: Representation Repair, source-state preservation decision audit.

## Exact Next Experiment
`C015_SOURCE_STATE_PRESERVATION_DECISION_AUDIT`

## Implementation Tasks
- Reconcile exact source-state constrained retrieval, teacher-source-neighborhood retrieval, online-source-neighborhood retrieval, and online/teacher geometry diagnostics.
- Compare current-seed and fresh-seed floor gaps.
- Decide whether to reopen a source-state preservation training candidate or continue with lower-compute diagnostics.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA candidate unless C015 explicitly approves reopening.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C015 source-state preservation decision audit



# Session Amendment 036: Multi-Seed Fresh-Cache Replication

## Trigger
`C015_DO_NOT_REOPEN_TRAINING_SOURCE_STATE_PRESERVATION_INSUFFICIENT`

## Evidence
C015 reconciled exact source-state, source-latent, online-source-neighborhood, and online/teacher geometry diagnostics. The source-state preservation path has current-seed headroom but fails the deployable current/fresh floor check. A single fresh seed is too weak to decide whether this is a stable representation limitation or seed-specific geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Source-state preservation training remains cooled. Do not run C016 unless G002 shows stable fresh-seed evidence that a deployable proxy reaches or approaches the protected recall floor.

## New Or Reopened Family
Family G: fresh synthetic seed geometry replication.

## Exact Next Experiment
`G002_MULTI_SEED_FRESH_CACHE_REPLICATION`

## Implementation Tasks
- Reuse existing seed1 latent cache.
- Generate fresh latent caches for seeds 2, 3, and 4 if missing.
- For each fresh seed, compute floor transition metrics, train-only endpoint/delta composite summary, nested online-source-neighborhood retrieval, and online-vs-teacher source geometry.
- Summarize whether source-state proxy behavior is stable enough to justify architecture reopening.

## Gates
Diagnostic only; no model promotion and no metric replacement. A training candidate remains disallowed unless a deployable non-metadata source proxy reaches or nearly reaches the protected recall floor across fresh seeds.

## Do-Not-Run List
Do not train source-state preserving JEPA. Do not modify splits/evaluators. Do not use cell-line labels as model inputs; labels are audit-only.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: G002 multi-seed fresh-cache replication



# Session Amendment 037: Protected-Floor Seed-Stability Audit

## Trigger
`G002_FRESH_REPLICATION_CONFIRMS_SOURCE_PROXY_INSTABILITY_BELOW_FLOOR`

## Evidence
G002 showed the deployable online-source proxy stays below the protected recall floor on seeds 1-4. It also showed the train-only floor recall itself is far lower on fresh synthetic seeds than on seed0. Before more architecture work, determine whether the recall gate is a stable protected target or a seed-specific artifact.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it. No model is promoted by this diagnostic.

## Retired / Cooled Families
Source-state preservation training remains cooled. Fresh-cache diagnostics have shown insufficient deployable proxy stability.

## New Or Reopened Family
Family F: metric/data redesign, protected-floor seed-stability audit.

## Exact Next Experiment
`F011_PROTECTED_FLOOR_SEED_STABILITY_AUDIT`

## Implementation Tasks
- Compare seed0 protected floor metrics against fresh-seed floor metrics from G002.
- Quantify recall, transition, delta-cosine, rank, and magnitude variability across synthetic seeds.
- Decide whether the architecture loop should continue under the current recall gate or pivot to benchmark/metric redesign documentation.

## Gates
Diagnostic only. Do not change the protected floor, split, evaluator, or promotion criteria in this experiment.

## Do-Not-Run List
Do not train new architecture. Do not replace recall gate. Do not use eval target rows for fitting or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F011 protected-floor seed-stability audit



# Session Amendment 036: Multi-Seed Source-State Stability Audit

## Trigger
`C015_DO_NOT_REOPEN_TRAINING_SOURCE_STATE_PRESERVATION_INSUFFICIENT`

## Evidence
C015 decided not to reopen source-state preservation training: the current seed nearly reached the protected recall floor under exact source-state diagnostics, but the learnable online-source-neighborhood proxy stayed below floor and fresh seed support was weaker.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preservation JEPA training is cooled unless multi-seed diagnostics show stable, learnable, floor-preserving source-state retrieval support.

## New Or Reopened Family
Family G: fresh synthetic seed geometry, source-state stability.

## Exact Next Experiment
`G002_MULTI_SEED_SOURCE_STATE_STABILITY_AUDIT`

## Implementation Tasks
- Reuse current seed and existing seed-1 cache.
- Generate additional tiny CPU latent caches for synthetic seeds 2 and 3.
- For each seed, run nested exact source-state retrieval, teacher-source-neighborhood retrieval, online-source-neighborhood retrieval, and online/teacher source geometry.
- Do not use source-state metadata, condition keys, or exact target keys as model inputs.
- Use metadata only as diagnostic scoring labels/constraints and document that limitation.

## Gates
Diagnostic only; no model promotion and no metric replacement. Reopen training only if online-source-neighborhood retrieval is stable across seeds and preserves the protected floor.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F011 recall-gate redesign memo



# Session Amendment 038: Benchmark Redesign Brief Before More Architecture

## Trigger
`F011_RECALL_FLOOR_SEED_SPECIFIC_CONTINUOUS_METRICS_STABLE`

## Evidence
F011 showed the protected seed0 recall floor is seed-specific: seed0 recall@1 is `0.481481`, while fresh seeds average `0.296296` and max out at `0.333333`. Continuous transition and delta-cosine metrics do not show the same seed0 advantage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it. No gate changes are authorized by this amendment alone.

## Retired / Cooled Families
Architecture search against the current seed0 recall floor is cooled until a benchmark redesign is documented. Source-state preservation remains cooled.

## New Or Reopened Family
Family F: metric/data redesign documentation.

## Exact Next Experiment
`F012_BENCHMARK_REDESIGN_BRIEF`

## Implementation Tasks
- Write a self-contained redesign brief that separates locked current benchmark facts from proposed future benchmark changes.
- Preserve the old protected model and old protected gate as historical references.
- Propose any new benchmark under a new name, with multi-seed baseline registry and no silent split/evaluator mutation.

## Gates
Documentation only. Do not promote any model. Do not change current gates/evaluators.

## Do-Not-Run List
Do not run new architecture under a changed gate. Do not overwrite current split semantics. Do not use eval targets for fitting or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F012 benchmark redesign brief



# Session Amendment 039: Multi-Seed Benchmark Registry

## Trigger
`F012_BENCHMARK_REDESIGN_REQUIRED_BEFORE_MORE_ARCHITECTURE`

## Evidence
F012 requires a new named multi-seed benchmark before more architecture search. The old seed0 benchmark remains locked as historical reference, but should not be silently modified.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## New Benchmark Name
`synthetic_genetic_anchor_lite_multiseed_v1`

## Exact Next Experiment
`F013_MULTISEED_BENCHMARK_REGISTRY`

## Implementation Tasks
- Use seed0-4 latent caches.
- Compute train-only baselines per seed: full action-ridge floor, source-only ridge, action-only ridge, mean-delta null, and source-as-target null.
- Aggregate mean/std and propose future gates without applying them to the old benchmark.
- Preserve leakage controls: fit only on train split per seed.

## Gates
Registry only. No model promotion. No old-gate mutation.

## Do-Not-Run List
Do not run new architecture until registry is complete.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F013 multiseed benchmark registry



# Session Amendment 040: Activate Multiseed v1 Benchmark

## Trigger
`F013_MULTISEED_BENCHMARK_REGISTRY_BUILT_PROPOSED_GATES_NOT_ACTIVE`

## Evidence
F013 built `synthetic_genetic_anchor_lite_multiseed_v1` across five synthetic seeds. The old seed0 benchmark remains locked as historical reference, but future architecture work should target the new named benchmark unless an amendment explicitly says otherwise.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout remains the model of record. Activation of this benchmark does not promote a model.

## Active Search Benchmark
`synthetic_genetic_anchor_lite_multiseed_v1`

## Baseline Registry
- full action-ridge transition mean: `0.011281` +/- `0.008060`
- full action-ridge delta cosine mean: `0.451515` +/- `0.057127`
- full action-ridge condition recall mean: `0.333333` +/- `0.077690`

## Provisional Active Gates For v1
These gates apply only to `synthetic_genetic_anchor_lite_multiseed_v1` and do not alter the old benchmark.

Primary continuous gates:
- mean transition improvement must be >= `0.003221` (baseline mean - 1 std)
- mean delta cosine must be >= `0.394388` (baseline mean - 1 std)

Secondary/no-regression gates:
- condition recall@1 must be >= `0.255644` (baseline mean - 1 std)
- effective rank must remain within 20% of full action-ridge mean
- magnitude ratio must remain within 25% of full action-ridge mean

Promotion rule:
A candidate can only be considered a real improvement if it exceeds the full action-ridge mean on at least one primary continuous metric without failing secondary gates. Tier 3 promotion still requires a full no-regression package.

## Exact Next Experiment
`BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY`

## Implementation Tasks
- Run the existing action-AdaLN residual candidate on seed caches 0-4.
- Use train-only calibration per seed.
- Aggregate against the v1 registry.
- Do not save/promote checkpoints unless a nonzero residual passes gates.

## Do-Not-Run List
Do not mutate old `test_heldout_perturbation` gates. Do not use eval target rows for fitting or residual selection. Do not promote a Tier 1/Tier 2 sanity result.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F014 activate multiseed benchmark gates



# Session Amendment 041: BGWM003 Multiseed Action-AdaLN Sanity

## Trigger
`F014_MULTISEED_V1_BENCHMARK_ACTIVATED_FOR_SEARCH`

## Evidence
F014 explicitly activated `synthetic_genetic_anchor_lite_multiseed_v1` for future architecture search and named `BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY` as the next experiment. The old seed0 benchmark remains locked and historical.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Active Benchmark Gates
- Mean transition improvement must meet the active F014 primary transition gate.
- Mean delta cosine must meet the active F014 primary delta-cosine gate.
- Mean condition recall@1 is secondary/no-regression under the active F014 gate.
- No Tier 1 result can promote the model of record.

## New Or Reopened Family
Phase 8 v3 action-AdaLN + RoPE floor-preserving residual sanity under multiseed v1.

## Exact Next Experiment
`BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY`

## Implementation Tasks
- Run the existing floor-preserving Action-AdaLN + RoPE residual path on seeds 0-4.
- Use each seed's train split only for floor and residual calibration.
- Score eval rows only after train-only calibration.
- Report per-seed and mean metrics under the F014 gates.
- Keep the result non-promotional.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: BGWM003 multiseed Action-AdaLN sanity



# Session Amendment 042: F015 Multi-Environment Transition Audit

## Trigger
`BGWM003_NO_IMPROVEMENT_ZERO_RESIDUAL_FLOOR_FALLBACK`

## Evidence
BGWM003 passed the active multiseed gates only through exact floor fallback. All seeds selected residual scale zero, and fold traces showed negligible residual signal with zero action-negative separation.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Active Benchmark
`synthetic_genetic_anchor_lite_multiseed_v1` remains active for search. The old seed0 benchmark remains locked and historical.

## New Or Reopened Family
Family F / environment-stable transition diagnostics.

## Exact Next Experiment
`F015_MULTI_ENVIRONMENT_TRANSITION_AUDIT`

## Implementation Tasks
- Compare per-seed action-ridge floor against pooled full ridge, pooled source-only ridge, pooled action-only ridge, train-only environment-centered ridge, and leave-one-seed-out full ridge.
- Use train splits only for all fitting and centering.
- Use eval rows only for scoring.
- Treat seed/environment labels as diagnostic grouping variables, not biological shortcuts.

## Gates
Diagnostic only. A pooled/environment baseline cannot promote the model of record. It can justify a future environment-stable JEPA amendment if it shows train-only headroom.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F015 multi-environment transition audit



# Session Amendment 043: F016 Environment Blend Calibration

## Trigger
`F015_POOLED_ENVIRONMENT_TRANSITION_PASSES_GATES_BELOW_PER_SEED_FLOOR`

## Evidence
F015 found pooled/environment transition headroom, but the strongest transition baselines regressed condition recall below the per-seed action-ridge floor. This makes immediate neural residual training premature.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Phase 8 v3 or Family F diagnostic can promote.

## Protected Transition Floor
The full train-only action-ridge floor remains the transition floor for residual/operator candidates. Zero blend scale must reproduce the per-seed floor exactly.

## New Or Reopened Family
Family F: Metric And Data Redesign, environment-blend calibration branch.

## Exact Next Experiment
`F016_ENVIRONMENT_BLEND_CALIBRATION`

## Implementation Tasks
- Build convex blends from each seed's per-seed action-ridge floor toward pooled source-only, pooled full, and environment-centered transition predictions.
- Select blend family and scale only by train-split global perturbation-group cross-fitting.
- Score held-out eval rows only after calibration is fixed.
- Default to the floor if no train-only rule preserves transition, recall, and delta cosine.

## Gates
Diagnostic only. A nonzero blend must preserve the per-seed floor on held-out scoring before it can justify a later JEPA mechanism. It cannot promote the model of record.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F016 train-only environment blend calibration



# Session Amendment 044: F017 Environment Blend Oracle Capacity

## Trigger
`F016_TRAIN_ONLY_BLEND_SELECTS_FLOOR_FALLBACK`

## Evidence
F016 found nonzero environment blends with mean transition gains, but every train-only candidate that mattered carried perturbation-fold recall risk. The calibrated rule therefore selected the exact floor fallback.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No oracle diagnostic can promote.

## New Or Reopened Family
Family F: Metric And Data Redesign, environment-blend capacity branch.

## Exact Next Experiment
`F017_ENVIRONMENT_BLEND_ORACLE_CAPACITY`

## Implementation Tasks
- Fit the same train-only per-seed floor and pooled/environment candidates.
- On held-out eval rows, measure an oracle row-level mask that only activates a blend when it preserves top-1 condition retrieval and row-level transition improvement relative to the floor.
- Report whether any candidate has nonzero capacity under this idealized mask.
- Treat the result as capacity-only; do not use it as a calibration rule.

## Gates
Diagnostic only. If oracle capacity is absent, cool environment blending. If oracle capacity exists, the next step must be a train-only risk proxy, not direct promotion.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F017 environment blend oracle capacity



# Session Amendment 045: F018 Environment Blend Risk Proxy

## Trigger
`F017_ORACLE_SAFE_ENVIRONMENT_BLEND_CAPACITY_EXISTS`

## Evidence
F017 found large eval-oracle safe blend capacity, but the oracle mask used held-out scoring labels and cannot be used as a candidate. The next step must test whether non-label train-only features can approximate the safe mask.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Family F diagnostic can promote.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only environment risk-proxy branch.

## Exact Next Experiment
`F018_ENVIRONMENT_BLEND_RISK_PROXY`

## Implementation Tasks
- Build train-only perturbation-group folds.
- Label safe activation only inside train folds by comparing blended and floor predictions.
- Select a simple threshold gate using non-label features only: action support, source support, delta support, and floor-vs-blend geometry.
- Apply the fixed selected rule to held-out eval rows for scoring only.

## Gates
Diagnostic only. A nonzero risk proxy must preserve the per-seed floor on held-out transition, delta cosine, and condition recall before it can justify a later JEPA mechanism.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F018 train-only environment risk proxy



# Session Amendment 046: F019 Risk-Proxy Failure Localization

## Trigger
`F018_TRAIN_ONLY_RISK_PROXY_DISCARDED_HELDOUT_BELOW_FLOOR`

## Evidence
F018 selected a nonzero train-only support-gain proxy, but held-out seed0 lost condition recall. Under the floor-preservation contract, the candidate is discarded even though mean transition and mean delta improved.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Or Reopened Family
Family F: Metric And Data Redesign, risk-proxy failure-localization branch.

## Exact Next Experiment
`F019_RISK_PROXY_FAILURE_LOCALIZATION`

## Implementation Tasks
- Reconstruct the F018 selected rule.
- Identify safe vs unsafe selected rows per seed.
- Compare non-label feature distributions for safe and unsafe selected rows.
- Run an eval-oracle second-gate capacity audit to determine whether a simple additional feature threshold could repair seed0 without using it as a promoted rule.

## Gates
Diagnostic only. If a simple second gate has oracle capacity, the next step must be nested train-only calibration. If not, cool environment blending and pivot.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F019 risk-proxy failure localization



# Session Amendment 047: F020 Learned Environment Risk Gate

## Trigger
`F019_NO_SIMPLE_SECOND_GATE_REPAIRS_F018_FAILURE`

## Evidence
F017 showed oracle safe-blend capacity, F018 showed a train-only simple proxy can improve mean transition while violating seed0 recall, and F019 showed no simple second feature threshold repairs that failure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Or Reopened Family
Family F: Metric And Data Redesign, learned risk-gate branch.

## Exact Next Experiment
`F020_LEARNED_ENVIRONMENT_RISK_GATE`

## Implementation Tasks
- Train a tiny linear risk score on train-only perturbation-fold safe/unsafe labels.
- Use out-of-fold train scores to select candidate, alpha, and threshold.
- Freeze the selected rule and score held-out eval rows.
- Enforce per-seed floor preservation for transition, delta cosine, and condition recall.

## Gates
Diagnostic only. If F020 cannot preserve the per-seed floor, cool environment blending and pivot away from risk-gated environment residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F020 learned environment risk gate



# Session Amendment 048: F021 Representation Pivot Requalification

## Trigger
`F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK`

## Closure Evidence
Environment-blend/risk-gate branch closure report: `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/phase_reports/phase_closure_report_048.md`.

F016 selected floor fallback, F018 violated seed0 recall, F019 found no simple second gate, and F020 selected floor fallback under out-of-fold learned risk calibration.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## Retired / Cooled Families
Family F environment-blend/risk-gate residuals are cooled unless a later benchmark change creates new evidence.

## New Or Reopened Family
Representation repair requalification under active `synthetic_genetic_anchor_lite_multiseed_v1` gates.

## Exact Next Experiment
`F021_REPRESENTATION_PIVOT_REQUALIFICATION`

## Implementation Tasks
- Recheck prior source-state and online-neighborhood representation diagnostics against the active F014 multiseed gates.
- Do not promote any diagnostic.
- If requalification passes, the next family may implement a tiny source-state-preserving JEPA candidate under strict identity checks.

## Gates
Diagnostic only. Requalification can only authorize a future Tier 1 candidate; it cannot supersede the protected model of record.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F021 representation pivot requalification



# Session Amendment 049: C016 Online Latent Transition Audit

## Trigger
`F021_REOPEN_REPRESENTATION_REPAIR_UNDER_ACTIVE_MULTISEED_GATES`

## Evidence
F021 reopened representation repair under the active multiseed gates. Prior C012/C013 evidence suggested source-state information is stronger in online latent geometry than in the teacher z_bio used by the transition floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Or Reopened Family
Family C: Representation Repair, online/source-state latent branch.

## Exact Next Experiment
`C016_ONLINE_LATENT_TRANSITION_AUDIT`

## Implementation Tasks
- Rebuild train-only action-ridge transition floors in teacher z_bio and online z_bio spaces across seeds 0-4.
- Score held-out eval rows only.
- Compare transition, delta cosine, condition recall, and latent rank under active F014 gates.

## Gates
Diagnostic only. A pass can justify a tiny source-state-preserving JEPA candidate, not promotion.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C016 online latent transition audit



# Session Amendment 050: F021 Cell-JEPA RNA Representation Warmstart

## Trigger
`C016_ONLINE_LATENT_REPRESENTATION_HAS_REPAIR_SIGNAL`

## User Amendment Integrated
The user added `papers/2602.02093v1.pdf` and instructed that Cell-JEPA should be used as a single-cell RNA representation warmstart, not as a promoted perturbation-transition solution.

## Paper Consulted
`papers/2602.02093v1.pdf`

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## Protected Transition Floor
The protected full train-only action-ridge transition floor remains the transition reference for residual/operator candidates.

## New Diagnostic Branch
`F021_CellJEPA_RNA_Representation_Warmstart`

## Implementation Tasks
- Train a tiny Cell-JEPA-style RNA warmstart on synthetic train rows only.
- Student RNA encoder receives masked expression values.
- EMA teacher RNA encoder receives unmasked expression values.
- Student predictor predicts stop-gradient teacher cell embeddings.
- Use cosine JEPA loss plus a light reconstruction anchor; `w_jepa` must dominate `w_rec`.
- Support per-cell quantile binning, random expressed-gene subsampling, and expression-value masking only.
- Evaluate representation before transition: same-cell/same-condition retrieval, RNA-image retrieval, rank, perturbation probe, batch probe, dropout robustness, and leakage controls.
- Rerun the protected train-only action-ridge floor on frozen Cell-JEPA `z_bio`.

## Gates
F021 cannot promote a transition model. Residual/risk-gate search reopens only if frozen warmstarted `z_bio` improves or preserves transition improvement, recall@1, delta cosine, effective rank, and per-seed stability.

## Forbidden Inputs
No perturbation ID one-hot as held-out shortcut, no `condition_key` or `biological_key` model input, no held-out target means, and no promotion based only on absolute post-perturbation Pearson.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F021_CellJEPA_RNA_Representation_Warmstart



# Session Amendment 051: F022 RNA Representation Ceiling Audit

## Trigger
`F021_CELLJEPA_WARMSTART_DISCARDED_NO_SAFE_REPRESENTATION_GAIN`

## Evidence
F021 implemented a JEPA-dominant Cell-JEPA RNA warmstart, but it did not improve representation probes or preserve the protected transition floor. The next step must not be training longer by default or reopening residual/risk-gate transition operators.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F022_RNA_REPRESENTATION_CEILING_AUDIT`

## Implementation Tasks
- Compare the protected old BioFlow teacher transition floor with train-only floors built from true synthetic `z_bio`, clean RNA PCA, observed RNA PCA, and observed count PCA.
- Fit PCA bases on synthetic train rows only.
- Use eval target rows only for scoring.
- Do not use `condition_key`, `biological_key`, held-out target means, or perturbation ID one-hot shortcuts as model inputs.

## Decision Use
If true synthetic `z_bio` has transition signal but RNA projections lose it, pivot to representation extraction. If true `z_bio` also fails, pivot to benchmark/action descriptor redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F022 RNA representation ceiling audit



# Session Amendment 052: F023 Old Latent Action Contract Audit

## Trigger
`F022_SYNTHETIC_Z_BIO_CEILING_DOES_NOT_SUPPORT_TRANSITION_SEARCH`

## Evidence
F022 showed the old positive BioFlow/BioTech teacher latent floor is not reproduced by true synthetic `z_bio` or train-only RNA PCA representations. Before changing architecture, the old floor's action/source contract must be audited.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F023_OLD_LATENT_ACTION_CONTRACT_AUDIT`

## Implementation Tasks
- Reproduce the old full action-ridge floor across seeds 0-4.
- Compare source-only ridge, action-only ridge, mean delta, wrong eval action, permuted train action, and metadata group-mean audit references.
- Treat metadata group-mean rows as diagnostic references only, never as candidate model inputs.
- Do not use eval target rows for fitting.

## Decision Use
If wrong/permuted action performs like the full floor, cool residual/operator work and pivot to target/benchmark redesign. If action ablation materially hurts, reopen action representation work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F023 old latent action/source contract audit



# Session Amendment 053: F024 Heldout Action Descriptor Support Audit

## Trigger
`F023_OLD_LATENT_ACTION_CONTRACT_AMBIGUOUS`

## Evidence
F023 found that source-only latents recover most of the old floor, while correct action still beats wrong action by a small amount. Action-only does not pass the active gates. The action descriptor support must be localized before training graph/program action priors.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_AUDIT`

## Implementation Tasks
- For every held-out perturbation and seed, list active action descriptor dimensions.
- Record which active dimensions were seen in train rows.
- Score full vs source-only vs wrong-action gains by held-out perturbation.
- Do not use held-out target means, `condition_key`, or `biological_key` as model inputs.

## Decision Use
If held-out action dimensions are unsupported, pivot to non-exact-key biological action descriptors or benchmark redesign. If support is sufficient and action gains are consistent, reopen action-representation repair.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F024 heldout action descriptor support audit



# Session Amendment 054: F025 Program-Only Action Descriptor Audit

## Trigger
`F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_LIMITS_ACTION_GENERALIZATION`

## Evidence
F024 showed held-out exact perturbation action dimensions are unsupported in train rows, while one program-level descriptor dimension remains supported. Correct action gives a small but inconsistent gain. Before implementing graph/program action JEPA, test whether supported non-exact action descriptors preserve the old floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F025_PROGRAM_ONLY_ACTION_DESCRIPTOR_AUDIT`

## Implementation Tasks
- Compare full exact+program action descriptors with source-only, supported-action-only, program-only, and supported-program-only descriptors.
- Fit all floors on train rows only.
- Use eval rows only for scoring.
- Do not use `condition_key`, `biological_key`, held-out target means, or exact target-key one-hot features as candidate model inputs.

## Decision Use
If program descriptors preserve the old floor, reopen non-exact biological action representation work. If they fail, pivot to benchmark/action-target redesign before more JEPA architecture.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F025 program-only action descriptor audit



# Session Amendment 055: Descriptor-Aligned Synthetic Benchmark Audit

## Trigger
`F025_NON_EXACT_PROGRAM_ACTION_DESCRIPTOR_PRESERVES_OLD_FLOOR`

## Evidence
F022 showed that true synthetic `z_bio` does not reproduce the current positive transition floor. F024/F025 localized a descriptor-support problem: exact held-out perturbation dimensions are unsupported, and pure program descriptors fail. Inspecting the synthetic generator shows why: perturbation directions are independent random vectors, so the program descriptor is not causally tied to the held-out target direction.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F026_DESCRIPTOR_ALIGNED_SYNTHETIC_BENCHMARK_AUDIT`

## Implementation Tasks
- Add a new named synthetic config that leaves the old benchmark locked.
- Generate perturbation directions from shared program factors plus small perturbation-specific noise.
- Keep genetic-guide dose fixed and ignore chemical dose for this benchmark.
- Run Step 0 floors across seeds 0-4 using true synthetic `z_bio`, clean RNA PCA, and observed RNA PCA.
- Compare source-as-target, mean delta, source-only ridge, full action ridge, supported-action ridge, and non-exact program-action ridge.
- Use train rows only for PCA/ridge fitting and eval rows only for scoring.

## Decision Use
Reopen JEPA architecture only if the new true `z_bio` plus non-exact program action descriptors passes the active gates and the direction geometry audit proves held-out program support. Otherwise keep architecture cooled and continue metric/data redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F026 descriptor-aligned synthetic benchmark audit



# Session Amendment 056: Cell-JEPA Warmstart On Descriptor-Aligned Benchmark

## Trigger
`F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN`

## Evidence
F026 approved the new `synth_program_aligned_genetic_lite` benchmark: true synthetic `z_bio` plus non-exact program actions passed the active Step 0 gates, and observed RNA PCA also passed. Before spending more compute on a full cross-modal/action JEPA, the representation path should be checked on the repaired benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F027_CELLJEPA_PROGRAM_ALIGNED_WARMSTART`

## Implementation Tasks
- Train a tiny Cell-JEPA-style RNA warmstart on train expression rows only.
- Use masked student RNA, unmasked EMA teacher RNA, stop-gradient latent targets, cosine JEPA loss, and light reconstruction anchor.
- Compare frozen Cell-JEPA `z_bio` against true synthetic `z_bio` and train-only observed RNA PCA under non-exact program-action transition floors.
- Do not use exact held-out perturbation one-hot features, `condition_key`, `biological_key`, held-out target means, or pooled train+test statistics.

## Decision Use
If Cell-JEPA preserves the repaired benchmark floor, proceed to a small full cross-modal/action JEPA probe. If Cell-JEPA falls below observed RNA PCA, prioritize representation repair or PCA-teacher distillation instead of transition residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F027 Cell-JEPA warmstart on descriptor-aligned benchmark



# Session Amendment 057: Train-Only PCA-Distilled RNA Encoder

## Trigger
`F027_CELLJEPA_BELOW_PCA_FLOOR_USE_REPRESENTATION_REPAIR`

## Evidence
F027 Cell-JEPA was JEPA-dominant and non-leaky but fell far below the observed RNA PCA transition floor on the repaired benchmark. This suggests the immediate issue is representation extraction/objective mismatch, not action-target geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F028_TRAIN_ONLY_PCA_DISTILLED_RNA_ENCODER`

## Implementation Tasks
- Fit an observed-RNA PCA teacher on train rows only.
- Train a tiny neural RNA encoder to predict that train-only PCA embedding from masked expression values.
- Score non-exact program-action transition floors on frozen learned embeddings.
- Compare to true synthetic `z_bio` and the original observed RNA PCA teacher.

## Decision Use
If the neural encoder preserves the PCA floor, use it as a bootstrap representation path for a full cross-modal/action JEPA probe. If it underfits, repair the learned RNA encoder objective before training more transition machinery.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F028 train-only PCA-distilled RNA encoder



# Session Amendment 058: PCA-Bootstrap Cross-Modal/Action JEPA

## Trigger
`F028_PCA_DISTILLED_RNA_ENCODER_PRESERVES_PROGRAM_FLOOR`

## Evidence
F028 showed that a neural RNA encoder can preserve the repaired non-exact program-action floor when bootstrapped from train-only PCA. The next step should no longer be another representation-only diagnostic; it should test a small real JEPA world-model path while keeping the PCA anchor auxiliary.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F029_PCA_BOOTSTRAP_CROSS_MODAL_ACTION_JEPA`

## Implementation Tasks
- Train online RNA and image encoders with EMA target encoders.
- Use stop-gradient latent targets.
- Use query-conditioned predictors.
- Include RNA->image and image->RNA latent JEPA losses.
- Include control RNA + non-exact program action -> perturbed RNA latent transition loss.
- Keep train-only PCA anchor auxiliary and lower-weight than JEPA losses.
- Use program descriptors only, not exact perturbation IDs.
- Score direct JEPA transitions, representation ridge floor, cross-modal retrieval, action-negative gap, rank, identity, and leakage diagnostics.

## Decision Use
If direct JEPA transitions pass gates and cross-modal retrieval is healthy, continue to a multi-seed/Tier 2 amendment. If the representation ridge floor remains good but direct JEPA transitions fall below it, pivot to transition predictor optimization rather than representation repair.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F029 PCA-bootstrap cross-modal/action JEPA



# Session Amendment 059: Delta-Direction Cross-Modal/Action JEPA Repair

## Trigger
`F029_REPRESENTATION_FLOOR_GOOD_DIRECT_JEPA_UNDER_FLOOR`

## Evidence
F029 produced positive direct transition improvement and healthy RNA/image retrieval, but its direct JEPA transition delta cosine remained below the active gate. Training longer is not the right first response; the predictor needs explicit direction supervision in latent delta space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA`

## Implementation Tasks
- Preserve the F029 real JEPA identity: online encoders, EMA target encoders, stop-gradient latent targets, query-conditioned predictors, cross-modal JEPA losses, and program-action transition loss.
- Add a stop-gradient teacher-delta direction loss.
- Add a source-improvement hinge so the predicted endpoint must improve over the source state.
- Keep action input as non-exact program descriptors only.
- Score direct JEPA transitions, representation ridge floor, cross-modal retrieval, action-negative gap, rank, identity, and leakage diagnostics.

## Decision Use
If F030 passes all direct transition gates and cross-modal retrieval remains healthy, design a Tier 2 amendment. If delta cosine improves but remains below gate, test an operator-initialized transition predictor rather than training longer. If it regresses, close this repair path.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F030 delta-direction transition repair



# Session Amendment 060: Delta-Direction JEPA Tier 2 Validation

## Trigger
`F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA_TIER1_PASS`

## Evidence
F030 passed the active Tier 1 gates with a real cross-modal/action JEPA, but Tier 1 cannot promote and does not prove stability. The next required step is multi-seed validation with explicit protected-floor and same-representation ridge-floor checks.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Validation Branch
`F031_DELTA_DIRECTION_TIER2_VALIDATION`

## Implementation Tasks
- Rerun the F030 architecture across synthetic seeds 0-4.
- Keep the same low-compute setting and do not train longer by default.
- Report mean/std, per-seed rows, and whether std is smaller than the claimed protected-floor effect.
- Check protected historical floor gates and same-representation train-only ridge floor preservation separately.
- Preserve JEPA identity: online encoders, EMA targets, stop-gradient latent targets, query predictors, cross-modal prediction, action-conditioned transition.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F031 passes protected Tier 2 but fails the same-representation ridge floor, pivot to a floor-preserving transition head before Tier 3. If it preserves both, design Tier 3/no-regression validation. No Tier 2 result can promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F031 Tier 2 delta-direction validation



# Session Amendment 061: Floor-Preserving JEPA Residual Calibration

## Trigger
`F031_PROTECTED_TIER2_PASS_LOCAL_RIDGE_FLOOR_NOT_PRESERVED`

## Evidence
F031 passed protected Tier 2 gates and maintained cross-modal retrieval, but the direct JEPA transition head fell below the same-representation train-only ridge floor on transition, delta cosine, and recall. The next experiment must preserve the local floor exactly before allowing any JEPA residual.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F032_FLOOR_PRESERVING_JEPA_RESIDUAL_CALIBRATION`

## Implementation Tasks
- Train the same low-compute F030 real JEPA on seeds 0-4.
- Fit the same-representation train-only ridge floor on frozen learned latents.
- Build residuals as `direct_jepa_delta - local_ridge_floor_delta`.
- Select residual scale using train rows only, requiring transition, delta cosine, and recall to preserve the train floor.
- Score held-out rows only after selection.
- Keep action input as non-exact program descriptors only.

## Decision Use
If nonzero residual scale preserves the held-out local floor, design a Tier 3/no-regression amendment. If scale zero is selected, pivot to an operator-initialized transition predictor instead of treating direct residuals as safe.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F032 train-only floor-preserving JEPA residual calibration



# Session Amendment 062: Conservative Floor Gate

## Trigger
`F032_NONZERO_RESIDUAL_DISCARDED_HELDOUT_BELOW_LOCAL_FLOOR`

## Evidence
F032 selected nonzero train-only residual scales and improved mean held-out transition, delta cosine, and recall over the local floor, but one seed fell below the local recall floor. The next experiment should not train longer; it should make calibration stricter and default to the exact local floor unless train evidence has slack.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F033_CONSERVATIVE_FLOOR_GATE`

## Implementation Tasks
- Reuse the F032 deterministic training setup.
- Use a smaller residual scale grid.
- Require positive train-side transition, delta cosine, and recall slack for nonzero residual deployment.
- Evaluate held-out rows only after train-only selection.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F033 preserves the held-out local floor with nonzero residuals, design a Tier 3/no-regression amendment. If it falls back to zero, pivot to operator-initialized direct predictor training. If it still violates local floor, discard this calibration family.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F033 conservative train-only residual gate



# Session Amendment 063: Operator-Initialized Transition Head

## Trigger
`F033_CONSERVATIVE_GATE_ZERO_FALLBACK`

## Evidence
F033 proved that conservative post-hoc residual deployment is safe only by selecting zero residual. The failure mode is architectural: the JEPA transition head is not initialized at the local train-only operator floor. F034 therefore makes floor preservation part of the predictor, not a later rescue step.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F034_OPERATOR_INITIALIZED_TRANSITION_HEAD`

## Implementation Tasks
- Train the same low-compute real JEPA representation path.
- Fit the same-representation train-only ridge floor on frozen train latents.
- Initialize a transition head exactly to that ridge floor.
- Add a zero-initialized action-conditioned residual branch.
- Train only the residual branch with endpoint, delta-direction, and floor-preservation losses.
- Select residual scale using train rows only, then score held-out rows.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F034 preserves the held-out local floor with nonzero residuals, design Tier 3/no-regression validation. If it still falls back to zero or violates the local floor, pivot away from residual scaling toward a different operator family or representation target.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F034 operator-initialized floor-preserving transition head



# Session Amendment 064: Inner-Validation Operator Gate

## Trigger
`F034_OPERATOR_INITIALIZED_RESIDUAL_DISCARDED_HELDOUT_BELOW_FLOOR`

## Evidence
F034 initialized the transition head exactly at the train-only local ridge floor and improved mean held-out transition, delta cosine, and recall, but it still had localized no-regression failures. Selection was based on train-fit metrics, so the next test must use train-only inner validation to choose residual scale.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F035_INNER_VAL_OPERATOR_GATE`

## Implementation Tasks
- Keep the F034 floor-initialized transition head.
- Split train condition rows into inner-fit and inner-validation sets.
- Fit the floor and residual head on inner-fit only.
- Select residual scale on inner-validation only.
- Refit the operator-initialized head on full train and deploy the selected scale on held-out rows for scoring only.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F035 preserves held-out local floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero or still violates local floor, pivot to a different operator/representation family.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F035 inner-validation operator residual gate



# Session Amendment 065: Retrieval Failure Localization

## Trigger
`F035_INNER_VAL_GATE_STILL_HELDOUT_BELOW_FLOOR`

## Evidence
F035 selected nonzero residual scales using train-only inner validation and improved held-out transition and delta cosine over the local ridge floor, but it still violated held-out recall no-regression. The strongest failure was a seed-level condition recall drop, so the next move must inspect retrieval ranks and margins before changing the JEPA architecture.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F036_RETRIEVAL_FAILURE_LOCALIZATION`

## Implementation Tasks
- Rerun the deterministic F035 low-compute setup on seeds 0-4.
- Keep action input as non-exact program descriptors only.
- For each held-out query, compare floor endpoint retrieval and selected residual endpoint retrieval.
- Report top-1 labels, ranks, correct-vs-best-wrong margins, rank changes, residual norms, transition-gain changes, and delta-cosine changes.
- Use held-out labels for diagnostic localization only; do not use them for residual selection or promotion.

## Decision Use
If failures are mostly near-tie margin instability, design a train-only retrieval-margin risk gate or dual endpoint+delta objective. If failures are real endpoint boundary crossings, pivot to retrieval-aware/prototype-aligned transition constraints. Do not train longer by default.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F036 retrieval-failure localization



# Session Amendment 066: Train-Only Retrieval-Margin Gate

## Trigger
`F036_RECALL_FAILURE_PRIMARILY_MARGIN_INSTABILITY`

## Evidence
F036 localized the F035 recall failure: all broken rows were near-tie nearest-neighbor flips, while transition gain and delta-cosine change stayed positive. This supports a train-only retrieval-margin safety gate rather than longer training or a post-hoc recall relaxation.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Calibration Branch
`F037_RETRIEVAL_MARGIN_GATE`

## Implementation Tasks
- Reuse the F035 floor-initialized transition head.
- Split train rows into inner-fit and inner-validation sets.
- Select residual scale on inner-validation only.
- Require transition, delta cosine, aggregate recall, and zero broken floor-correct retrieval rows on inner validation.
- Refit the head on full train and score held-out rows without using held-out labels for selection.

## Decision Use
If F037 preserves held-out local floor with nonzero residuals, design a stricter Tier 3/no-regression validation. If it falls back to zero, pivot to a retrieval-aware transition objective. If it still violates held-out recall, the inner validation gate is insufficient and should be retired.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F037 train-only retrieval-margin gate



# Session Amendment 067: Oracle Safe-Scale Capacity Audit

## Trigger
`F037_RETRIEVAL_MARGIN_GATE_STILL_HELDOUT_BELOW_FLOOR`

## Evidence
F037 improved mean transition, delta cosine, and recall, but still violated a per-seed held-out recall gate. Because the train-only gate had zero inner-validation broken rows, the next question is whether safe nonzero residual deployment exists at all on held-out rows.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F038_ORACLE_SAFE_SCALE_CAPACITY`

## Implementation Tasks
- Rerun the same floor-initialized transition head.
- Evaluate a fixed residual scale grid on held-out rows for diagnosis only.
- Mark a scale safe only if transition, delta cosine, recall, and zero broken floor-correct retrieval rows all preserve the floor.
- Do not use oracle scales for promotion, calibration, or model selection.

## Decision Use
If nonzero safe scales exist, the next branch should learn a train-only proxy for that capacity. If safe scales are zero or absent, retire residual scaling and pivot to a retrieval-aware transition objective or representation redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F038 held-out oracle safe-scale capacity audit



# Session Amendment 068: Train-Only Safe-Scale Proxy

## Trigger
`F038_ORACLE_SAFE_NONZERO_SCALE_CAPACITY_EXISTS`

## Evidence
F038 showed nonzero safe residual scales exist for every seed under oracle held-out selection. The remaining problem is not representation identity or operator signal; it is selecting a safe scale without held-out leakage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Calibration Branch
`F039_TRAIN_PROXY_SAFE_SCALE`

## Implementation Tasks
- Reuse the F038 floor-initialized transition head and low-compute setting.
- Select scale using train rows only.
- Require no aggregate train metric regression, zero broken train retrieval rows, no erosion on floor-correct near-tie train rows, and nonnegative lower-tail margin change.
- Score held-out rows only after train-only scale selection.
- Do not use held-out oracle scales for fitting, selection, or promotion.

## Decision Use
If F039 preserves held-out floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero or still violates held-out recall, pivot to a retrieval-aware transition loss rather than more scale heuristics.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F039 train-only safe-scale proxy



# Session Amendment 069: Retrieval-Aware Transition Head

## Trigger
`F039_TRAIN_PROXY_ZERO_FALLBACK`

## Evidence
F038 proved nonzero safe residual capacity exists under oracle held-out scale selection, but F039's deployable train-only proxy selected the zero floor fallback. The next branch should change the residual target geometry during training instead of only changing post-hoc scale selection.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F040_RETRIEVAL_AWARE_TRANSITION_HEAD`

## Implementation Tasks
- Reuse the floor-initialized transition head.
- Add train-only paired-target retrieval margin preservation during residual-head training.
- Do not use condition keys as model inputs, held-out labels, or held-out oracle scales.
- After training, select scale with the F039 train-only safe-scale proxy.
- Score held-out rows only after train-only selection.

## Decision Use
If F040 preserves held-out floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero, pivot to representation/objective redesign. If it violates held-out recall, retire paired-target margin training under this benchmark.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F040 retrieval-aware transition-head training



# Session Amendment 070: Constraint Ablation Diagnostic

## Trigger
`F040_RETRIEVAL_AWARE_ZERO_FALLBACK`

## Evidence
F038 found safe nonzero held-out residual scales, but F039 and F040 both selected zero under train-only residual calibration. The next step is not more training; it is to identify which train-only constraint blocks the oracle-safe scales.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F041_CONSTRAINT_ABLATION`

## Implementation Tasks
- Join the F039 train-only proxy scale grid with the F038 held-out oracle scale grid by seed and scale.
- Evaluate constraint ablations: full F039 proxy, drop q10, drop near-tie erosion, drop all margin constraints, continuous-only, and small-scale caps.
- Use held-out oracle outcomes only for diagnostic attribution.
- Do not train a new model, promote a candidate, or use held-out oracle outcomes as deployable calibration labels.

## Decision Use
If a simple train-only ablation recovers safe nonzero residuals, design a future validation branch with the rule pre-registered on fresh splits/seeds. If no ablation recovers safe nonzero residuals, pivot away from post-hoc residual scaling.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F041 constraint-ablation diagnostic



# Session Amendment 071: Fresh-Seed Tiny-Cap Validation

## Trigger
`F041_SIMPLE_TRAIN_PROXY_ABLATION_RECOVERS_SAFE_NONZERO_DIAGNOSTIC`

## Evidence
F041 showed the full F039 proxy rejected every oracle-safe nonzero row because near-tie erosion failed on all oracle-safe nonzero rows. A small fixed residual cap of `0.05` preserved held-out gates in the diagnostic grid, but that rule was selected with held-out evidence.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Validation Branch
`F042_FRESH_SMALL_CAP_VALIDATION`

## Implementation Tasks
- Use fresh synthetic seeds 5-9, not the F038/F041 diagnostic seeds.
- Train the same low-compute PCA-bootstrap JEPA and floor-initialized transition head.
- Select residual scale only from `(0.0, 0.025, 0.05)` using train continuous transition/delta/recall preservation.
- Score held-out rows only after train-only scale selection.
- Do not promote this result; a pass can only motivate a stricter future Tier 2/Tier 3 design.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F042 fresh-seed tiny-cap validation



# Session Amendment 072: Action-AdaLN Tiny-Cap Calibration

## Trigger
`F042_FRESH_SMALL_CAP_VALIDATION_PASS_DIAGNOSTIC`

## Evidence
F042 validated the tiny-cap residual rule on fresh synthetic seeds, but it was still an operator-style diagnostic. The next step must test whether the same rule is useful in the named BioGuard-WM Action-AdaLN + RoPE path.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Diagnostic
`F043_ACTION_ADALN_SMALL_CAP`

## Implementation Tasks
- Add a `small_cap_continuous` calibration mode to the Action-AdaLN + RoPE BioGuard-WM frozen-latent training path.
- Restrict candidate residual scales to `<= 0.05`.
- Select scale using train-only crossfit continuous transition, delta, and recall metrics.
- Score held-out rows only after train-only selection.
- Do not promote this result; a pass only permits designing stricter Tier 2/Tier 3 validation.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F043 Action-AdaLN tiny-cap calibration



# Session Amendment 073: Action-AdaLN Failure Audit

## Trigger
`F043_ACTION_ADALN_SMALL_CAP_HELDOUT_BELOW_FLOOR`

## Evidence
F043 applied the tiny-cap rule to the Action-AdaLN + RoPE BioGuard-WM path. One seed selected a nonzero residual, but the exact held-out transition and delta floor gaps were negative, so the candidate must be discarded.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F044_ACTION_ADALN_FAILURE_AUDIT`

## Implementation Tasks
- Read the F043 seed metrics without retraining.
- Check whether stricter train crossfit LCB constraints would have selected any nonzero Action-AdaLN residual.
- Check whether action-negative separation is nonzero.
- Identify whether the failure is a calibration/reporting issue or a residual-path signal issue.

## Decision Use
If Action-AdaLN residual signal is unstable, cool that path and pivot to either an exact floor-preserving operator wrapper or a representation/target redesign. Do not train F043 longer by default.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F044 Action-AdaLN failure localization



# Session Amendment 074: Exact Floor-Initialized Operator Wrapper

## Trigger
`F044_ACTION_ADALN_FAILURE_LOCALIZED_TO_UNSTABLE_NO_ACTION_SIGNAL`

## Evidence
F044 localized the Action-AdaLN tiny-cap failure to unstable train evidence and zero action-negative separation. The earlier F042 result showed a tiny-cap residual can be safe in the exact floor-initialized operator path, so the next step is to test that path on the active multiseed latent benchmark.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Diagnostic
`F045_ACTIVE_OPERATOR_WRAPPER`

## Implementation Tasks
- Use active cached multiseed latent bundles, seeds 0-4.
- Fit the train-only local ridge floor.
- Initialize the operator exactly at the ridge floor.
- Train only a zero-initialized residual head.
- Select scale from `(0.0, 0.025, 0.05)` using train-only continuous metrics.
- Require exact held-out floor preservation and zero broken retrieval rows.
- Do not promote; a pass only motivates stricter validation or full JEPA wrapping.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F045 exact floor-initialized operator wrapper



# Session Amendment 075: Operator Train-Heldout Mismatch Audit

## Trigger
`F045_ACTIVE_OPERATOR_WRAPPER_HELDOUT_BELOW_FLOOR`

## Evidence
F045 tested the exact floor-initialized operator wrapper on the active multiseed latent benchmark. It selected tiny nonzero residuals on every seed using train-only metrics, but held-out transition, delta, recall, and retrieval no-regression failed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F046_OPERATOR_MISMATCH_AUDIT`

## Implementation Tasks
- Read F045 seed rows without retraining.
- Compare train-selected transition/delta/recall gaps against held-out gaps.
- Quantify train-positive/eval-negative mismatch rates.
- Use the result to decide whether the next branch should be a target/representation redesign rather than another residual-cap search.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F046 operator train-heldout mismatch audit



# Session Amendment 076: Target/Action Support Geometry Audit

## Trigger
`F046_OPERATOR_TRAIN_HELDOUT_MISMATCH_DOMINATES`

## Evidence
F046 showed that the exact floor-initialized operator wrapper selected nonzero tiny residuals from train-only metrics, but held-out transition and delta metrics fell below the protected floor on most seeds. The next branch must identify whether this is caused by held-out action/delta/residual targets being outside train support.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F047_TARGET_GEOMETRY_AUDIT`

## Implementation Tasks
- Use active cached multiseed latent bundles, seeds 0-4.
- Fit only the train full-ridge transition floor per seed.
- Measure held-out perturbation/action exact support, active action-dimension support, action cosine support, teacher-delta support, predicted-delta support, residual support, source support, and target support.
- Compare held-out geometry against train-only leave-action pseudoheldout geometry.
- Do not fit, calibrate, select, or promote a new candidate.

## Decision Use
If held-out exact actions or residual targets are outside train support, pivot to action-target contract or representation/benchmark redesign. If support is sufficient, reopen a targeted operator mechanism with the support failure ruled out.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F047 target/action support geometry audit



# Session Amendment 077: Non-Exact Program Action Contract

## Trigger
`F047_HELDOUT_EXACT_ACTION_SUPPORT_GAP_CONFIRMED`

## Evidence
F047 showed held-out perturbations have zero exact action matches and only partial active action-dimension support, while teacher-delta and residual latent support are not absent. This suggests the residual head may be overfitting exact train perturbation descriptors rather than using shared biological action structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F048_NON_EXACT_ACTION_CONTRACT`

## Implementation Tasks
- Remove exact perturbation one-hot action columns from the active operator input.
- Keep only deterministic non-exact program-level action descriptors.
- Reproduce the local program-action ridge floor.
- Train the same exact floor-initialized residual head under the tiny-cap train-only selection rule.
- Require exact held-out local floor preservation before any future full JEPA wrapping.

## Decision Use
If non-exact actions repair the train-heldout mismatch, design a proper BioGuard-WM action-token contract around shared biological programs. If they still fall below floor or select zero, pivot away from residual operator heads toward representation/data contract redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F048 non-exact program action contract



# Session Amendment 078: Protected Floor Feature Contract Audit

## Trigger
`F048_NON_EXACT_ACTION_OPERATOR_STILL_BELOW_FLOOR`

## Evidence
F048 removed exact perturbation action one-hots and used only shared program action descriptors. The program-action floor and program-action residual stayed below the protected full floor, while the full floor remained much stronger. This suggests the protected floor may benefit from train exact-action feature centering or extrapolation even though candidate JEPA paths cannot use exact perturbation one-hots.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. This audit does not demote or replace it.

## New Diagnostic Branch
`F049_FLOOR_FEATURE_CONTRACT`

## Implementation Tasks
- Fit the protected full action-ridge floor on train only.
- Decompose held-out predictions into source, train exact-action, eval-unseen exact-action, and program-action feature contributions.
- Score the full prediction and counterfactual predictions with train exact-action contribution removed.
- Quantify whether the full floor advantage depends on exact train-action feature centering.

## Decision Use
If the full floor depends materially on exact train-action contribution, keep it as protected audit baseline but stop treating it as a candidate-legal action contract. The next branch should target a valid biological action descriptor or benchmark contract.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F049 protected floor feature contract audit



# Session Amendment 079: Source/Action Contract Synthesis

## Trigger
`F049_FLOOR_FEATURE_CONTRACT_AMBIGUOUS`

## Evidence
F049 found no direct train exact-action contribution to held-out full-ridge predictions, while F048 showed program-only action features fall below the protected full floor. The remaining issue is that the full floor appears to be a covariate-adjusted source-state baseline rather than a deployable action-transition contract.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. The full-ridge transition floor remains an audit reference.

## New Diagnostic Branch
`F050_SOURCE_ACTION_CONTRACT_SYNTHESIS`

## Implementation Tasks
- Read F047-F049 artifacts.
- Compare full protected floor, source-only floor, candidate-legal program-action floor, no-exact full fit, and no-action full fit.
- Decide whether further operator residual training is cooled.
- Recommend the next autonomous branch around biological action-target contract redesign if the active floor is source/covariate dominated.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F050 source/action contract synthesis



# Session Amendment 080: Descriptor-Aligned Action Contract Replication

## Trigger
`F050_PROTECTED_FLOOR_IS_COVARIATE_ADJUSTED_SOURCE_DOMINATED`

## Evidence
F050 showed that the active protected full-ridge floor is source/covariate dominated and is a poor identity target for a biological action-transition JEPA. Earlier F026 approved the descriptor-aligned synthetic benchmark, and F042 showed fresh-seed nonzero residual safety on that benchmark. The next step is a low-compute replication before designing another full JEPA phase.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Phase 8/F051 diagnostic can promote.

## New Diagnostic Branch
`F051_DESCRIPTOR_ALIGNED_REPLICATION`

## Implementation Tasks
- Use `synth_program_aligned_genetic_lite`.
- Use non-exact program action descriptors only.
- Train the existing small real JEPA path with RNA online/context encoder, image online/context encoder, EMA target encoders, stop-gradient latent targets, query predictors, cross-modal RNA/image losses, and action-conditioned transition JEPA.
- Fit ridge floor and residual head on train rows only.
- Select residual scale using the F042 train-only tiny-cap rule.
- Score held-out rows only after train-only selection.

## Decision Use
If F051 replicates safe nonzero behavior, design the next full descriptor-aligned BioGuard-WM/JEPAction branch. If it fails, close residual/operator heads and pivot to representation or action-descriptor redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F051 descriptor-aligned action-contract replication



# Session Amendment 081: Descriptor Near-Tie Retrieval Audit

## Trigger
`F051_DESCRIPTOR_ALIGNED_ACTION_CONTRACT_HELDOUT_BELOW_FLOOR`

## Evidence
F051 used non-exact program actions and a real small JEPA path on the descriptor-aligned benchmark. It passed the active continuous gates and selected nonzero residual scale on all fresh seeds, but failed exact local floor preservation because one held-out retrieval row flipped under a tiny margin.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F052 is diagnostic-only and cannot promote.

## New Diagnostic Branch
`F052_DESCRIPTOR_NEAR_TIE_AUDIT`

## Implementation Tasks
- Read F051 metrics, seed rows, train scale grid, and query-level retrieval rows.
- Do not train, refit, recalibrate, or choose a new residual scale.
- Quantify whether broken held-out rows are near-ties and whether continuous transition/delta metrics still improved on those rows.
- Decide whether the next branch should be a train-only retrieval-margin safety gate rather than a larger JEPA.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F052 descriptor near-tie retrieval audit



# Session Amendment 082: Descriptor Margin Gate

## Trigger
`F052_DESCRIPTOR_FAILURE_IS_NEAR_TIE_RETRIEVAL_MARGIN`

## Evidence
F052 localized the F051 failure to one held-out near-tie retrieval row. The nonzero residual slightly improved continuous transition and delta metrics on that row but moved a tiny floor-correct retrieval margin below zero.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F053 is calibration-diagnostic only and cannot promote.

## New Calibration Diagnostic
`F053_DESCRIPTOR_MARGIN_GATE`

## Implementation Tasks
- Read the F051 train-only scale grid.
- Certify a nonzero residual only if the train grid proves continuous floor preservation, no train retrieval breaks, nonnegative train margin movement, and explicit near-tie lower-tail safety diagnostics.
- Default to exact floor fallback if the certificate is missing or fails.
- Do not use F051 held-out broken-row behavior to tune a new residual scale.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F053 train-only descriptor margin gate



# Session Amendment 083: Fresh Descriptor Margin-Gated Rerun

## Trigger
`F053_DESCRIPTOR_MARGIN_GATE_ZERO_FALLBACK_INSUFFICIENT_TRAIN_DIAGNOSTICS`

## Evidence
F053 repaired safety by falling back to the exact floor, but that was not a scientific improvement. The missing piece is a fresh run that logs train near-tie lower-tail diagnostics before held-out scoring, using a smaller residual scale grid.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F054 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F054_DESCRIPTOR_MARGIN_RERUN`

## Implementation Tasks
- Use fresh seeds 13, 14, and 15.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Train the same low-compute real ProgramBootstrapJEPA path.
- Select scale only from train rows using near-tie lower-tail diagnostics on `(0, 0.005, 0.01, 0.0125, 0.025, 0.05)`.
- Score held-out rows only after train-only scale selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F054 fresh descriptor margin-gated rerun



# Session Amendment 084: Row-Wise Residual Abstention

## Trigger
`F054_DESCRIPTOR_MARGIN_RERUN_ZERO_FALLBACK`

## Evidence
F054 selected zero residual because every nonzero global scale caused train near-tie margin erosion. The residual direction still improved train transition and delta metrics, so the next mechanism should be local abstention rather than another scalar scale.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F055 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F055_ROWWISE_ABSTENTION`

## Implementation Tasks
- Use fresh seeds 16, 17, and 18.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Train the same low-compute real ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Build train-only unsafe-margin labels from train retrieval/margin diagnostics.
- Use only inference-available source/action/floor/head-derived features for row-wise abstention.
- Score held-out rows only after the train-only rule is fixed.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F055 row-wise residual abstention



# Session Amendment 085: Row-Wise Abstention Replication

## Trigger
`F055_ROWWISE_ABSTENTION_SAFE_NONZERO_DIAGNOSTIC`

## Evidence
F055 selected nonzero row-wise residuals with train-only abstention, improved held-out transition and delta metrics, preserved recall, and had zero held-out broken retrieval rows. This is promising but not promotable.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F056 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F056_ROWWISE_ABSTENTION_REPLICATION`

## Implementation Tasks
- Use fresh seeds 19, 20, and 21.
- Repeat the exact F055 train-only row-wise abstention mechanism.
- Score held-out rows only after the train-only rule is fixed.
- Preserve the full JEPA identity and leakage constraints.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F056 independent row-wise abstention replication



# Session Amendment 086: Row-Wise Abstention Synthesis

## Trigger
`F056_ROWWISE_ABSTENTION_REPLICATES_SAFE_NONZERO`

## Evidence
F055 and F056 both selected train-only row-wise nonzero residuals, improved held-out transition and delta metrics, preserved recall, and produced zero held-out retrieval breaks. The evidence is still diagnostic and synthetic-only.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F057 is synthesis only and cannot promote.

## New Diagnostic Synthesis
`F057_ROWWISE_SYNTHESIS`

## Implementation Tasks
- Aggregate F055 and F056 seed-level rows.
- Compute six-seed mean, standard deviation, minimum floor gaps, active fraction, and held-out retrieval-break count.
- Decide whether row-wise abstention is ready for a formal non-promoting Tier 2 design.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F057 six-seed row-wise synthesis



# Session Amendment 087: Row-Wise Abstention Tier 2 Validation

## Trigger
`F057_ROWWISE_ABSTENTION_READY_FOR_TIER2_DESIGN`

## Evidence
F057 aggregated F055/F056 into six fresh seeds with nonzero row-wise residual abstention, positive transition and delta gaps, preserved recall, and zero held-out retrieval breaks. The effect is small, so the next step must be a promotion-ineligible Tier 2 validation with stricter seed and stability gates.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F058 is Tier 2 validation only and cannot promote.

## New Validation Branch
`F058_ROWWISE_TIER2`

## Implementation Tasks
- Use fresh seeds 22, 23, 24, 25, 26, and 27.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Repeat the exact train-only row-wise risk abstention mechanism from F055/F056.
- Export per-seed metrics, train selection grids, held-out query rows, and train traces.
- Require exact held-out floor preservation, no retrieval breaks, positive transition and delta gaps, and gap standard deviation not exceeding the mean gap.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F058 row-wise abstention Tier 2 validation



# Session Amendment 088: Row-Wise Transition Stability Audit

## Trigger
`F058_ROWWISE_TIER2_SAFE_BUT_WEAK`

## Evidence
F058 selected nonzero row-wise residuals for every fresh seed, preserved recall, and produced zero held-out retrieval breaks. However, it did not pass Tier 2 because transition-source cosine improvement was too small and unstable: the mean gap was positive but below the seed-level standard deviation, and at least one seed had a negative transition gap.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F059 is a read-only diagnostic and cannot promote.

## New Diagnostic Branch
`F059_ROWWISE_STABILITY_AUDIT`

## Implementation Tasks
- Read F055, F056, and F058 seed rows.
- Isolate whether F058 weakness comes from retrieval breaks, delta collapse, recall regression, or transition-source cosine instability.
- Compare train-selected transition gains to held-out transition gaps.
- Recommend the next mechanism only after localization.
- Do not fit models and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F059 row-wise transition stability audit



# Session Amendment 089: Floor-Direction-Projected Row-Wise Residual

## Trigger
`F059_TIER2_WEAKNESS_IS_TRANSITION_GENERALIZATION_NOISE`

## Evidence
F059 localized F058's Tier 2 miss to transition-source cosine generalization noise. Delta cosine was positive on every seed, recall was preserved, and no retrieval rows broke, but train transition gains were anti-correlated with held-out transition gaps. A train transition gate alone is therefore not trustworthy.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F060 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F060_FLOOR_DIRECTION_PROJECTION`

## Implementation Tasks
- Use fresh seeds 28, 29, and 30.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Project the residual onto the positive ridge floor-delta direction before train-only row-wise abstention.
- Score held-out rows only after train-only rule selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F060 floor-direction-projected row-wise residual



# Session Amendment 090: Projection Collapse Audit

## Trigger
`F060_FLOOR_DIRECTION_PROJECTION_SAFE_BUT_WEAK`

## Evidence
F060 preserved retrieval but was safe-but-weak. Positive floor-direction projection removed the delta-cosine gain and still produced negative transition-source cosine gaps on two of three seeds. This suggests pure projection is too restrictive and does not solve the F058 instability.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F061 is read-only diagnostics and cannot promote.

## New Diagnostic Branch
`F061_PROJECTION_COLLAPSE_AUDIT`

## Implementation Tasks
- Read F058, F059, and F060 artifacts.
- Compare unprojected row-wise residual signal with projected row-wise residual signal.
- Decide whether pure floor-direction projection should be cooled.
- Recommend the next branch without using eval targets for fitting or selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F061 projection collapse audit



# Session Amendment 091: Oracle Residual-Cone Capacity Diagnostic

## Trigger
`F061_PROJECTION_KILLS_DELTA_WITHOUT_STABILIZING_TRANSITION`

## Evidence
F061 closed pure floor-direction projection because it removed the delta-cosine gain and still left negative transition-source cosine gaps. The remaining question is whether an intermediate residual cone between unprojected and projected residuals has any safe nonzero held-out capacity at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F062 is an oracle diagnostic only and cannot promote.

## New Diagnostic Branch
`F062_ORACLE_RESIDUAL_CONE`

## Implementation Tasks
- Use fresh seeds 31, 32, and 33.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- For each cone mix between unprojected and floor-direction-projected residuals, fit train-only row-wise abstention.
- Use held-out labels only to select the best cone mix diagnostically.
- Report whether safe nonzero residual geometry capacity exists.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F062 oracle residual-cone capacity diagnostic



# Session Amendment 092: Train-Only Residual-Cone Selector

## Trigger
`F062_ORACLE_CONE_CAPACITY_EXISTS_ALL_SEEDS`

## Evidence
F062 found held-out oracle residual-cone capacity on all fresh seeds, but it is not deployable because held-out labels selected the cone mix. The next step is to replace oracle cone selection with a train-only cone selector and score fresh held-out seeds once.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F063 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F063_TRAIN_CONE_SELECTOR`

## Implementation Tasks
- Use fresh seeds 34, 35, and 36.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Evaluate cone mixes using train-only row-wise abstention metrics only.
- Select one cone mix per seed before held-out scoring.
- Score held-out rows once and document whether the oracle capacity is recoverable without leakage.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F063 train-only residual-cone selector



# Session Amendment 093: Train-Heldout Selector Mismatch Audit

## Trigger
`F063_TRAIN_CONE_SELECTOR_HELDOUT_BELOW_FLOOR`

## Evidence
F063 replaced held-out oracle cone selection with train-only cone selection. It selected nonzero residuals on all fresh seeds, but the selected residual fell below the protected held-out floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F064 is read-only diagnostic work and cannot promote.

## New Diagnostic Branch
`F064_TRAIN_HELDOUT_SELECTOR_AUDIT`

## Implementation Tasks
- Read F063 seed, cone-grid, and query-row artifacts.
- Compare train selected transition/delta/recall gaps against held-out selected gaps.
- Count train-positive but heldout-negative seeds.
- Separate continuous transition/delta failure from retrieval-row breakage.
- Document whether another residual rerun is justified before a representation/data pivot.
- Do not train a new model and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F064 train-heldout selector mismatch audit



# Session Amendment 094: Train-Only Action-Heldout Calibration

## Trigger
`F064_TRAIN_SELECTOR_OVERFITS_CONTINUOUS_TRANSITION_GAINS`

## Evidence
F064 showed that train-selected residual gains are optimistic: train transition gap was positive while held-out transition and delta gaps were negative. The failure was continuous transition/delta generalization, not retrieval-row breakage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F065 is calibration diagnostic work and cannot promote.

## New Diagnostic Branch
`F065_ACTION_HELDOUT_CALIBRATION`

## Implementation Tasks
- Reuse the descriptor-aligned synthetic benchmark and low-compute real JEPA path.
- Within train rows only, hold out one perturbation ID at a time.
- Fit ridge/head/rules on the remaining train perturbations.
- Select a residual cone only if it preserves transition, delta, recall, and retrieval on the held-out train perturbation folds.
- If no cone passes, default exactly to the protected floor.
- Score the real held-out perturbation split once after train-only calibration.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F065 train-only action-heldout calibration



# Session Amendment 095: Action-Heldout Calibration Mismatch Audit

## Trigger
`F065_ACTION_HELDOUT_GATE_HELDOUT_BELOW_FLOOR`

## Evidence
F065 passed train-only leave-perturbation-out inner calibration but still fell below the protected floor on real held-out perturbations. That means even split-aware train calibration is not currently sufficient.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F066 is read-only diagnostic work and cannot promote.

## New Diagnostic Branch
`F066_ACTION_HELDOUT_MISMATCH_AUDIT`

## Implementation Tasks
- Read F065 seed, inner-fold, and cone-summary artifacts.
- Compare inner train-heldout gaps against real heldout gaps.
- Count train-action-positive but real-negative seeds.
- Regenerate the same synthetic benchmark only for direction-support diagnostics.
- Compare train inner perturbation support with real heldout perturbation support.
- Decide whether the next family should pivot away from residual selectors.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F066 action-heldout mismatch audit



# Session Amendment 096: Close Residual-Selector Family Under Current Data Contract

## Trigger
`F066_TRAIN_ACTION_HELDOUT_STILL_OPTIMISTIC_FOR_REAL_HELDOUT`

## Evidence
F055-F066 tested row-wise abstention, Tier 2 validation, projection, residual cones, train-only cone selection, leave-perturbation-out train calibration, and mismatch audits. Oracle residual capacity exists, but train-only selectors remain optimistic and fall below the protected floor on real held-out perturbation scoring.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F067 is synthesis/closure work and cannot promote.

## Family Status
Residual selector variants under the current descriptor-aligned data contract are closed/cooled:
- scalar cap search;
- row-wise abstention variants;
- pure floor-direction projection;
- residual-cone selectors;
- train action-heldout calibration gates.

## Next Family To Open
`F068_DATA_CONTRACT_CALIBRATION_REDIRECT`

## Implementation Tasks For Next Family
- Design a read-only benchmark/data-contract diagnostic before new model training.
- Determine why train-heldout calibration is optimistic for real heldout perturbations despite similar program-direction support.
- Compare random perturbation holdout, stratified program holdout, and extrapolative perturbation-index holdout on the synthetic generator.
- Keep all existing held-out results locked; do not mutate old benchmarks.
- If redesign is needed, create a new named synthetic benchmark rather than rewriting prior data.
- Do not promote any model from F067/F068.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F067 residual-selector family closure



# Session Amendment 097: Data-Contract Calibration Redirect

## Trigger
`F067_CLOSE_RESIDUAL_SELECTOR_FAMILY_PIVOT_TO_DATA_CONTRACT`

## Evidence
F067 closed residual selectors under the current descriptor-aligned split contract. The next smallest evidence-based step is to test whether train-internal calibration can become predictive under a different synthetic perturbation holdout contract.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F068 is diagnostic and cannot promote.

## New Diagnostic Branch
`F068_DATA_CONTRACT_CALIBRATION_REDIRECT`

## Implementation Tasks
- Keep `synth_program_aligned_genetic_lite` locked.
- Add/use separate named synthetic split-contract configs.
- Compare extrapolative index, random perturbation, and stratified program holdout contracts.
- Use true `z_bio` and train-only observed RNA PCA representations.
- Fit PCA/ridge only on train rows for each split contract.
- Score heldout perturbations only after fitting.
- Report train-inner versus real-heldout transition/delta/recall optimism.
- Do not train JEPA and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F068 data-contract calibration redirect



# Session Amendment 098: Representation Noise Amplification Audit

## Trigger
`F068_TRAIN_REAL_MISMATCH_PERSISTS_ACROSS_SPLIT_CONTRACTS`

## Evidence
F068 found that extrapolative, random, and stratified split contracts all retain train-real calibration mismatch. The policy table suggests observed RNA PCA may amplify mismatch relative to true biological latents.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F069 is diagnostic and cannot promote.

## New Diagnostic Branch
`F069_REPRESENTATION_NOISE_AUDIT`

## Implementation Tasks
- Read F068 policy summary.
- Compare true `z_bio` versus observed RNA PCA calibration optimism for each split contract.
- Quantify transition and delta optimism amplification.
- Decide whether the next family should be representation denoising/latent recovery rather than another split or residual selector.
- Do not train JEPA and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F069 representation-noise amplification audit



# Session Amendment 099: Denoised Representation Ceiling Diagnostic

## Trigger
`F069_OBSERVED_RNA_REPRESENTATION_AMPLIFIES_CALIBRATION_MISMATCH`

## Evidence
F069 showed that observed RNA PCA amplifies train-real calibration mismatch relative to true synthetic `z_bio`. The next step should identify whether a train-only denoising transform or oracle clean RNA can reduce that mismatch before training another representation model.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F070 is diagnostic and cannot promote.

## New Diagnostic Branch
`F070_DENOISED_REPRESENTATION_CEILING`

## Implementation Tasks
- Reuse the F068 split-contract benchmarks and seeds.
- Compare observed RNA PCA, observed-count PCA, train-only batch-residualized RNA PCA, clean RNA PCA, and true `z_bio`.
- Fit PCA/ridge/batch residualization from train rows only.
- Use held-out perturbation rows only for scoring.
- Report train-inner versus real-heldout optimism, latent rank, perturbation probe, and batch probe.
- Do not train JEPA and do not promote.

## Decision Use
If a train-only denoised candidate reduces mismatch, design a small representation repair branch around that target. If only clean RNA or true `z_bio` helps, prioritize learned biological denoising rather than more transition residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F070 denoised representation ceiling diagnostic



# Session Amendment 100: z_bio Observability Audit

## Trigger
`F070_TRUE_ZBIO_ONLY_REDUCES_MISMATCH_RNA_OBSERVATION_LIMIT`

## Evidence
F070 showed that observed RNA PCA, observed-count PCA, and train-only batch residualization do not reduce the calibration mismatch. Only true synthetic `z_bio` clearly reduces both transition and delta optimism, while clean RNA helps delta but not transition. Before training another JEPA representation, the loop must test whether `z_bio` is recoverable from the available observed modalities at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F071 is diagnostic and cannot promote.

## New Diagnostic Branch
`F071_ZBIO_OBSERVABILITY_AUDIT`

## Implementation Tasks
- Reuse the F068/F070 split-contract benchmarks and seeds.
- Fit train-only ridge probes from observed RNA, batch-residualized RNA, clean RNA, image, and RNA+image features to synthetic `z_bio`.
- Apply probes to eval rows without refitting or using eval targets.
- Rerun train-inner versus real-heldout transition/delta/recall optimism on the recovered latent states.
- Report eval cell-level `z_bio` recovery cosine, latent rank, perturbation probe, and batch probe.
- Do not promote and do not use this supervised oracle path as a candidate main representation.

## Decision Use
If train-only observation-to-`z_bio` recovery reduces mismatch, design a self-supervised/cross-modal representation repair branch to approximate this target without oracle labels. If not, pivot to generator/data-contract redesign before more architecture search.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F071 train-only z_bio observability audit



# Session Amendment 101: Non-Oracle Image Latent Ceiling Audit

## Trigger
`F071_CROSS_MODAL_ZBIO_RECOVERY_REDUCES_MISMATCH`

## Evidence
F071 showed that image observations can recover synthetic `z_bio` nearly perfectly using train-only oracle supervision, while RNA-only recovery is much weaker. The next step must remove the oracle target and test whether image-derived latents themselves carry the transition geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F072 is diagnostic and cannot promote.

## New Diagnostic Branch
`F072_IMAGE_LATENT_CEILING_AUDIT`

## Implementation Tasks
- Reuse the F068-F071 split-contract benchmarks and seeds.
- Build train-only image PCA, RNA+image concat PCA, RNA-to-image PCA ridge, and image-to-RNA PCA ridge latents.
- Rerun train-inner versus real-heldout transition/delta/recall optimism.
- Report image latent recovery cosine for cross-modal ridge branches.
- Do not use synthetic `z_bio` targets in non-oracle branches.
- Do not train JEPA and do not promote.

## Decision Use
If non-oracle image latents reduce mismatch, reopen a small cross-modal JEPA representation repair branch centered on image-teacher/RNA-student alignment. If not, the useful image signal requires more structured objectives than PCA/ridge, and the next branch should implement a tiny image-teacher JEPA with strict floor checks.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F072 non-oracle image latent ceiling audit



# Session Amendment 102: Image-Teacher RNA-Student JEPA Repair

## Trigger
`F072_IMAGE_LATENT_NONORACLE_REDUCES_MISMATCH`

## Evidence
F072 showed that non-oracle image PCA reduces calibration mismatch versus observed RNA PCA, but train-only linear RNA-to-image PCA prediction underfits and worsens delta optimism. The next smallest architecture step is a masked-RNA student/predictor trained against stop-gradient image-teacher targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F073 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F073_IMAGE_TEACHER_RNA_STUDENT_JEPA`

## Implementation Tasks
- Fit image PCA on train rows only.
- Train a tiny masked-RNA student encoder plus predictor on train rows only.
- Use stop-gradient image PCA targets and JEPA-dominant cosine loss with a light MSE/variance anchor.
- Score the learned RNA-side predicted image latent under the same train-inner versus real-heldout transition/delta/recall optimism audit.
- Compare against observed RNA PCA and image PCA teacher ceiling.
- Do not use synthetic `z_bio`, `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F073 preserves image-teacher signal, reopen a small cross-modal BioGuard-WM-JEPA representation branch. If it underfits, run a failure localization audit before increasing model size.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F073 image-teacher RNA-student JEPA repair



# Session Amendment 103: Image-Teacher Failure Localization

## Trigger
`F073_RNA_STUDENT_IMAGE_TEACHER_WEAK_OR_UNSTABLE`

## Evidence
F073 trained a tiny masked-RNA student against stop-gradient train-only image PCA targets. It achieved nontrivial target alignment but failed to preserve the image-teacher transition floor and recall. Before increasing capacity, the loop must localize whether the failure is image-target underfit or loss of perturbation structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F074 is diagnostic and cannot promote.

## New Diagnostic Branch
`F074_IMAGE_TEACHER_FAILURE_LOCALIZATION`

## Implementation Tasks
- Read F073 seed metrics and training trace.
- Compare candidate versus image teacher and observed RNA by policy and seed.
- Quantify image-target cosine, train prediction-target cosine, recall gap, delta cosine gap, transition gap, perturbation-probe gap, and batch-probe gap.
- Do not train, do not select a model, and do not promote.

## Decision Use
If target underfit dominates, consider a modest capacity/optimization repair. If perturbation structure is lost despite target alignment, add condition/program-preserving and multi-positive image-teacher objectives before any full JEPA wrapper.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F074 image-teacher failure localization



# Session Amendment 104: Relational Image-Teacher RNA-Student JEPA

## Trigger
`F074_FAILURE_IS_PERTURBATION_STRUCTURE_LOSS`

## Evidence
F074 showed the F073 RNA student did not simply underfit image targets. It reached moderate image-latent cosine but lost perturbation-probe signal, recall, and transition geometry relative to the image teacher. The next smallest repair is to preserve image-teacher pairwise structure during RNA-student training.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F075 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F075_RELATIONAL_IMAGE_TEACHER_RNA_STUDENT`

## Implementation Tasks
- Keep the F073 masked-RNA student and stop-gradient train-only image PCA teacher.
- Add a train-only pairwise teacher-similarity distillation term on mini-batches.
- Keep JEPA/cosine endpoint prediction dominant with light MSE and variance anchors.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare directly against F073 and the image-teacher ceiling.
- Do not use synthetic `z_bio`, `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If relational training preserves image-teacher transition structure, reopen a small cross-modal BioGuard-WM-JEPA representation branch. If it only partially repairs structure, localize whether condition grouping or action descriptors are needed. If it fails, pivot away from per-cell image-teacher distillation.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F075 relational image-teacher RNA-student JEPA



# Session Amendment 105: Condition-Centroid Image-Teacher RNA-Student

## Trigger
`F075_RELATIONAL_IMAGE_TEACHER_STILL_LOSES_STRUCTURE`

## Evidence
F075 added pairwise image-teacher structure but still lost perturbation/action geometry relative to the image teacher. The remaining local hypothesis is that per-cell image targets are too noisy for the small RNA student, so the next diagnostic should test train-only condition-centroid image targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F076 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F076_CONDITION_CENTROID_IMAGE_TEACHER`

## Implementation Tasks
- Fit image PCA on train rows only.
- Compute train-only image-target centroids by perturbation ID, cell line, and dose.
- Train the masked-RNA student against these centroid targets with no label features as inputs.
- Use a light relational term only inside train mini-batches.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare against F075 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F076 repairs structure, reopen representation learning with train-only pseudobulk teachers. If it fails, pivot toward action-descriptor or source-state objectives because per-cell and condition-level image distillation both failed.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F076 condition-centroid image-teacher RNA-student



# Session Amendment 106: Source-State Plus Program-Action Image Teacher

## Trigger
`F076_CONDITION_CENTROID_TEACHER_STILL_LOSES_STRUCTURE`

## Evidence
F073-F076 all failed to transfer image-teacher perturbation structure into an RNA-side endpoint representation. Per-cell targets, pairwise relational image targets, and train-only condition-centroid targets all preserved only moderate image alignment while losing recall and perturbation-probe signal.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F077 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F077_SOURCE_PROGRAM_IMAGE_TEACHER`

## Implementation Tasks
- Build train-only control/source expression centroids by cell line and dose.
- Use coarse program-action one-hot descriptors only; do not use exact perturbation one-hot features.
- Predict train-only condition-centroid image PCA targets from source state plus program action.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, and image-latent cosine.
- Compare against F076 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F077 repairs structure, build a real source/action-conditioned JEPA wrapper around this factorization. If it fails, the issue is not endpoint noise or missing source-action factorization alone; pivot to action descriptor adequacy or synthetic benchmark identifiability audits.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F077 source-state plus program-action image teacher



# Session Amendment 107: Source-Program Failure Localization

## Trigger
`F077_SOURCE_PROGRAM_ACTION_REPAIRS_STRUCTURE_BUT_WEAK`

## Evidence
F077 was the first branch to repair heldout recall and improve delta cosine relative to F076, but it still had very low transition improvement, low effective rank, and almost no perturbation-probe signal. Before adding capacity or descriptors, the loop must localize the remaining failure mode.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F078 is diagnostic and cannot promote.

## New Diagnostic Branch
`F078_SOURCE_PROGRAM_FAILURE_LOCALIZATION`

## Implementation Tasks
- Read F077 seed metrics and training traces.
- Compare candidate versus image teacher and observed RNA by policy and seed.
- Quantify recall gap, delta cosine gap, transition gap, effective-rank gap, perturbation-probe gap, batch-probe gap, and image-target cosine.
- Do not train, do not select a model, and do not promote.

## Decision Use
If recall is high but transition/action signal is low, pivot to delta/source-improvement objectives or richer non-exact action descriptors. If low-rank prototype collapse dominates, add rank/adversarial diversity constraints before a full JEPA wrapper.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F078 source-program failure localization



# Session Amendment 108: Source Delta Rank Repair

## Trigger
`F078_SOURCE_PROGRAM_RECALL_WITH_LOW_TRANSITION_AND_ACTION_SIGNAL`

## Evidence
F078 localized the F077 weak pass as high recall but low transition improvement, low perturbation-probe signal, and low effective rank. This suggests endpoint/prototype prediction is insufficient; the next diagnostic must directly optimize source improvement and delta geometry before trying richer action descriptors or a full JEPA wrapper.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F079 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F079_SOURCE_DELTA_RANK_REPAIR`

## Implementation Tasks
- Keep coarse program-action descriptors only; do not use exact perturbation one-hot features.
- Use train-only control expression centroids as source state.
- Use train-only control image PCA centroids as source teacher targets.
- Use train-only condition-centroid image PCA as perturbed teacher targets.
- Add source endpoint, target endpoint, delta-direction, source-improvement hinge, magnitude, relational, and delta-rank variance losses.
- Score transition, delta, recall, effective rank, perturbation probe, batch probe, image-latent cosine, and train-loss diagnostics.
- Compare against F077 and the image-teacher ceiling.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F079 repairs transition while preserving recall/delta, build a real source/action-conditioned JEPA wrapper. If transition stays low, pivot to action descriptor adequacy or benchmark identifiability audits.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F079 source/delta/rank-constrained program repair



# Session Amendment 109: Source Delta Rank JEPA Wrapper

## Trigger
`F079_SOURCE_DELTA_RANK_REPAIR_READY_FOR_WRAPPER`

## Evidence
F079 repaired the source-program branch enough to justify the previously deferred full wrapper: transition improved to a useful Tier 1 diagnostic level while delta cosine, recall, and rank were preserved. The next falsifiable question is whether that repair remains valid when implemented as a real cross-modal, action-conditioned JEPA rather than a standalone image-teacher predictor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F080 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F080_SOURCE_DELTA_RANK_JEPA_WRAPPER`

## Implementation Tasks
- Use ProgramBootstrapJEPA online RNA/image encoders and EMA target encoders.
- Preserve stop-gradient latent teachers and query-conditioned predictors.
- Include RNA->image, image->RNA, and control+coarse-program-action transition losses.
- Carry forward F079 source teacher, target teacher, delta-direction, source-improvement, magnitude, relational, and delta-rank constraints.
- Use train-only control expression/image PCA centroids for source state/teacher.
- Use train-only condition-centroid image PCA for training target anchors.
- Use coarse program one-hot action descriptors only; do not use exact perturbation one-hot.
- Score learned-latent transition and image-teacher-aligned transition separately.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F080 preserves F079 transition signal and cross-modal retrieval, design a low-compute Tier 2/no-regression validation. If the full wrapper loses transition signal, localize whether EMA teacher lag, target anchoring, or coarse action descriptors are the bottleneck.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F080 full source/delta/rank JEPA wrapper



# Session Amendment 110: Train-Only Delta-Calibrated JEPA Wrapper

## Trigger
`F080_FULL_JEPA_WRAPPER_MIXED_OR_INCONCLUSIVE`

## Evidence
F080 was not a clean pass: it improved image-teacher transition and kept recall, but delta cosine remained below the gate. The magnitude ratio was also high, suggesting a transition endpoint that is useful but geometrically miscalibrated in delta space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F081 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F081_DELTA_CALIBRATED_JEPA_WRAPPER`

## Implementation Tasks
- Reuse the F080 real ProgramBootstrapJEPA path.
- Fit a train-only ridge correction from JEPA predicted delta to train image-PCA teacher delta.
- Apply the fixed calibrator to heldout perturbation JEPA predicted deltas.
- Report raw versus calibrated transition, delta cosine, recall, rank, magnitude, action-negative gap, and train/eval calibration gap.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F081 repairs heldout delta without losing transition/recall, convert the calibrator into an internal JEPA delta-head or loss. If it overfits train direction, pivot to action descriptor adequacy or environment-stable delta targets.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F081 train-only JEPA delta calibrator



# Session Amendment 111: Fresh-Seed Delta-Calibrated JEPA Validation

## Trigger
`F081_DELTA_CALIBRATED_JEPA_TIER1_PASS_NONPROMOTING`

## Evidence
F081 passed a Tier 1 diagnostic by showing that F080's JEPA predicted deltas contain train-only linearly recoverable heldout delta direction. Because Tier 1 cannot promote, the next autonomous step is fresh-seed validation under the same no-leakage rules.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F082 is Tier 2-style validation and cannot promote.

## New Branch
`F082_DELTA_CALIBRATED_TIER2_VALIDATION`

## Implementation Tasks
- Rerun the F081 real JEPA plus train-only delta calibrator on fresh seeds 37-42.
- Keep the same three split policies.
- Report mean/std/min transition, delta cosine, recall, rank, magnitude, cross-modal retrieval, and action-negative gap.
- Treat any seed/policy collapse as a pivot event.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F082 is stable, write a Tier 3 design amendment with no-regression validators and a locked candidate code path. If it is unstable, localize the failing policy/seed before changing architecture.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F082 fresh-seed delta-calibrated JEPA validation



# Session Amendment 112: Tier 3 No-Regression Preflight

## Trigger
`F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`

## Evidence
F082 produced a fresh-seed non-promoting Tier 2-style pass. The autonomy protocol allows Tier 2 to justify Tier 3 design only; it cannot promote the candidate. Before launching a Tier 3 run, all no-regression gates must be made explicit, including the locked delta-rank floor and availability of a paired external validator.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Branch
`F083_TIER3_PREFLIGHT`

## Implementation Tasks
- Read F082 metrics.
- Compare transition, delta cosine, recall, delta rank, and magnitude ratio against locked floors.
- Check cross-modal retrieval gates.
- Check paired external validator availability; Norman RNA-only can be noted but cannot validate imaging.
- Lock candidate code path for any future Tier 3.
- Do not train and do not promote.

## Decision Use
If all gates pass, launch a locked low-compute Tier 3 validation. If rank or validator gates fail, pivot to the smallest targeted repair or validator discovery branch.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F083 Tier 3 no-regression preflight



# Session Amendment 113: Rank-Floor Comparability Audit

## Trigger
`F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS`

## Evidence
F083 showed F082 passes transition, delta cosine, recall, magnitude, and cross-modal gates but fails the locked delta-rank no-regression gate. Before adding rank capacity, the loop must determine whether the locked rank floor is comparable to the current image-teacher latent target space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F084_RANK_FLOOR_COMPARABILITY`

## Implementation Tasks
- Read F082 per-seed metrics.
- Compare calibrated delta rank, raw direct delta rank, image-teacher delta rank, and target-image effective rank against the locked rank floor.
- Decide whether the rank failure is repairable in the current target space or requires a representation-specific floor registry before Tier 3.
- Do not train and do not promote.

## Decision Use
If the teacher/target rank ceiling is below the locked floor, reproduce a train-only floor inside the current image-teacher latent registry before Tier 3. If the teacher rank supports the floor but the calibrator loses rank, implement a rank-preserving calibrator.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F084 rank-floor comparability audit



# Session Amendment 114: Current Image-Teacher Latent Floor Registry

## Trigger
`F084_LOCKED_RANK_FLOOR_EXCEEDS_IMAGE_TEACHER_CEILING`

## Evidence
F084 showed the locked delta-rank floor is above the current image-teacher target ceiling. This indicates a cross-representation baseline mismatch. The protocol allows a fresh baseline discrepancy to be documented rather than silently picking a convenient number.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F085_CURRENT_LATENT_FLOOR_REGISTRY`

## Implementation Tasks
- Rebuild the current train-only image-teacher latent tables for F082 seeds and split policies.
- Fit train-only action ridge floors in that current latent registry.
- Compute true delta effective-rank ceilings in the same registry.
- Compare F082 calibrated JEPA against those current-registry floors.
- Do not train and do not promote.

## Decision Use
If F085 shows the candidate preserves the current-registry floor and rank ceiling, design Tier 3 with an explicit representation-specific rank registry plus external-validator plan. If it shows a current-registry rank gap, implement a rank-preserving calibrator.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F085 current-latent floor registry



# Session Amendment 115: Current-Registry Tier 3 Design And Validator Preflight

## Trigger
`F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE`

## Evidence
F085 showed that the F082 delta-calibrated JEPA wrapper preserves the current image-teacher latent floor under a representation-specific registry, while the old locked rank floor is not comparable to that target geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F086_CURRENT_REGISTRY_TIER3_DESIGN`

## Implementation Tasks
- Read F082 and F085 metrics.
- Lock corrected current-registry gates for transition, delta cosine, recall, and rank.
- Carry forward F082 cross-modal RNA->image and image->RNA retrieval gates.
- Check local paired scRNA plus imaging validator availability.
- Record external validator candidates without downloading large datasets.
- Do not train and do not promote.

## Decision Use
If F086 finds all synthetic gates pass and a paired validator is locally ready, launch a locked low-compute Tier 3 run. If only an external paired validator candidate exists, write an adapter/ingest preflight next. If no paired validator exists, pivot to validator discovery and keep Norman RNA-only as non-promoting diagnostics.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F086 current-registry Tier 3 design



# Session Amendment 116: scGeneScope Metadata Adapter Preflight

## Trigger
`F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED`

## Evidence
F086 passed corrected synthetic current-registry gates but could not launch Tier 3 because no paired scRNA plus imaging validator is local. Public source pages identify scGeneScope as the best future condition-paired scRNA-seq plus Cell Painting validator candidate, but the dataset is large, so ingestion must start with metadata and split contracts.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F087_SCGENESCOPE_ADAPTER_PREFLIGHT`

## Implementation Tasks
- Add a metadata-only scGeneScope adapter contract.
- Require modality, treatment, dose, round, batch, replicate, split, and URI columns.
- Reject `condition_key`, `biological_key`, exact target-key, and target-key shortcut manifest columns.
- Build condition-pair tables from metadata only.
- Audit whether a local manifest exists; do not download large image/count payloads.
- Do not train and do not promote.

## Decision Use
If a local manifest validates, run a low-compute Tier 3 dry run on profile/embedding-level features. If no manifest is local, keep building adapter documentation and avoid full dataset download until metadata access is solved.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F087 scGeneScope metadata adapter preflight



# Session Amendment 117: Remote scGeneScope Metadata Discovery

## Trigger
`F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL`

## Evidence
F087 proved the local scGeneScope adapter contract but found no local dataset root. The next step is remote metadata discovery only, because scGeneScope is large and the loop must preserve low-compute and storage-safety constraints.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F088_SCGENESCOPE_REMOTE_DISCOVERY`

## Implementation Tasks
- Query Hugging Face dataset tree metadata for scGeneScope without downloading payload files.
- Identify small manifest, metadata, split, treatment, or sample files if exposed.
- Record visible RNA and imaging feature-file families and sizes.
- Determine whether a feature-level Tier 3 dry run is feasible under low-compute constraints.
- Do not train, do not download large payloads, and do not promote.

## Decision Use
If light metadata is found, build a manifest from it. If only large H5AD/image payloads are visible, search official supplementary/code metadata next or wait for a user-supplied manifest while continuing non-downloadable adapter work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F088 remote scGeneScope metadata discovery



# Session Amendment 118: Official scGeneScope Supplement Metadata Harvest

## Trigger
`F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE`

## Evidence
F088 showed the Hugging Face file tree is accessible but exposes no light manifest and the smallest visible paired RNA/image feature footprint is about 13.6 GB. The NeurIPS proceedings page exposes a small official supplemental archive, so the next low-compute step is to harvest metadata from that archive without downloading dataset payloads.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F089_SCGENESCOPE_SUPPLEMENT_HARVEST`

## Implementation Tasks
- Download only the small official supplemental archive under a strict byte cap.
- Extract Croissant metadata, code README, split contract, and file inventory.
- Update the adapter contract if metadata contradicts assumptions such as dose availability.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If Croissant metadata provides split/field contracts, use it to validate the adapter and design a capped feature-level dry run. If metadata is insufficient, search the official code repo for manifest builders next.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F089 official scGeneScope supplement metadata harvest



# Session Amendment 119: scGeneScope Croissant Contract Validation

## Trigger
`F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED`

## Evidence
F089 recovered Croissant metadata and the official replicate-based split contract. Before any feature download, the adapter must prove it can map those fields into the project's no-leakage condition-paired manifest contract.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F090_SCGENESCOPE_CROISSANT_CONTRACT`

## Implementation Tasks
- Validate required Croissant fields: `cell_id`, `Treatment`, `Replicate`, `batch`, and `Group`.
- Validate replicate split mapping: `3=train`, `5=validation`, `4=test`, `1/2=alternate_test`.
- Build a dry-run paired RNA/image manifest with fixed dose and no payload access.
- Build condition-pair tables for train, validation, test, and alternate-test splits.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If F090 passes, design a storage-gated feature-level preflight. If it fails, fix the adapter contract before any feature-level dry run.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F090 scGeneScope Croissant contract validation



# Session Amendment 120: scGeneScope Feature Storage And Backed-IO Preflight

## Trigger
`F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE`

## Evidence
F090 validated the scGeneScope Croissant field contract and replicate split mapping without touching payload files. The remaining bottleneck is whether the smallest paired feature files can be inspected safely under low-compute and storage/RAM constraints.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F091_SCGENESCOPE_FEATURE_PREFLIGHT`

## Implementation Tasks
- Read the saved F088 Hugging Face tree summary and F090 contract metrics.
- Pair RNA and image feature H5AD candidates by round.
- Estimate smallest paired feature footprint, disk buffer, backed-IO RAM requirement, and dependency readiness.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If all gates pass, the next permissible step is an obs-only/backed H5AD open dry run with strict byte accounting. If any gate fails, require a smaller manifest-backed feature subset or continue metadata-only validation work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F091 scGeneScope feature storage/backed-IO preflight



# Session Amendment 121: scGeneScope Obs-Only Backed Dry Run

## Trigger
`F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN`

## Evidence
F091 found the smallest paired scGeneScope feature footprint is within the registered low-compute cap and the current workspace has enough disk/RAM with `anndata` and `h5py` available. The next step is to prove the actual files can be opened safely in backed mode before any model code touches them.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F092_SCGENESCOPE_OBS_ONLY_DRY_RUN`

## Implementation Tasks
- Download only the F091-approved smallest RNA and image feature H5AD files if not already local.
- Enforce the F091 byte cap before attempting download.
- Open each H5AD with `backed="r"` and inspect obs metadata only.
- Verify required Croissant-derived obs columns are present in both modalities.
- Do not train, fit encoders, fit calibrators, compute target means, whiten, score JEPA metrics, or promote.

## Decision Use
If F092 passes, run a shape/split/pairing audit on backed feature metadata next. If download/access fails because of license or auth gating, write a hard escalation report. If obs contract fails, return to adapter repair or smaller manifest construction.

## Hard Escalation Check
No hard escalation trigger present before launch.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F092 scGeneScope obs-only backed dry run



# Session Amendment 122: Split-Safe JEPA Calibration Abstention Gate

## Trigger
`F093_CALIBRATION_AND_DESCRIPTOR_REPAIR_REQUIRED`

## Evidence
F082 external validation failed without identity or leakage violations. F093 showed that train-only delta calibration consistently damages held-out delta cosine relative to the protected floor, while the uncalibrated JEPA output preserves delta direction better but is not transition-floor-safe on the external round shift.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real Tier 3 pass explicitly supersedes it.

## New Repair Branch
`F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`

## Implementation Tasks
- Add a train-only multiobjective calibration gate for the JEPA path.
- The gate may select JEPA raw, JEPA calibrated, or a train-only JEPA blend.
- The gate must not use PLS/full-ridge as a candidate representation path or fallback output.
- Use the protected floor only as an audit threshold.
- Select gate parameters using train/internal replicate evidence only.
- Report transition improvement, delta cosine, recall@1, RNA->image retrieval, image->RNA retrieval, rank, leakage, identity, and floor gaps on validation, test, and alternate_test.
- Keep source-as-target, protected full-ridge floor, and no-residual baselines.

## Decision Use
If F094 clears floor-safe transition and delta cosine without identity or leakage violations, rerun the full F082 scGeneScope external report and only then consider Tier 3 pass language. If F094 still fails, execute the descriptor repair branch `F095_NON_EXACT_ACTION_DESCRIPTOR_UPGRADE` before any architecture redesign.

## Do-Not-Run List
Do not train a new architecture before F094. Do not promote. Do not use condition_key, biological_key, exact treatment one-hot, held-out target means, pooled train+test statistics, or protected floor predictions as model outputs.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F094 split-safe JEPA calibration abstention gate



# Session Amendment 123: Non-Exact PubChem Fingerprint Descriptor Rerun

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`

## Evidence
F094 selected raw JEPA for every seed, which restored delta-cosine floor safety but did not restore transition-floor safety on the harder external splits. The action input remains a 12-dimensional scalar PubChem descriptor with limited perturbation mechanism content.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real Tier 3 pass explicitly supersedes it.

## New Repair Branch
`F095_NON_EXACT_PUBCHEM_FINGERPRINT_DESCRIPTOR_RERUN`

## Implementation Tasks
- Add a `pubchem_fingerprint` descriptor mode using public non-exact PubChem structure fingerprints plus scalar properties and missingness flags.
- Do not use condition_key, biological_key, exact treatment one-hot, held-out target means, pooled train+test statistics, or protected floor predictions as model outputs.
- Keep the F094 split-safe JEPA calibration gate.
- Run on GPU unless unavailable or occupied.
- Report the same external metrics and baselines as F082/F094.

## Decision Use
If F095 clears transition, delta cosine, and recall floor gaps on all external splits with identity/leakage checks clean, write a non-promoting Tier 3 pass report for review. If F095 fails, move to cross-modal representation repair only after another council.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F095 non-exact PubChem fingerprint descriptor rerun



# Session Amendment 124: Frozen Selector Confirmation Or Gate Contract Audit

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from `F095_NON_EXACT_PUBCHEM_FINGERPRINT_DESCRIPTOR_RERUN`

## Evidence
F095 showed that non-exact PubChem fingerprints materially improve the scGeneScope external floor comparison. The official split-safe gate still failed by a small alternate-test recall gap. The calibrated fingerprint row cleared all floor gaps, but it was not the predeclared selected candidate and cannot be promoted post hoc.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real Tier 3 pass explicitly supersedes it.

## New Branch
`F096_FROZEN_SELECTOR_CONFIRMATION_OR_GATE_CONTRACT_AUDIT`

## Implementation Tasks
- Do not promote F095.
- Freeze the useful candidate lesson: PubChem fingerprint descriptors plus train-only JEPA calibration are the next candidate family.
- Before any promotion language, either validate a frozen selector on a genuinely new external protocol or run a non-promoting gate-contract audit.
- The gate audit must explain whether action-heldout train CV is the wrong selector for same-treatment replicate/round external validation.
- Keep PLS/full-ridge as audit floor only.
- Continue to run model work on GPU unless unavailable or occupied.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F096 frozen selector confirmation or gate contract audit



# Session Amendment 125: Rosetta Source Geometry Audit

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from `F098_CPG0003_ROSETTA_REPLICATE_HOLDOUT`

## Evidence
F098 used the frozen F096/F082 ProgramBootstrapJEPA path with non-exact
SMILES/dose descriptors and train-only delta calibration on a fresh cpg0003
Rosetta same-condition replicate-holdout protocol. It failed promotion gates:
test recall was below the audit floor and absolute transition improvement was
negative on every split.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F099_ROSETTA_SOURCE_GEOMETRY_AUDIT`

## Implementation Tasks
- Do not train a new model.
- Do not promote F097 or F098.
- Read only F097/F098 saved artifacts and compact metadata.
- Quantify source-as-target, target-target, and control-source geometry.
- Determine whether cpg0003 L1000/Cell Painting profiles are already centered
  signatures where the current control source definition is mismatched.
- Report whether the failure is a model failure, a source-state contract failure,
  or a validator-mismatch failure.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F099 Rosetta source geometry audit



# Session Amendment 126: Strict scRNA Imaging Fresh Dataset Preflight

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from `F101_CPG0003_ROSETTA_SMALL_SCALE_CALIBRATION`

## Evidence
Rosetta repairs did not produce a registered pass. F099 diagnosed a source-state
contract and validator mismatch. F100 zero-signature source failed. F101
train-only small-scale calibration fixed delta cosine but left transition
improvement slightly negative on all splits.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`

## Implementation Tasks
- Do not promote F097, F098, F100, or F101.
- Do not redesign the model from the Rosetta failure.
- Treat cpg0003 Rosetta as an auxiliary L1000 plus Cell Painting stress test,
  not as strict paired scRNA plus imaging validation.
- Search for or recover a strict paired scRNA plus imaging fresh validation
  protocol, preferably scGeneScope with an unused sealed split or another public
  paired dataset.
- Run only metadata, obs-only, backed, or manifest-level checks first.
- Keep raw data and checkpoints outside git.
- Use GPU for model runs unless unavailable or occupied.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
If no strict fresh paired scRNA plus imaging dataset is available, document that
as a validation blocker rather than promoting from Rosetta.

Council-selected proposal: F102 strict scRNA imaging fresh dataset preflight



# Session Amendment 127: PerturbMulti RNA Obs And Pairing Preflight

## Trigger
`F102_STRICT_SCRNA_IMAGING_CANDIDATE_FOUND_RNA_OBS_PREFLIGHT_PENDING`

## Evidence
PerturbMulti is the first public strict paired candidate found after stopping
the Rosetta loop. Its manifest includes CRISPR-screen RNA H5AD, protein-intensity
H5AD, perturbation metadata, spatial coordinates, and cell-ID-keyed image tar
archives. A backed obs/schema probe of the small protein H5AD passed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT`

## Implementation Tasks
- Do not train or promote.
- Download only the CRISPR RNA H5AD to ignored storage if storage permits.
- Inspect RNA obs with backed or HDF5-level access; do not load `.X`.
- Confirm RNA obs columns for cell ID, perturbation, coordinates, and batch/split
  design.
- Confirm metadata-only overlap among RNA cell IDs, protein H5AD cell IDs, and
  image tar member IDs.
- Only after pairing passes, design a small sealed split and run the frozen
  F082/F096 ProgramBootstrapJEPA path on GPU.

## Hard Escalation Check
No hard escalation trigger present, but available RAM was low during F102.

## Continuation Rule
If RNA obs or image-key pairing fails, document validation blocked rather than
promoting from Rosetta or F096.

Council-selected proposal: F103 PerturbMulti RNA obs and pairing preflight
