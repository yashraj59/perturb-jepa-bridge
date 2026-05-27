# Debate Council 122

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` plus `F093_CALIBRATION_AND_DESCRIPTOR_REPAIR_REQUIRED`

## Evidence Summary
F082 was validated on the ProgramBootstrapJEPA / BioGuard-WM path against real paired scGeneScope RNA `scvi_n200` and image `imagenet/vit-l` features. The data path passed backed obs contracts, split mapping, RNA-image pairing, identity checks, and leakage checks. The candidate did not pass Tier 3 and no model is promoted.

The decisive failure is metric-level: F082 delta calibration has positive transition improvement and recall, but delta cosine is below the protected full-ridge floor in every held-out split: `-0.092407` on `alternate_test`, `-0.042797` on `test`, and `-0.016252` on `validation`. The external round shift is also not floor-safe on transition for `alternate_test` with `-0.024421`.

F093 shows why this is fixable without redesigning the architecture first. The uncalibrated JEPA output beats the floor on delta cosine in all held-out splits, while the calibrated output gives up `0.066108` to `0.113186` delta cosine versus raw. Calibration improves transition and recall, but it transfers by damaging directional biology. Descriptor coverage is also weak: the action descriptor is only 12 scalar PubChem/found/control features, with missing non-control PubChem descriptors for `PQ401` and `SKII`.

## Independent Proposals
- Architect: implement `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`, selecting among JEPA raw, JEPA calibrated, and train-only JEPA blend outputs using only train/internal replicate evidence.
- Skeptic: the failure may mostly be action descriptor insufficiency; do not tune to the external splits, and do not let an abstention gate hide a biologically weak action encoder.
- Methodologist: make the next repair a train-only multiobjective calibration gate with explicit lower-confidence checks for transition, delta cosine, recall, and negative controls.
- Biologist: add a later descriptor repair branch with non-exact public chemical structure and coarse mechanism descriptors, explicit missingness flags, and no exact treatment one-hot or condition-key encoding.
- Monitor: no hard escalation trigger is present; F093 used only saved artifacts; no raw data or checkpoints are in git; no promotion is allowed.

## Steelman Of Opposing Proposals
- Descriptor-first steelman: calibration cannot recover missing biological mechanism information, and scGeneScope perturbations are chemically diverse enough that scalar PubChem properties are likely underpowered. A descriptor upgrade may produce the largest eventual effect.
- Architecture-repair steelman: RNA->image and image->RNA retrieval are weak, so a deeper cross-modal alignment repair may be necessary. However, this should wait because the current artifact already isolates a calibration transfer failure that can be tested without architectural churn.

## Three-Round Debate Summary
Round 1 favored descriptor repair because the action vector is visibly thin and incomplete. Round 2 shifted toward calibration repair because F093 proved raw JEPA has better held-out delta direction than the calibrated output, so the immediate damage is local and testable. Round 3 selected a JEPA-only calibration abstention/blend gate first, with descriptor repair queued next if F094 cannot clear floor-safe direction and transition without external split selection.

## Scoring Table
High leakage-risk scores mean safer, lower leakage risk.

| proposal | novelty against current loop | feasibility | identity preservation | expected effect size | falsifiability | leakage risk | compute cost | average | minimum | next action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F094 split-safe JEPA calibration abstention gate | 0.75 | 0.80 | 0.95 | 0.70 | 0.90 | 0.85 | 0.85 | 0.829 | 0.70 | Implement a train-only gate that can abstain from harmful calibration or choose a JEPA blend, then rerun scGeneScope external metrics. |
| F095 non-exact action descriptor upgrade | 0.85 | 0.65 | 0.85 | 0.75 | 0.80 | 0.60 | 0.55 | 0.721 | 0.55 | Add public chemical/fingerprint/coarse mechanism descriptors with missingness flags after F094. |
| F096 cross-modal alignment repair | 0.80 | 0.45 | 0.70 | 0.70 | 0.70 | 0.55 | 0.40 | 0.614 | 0.40 | Defer until calibration and descriptor audits show remaining retrieval-specific failure. |
| Train a new architecture immediately | 0.40 | 0.35 | 0.40 | 0.55 | 0.45 | 0.25 | 0.25 | 0.379 | 0.25 | Reject: violates the requested ordering and would confound the external validation failure diagnosis. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`.

## Next Experiment Command Or Implementation Target
Implement the F094 repair in `perturb_jepa/training/bioguard_wm_calibration.py` and `scripts/run_f082_scgenescope_external_validation.py`: add a train-only multiobjective JEPA calibration gate that never uses PLS/full-ridge as the candidate output, then rerun F082 external validation with raw/calibrated/blend gate metrics and the same source-as-target, protected floor, and no-residual baselines.
