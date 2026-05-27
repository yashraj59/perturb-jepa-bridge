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
