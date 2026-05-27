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
