# Debate Council 123

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from F094 split-safe JEPA calibration gate

## Evidence Summary
F094 ran on GPU with the ProgramBootstrapJEPA / BioGuard-WM path and reused the F082 condition cache. The gate selected `blend_alpha=0.0` for seeds 37, 38, and 39, meaning it abstained from delta calibration and preserved raw JEPA output. This repaired the main F082 pathology: delta cosine was above the protected floor on all external splits.

The remaining failure is transition strength under external replicate and round shift. `alternate_test` had `floor_gap_transition=-0.090531` and `floor_gap_recall=-0.012121`; `test` had `floor_gap_transition=-0.029148`; only `validation` cleared transition, delta cosine, and recall. The current action descriptor is still 12 scalar PubChem/found/control features, with weak mechanism content.

## Independent Proposals
- Architect: keep the same JEPA architecture and add a non-exact descriptor branch using PubChem fingerprints plus scalar properties, with train-only scaling and the same calibration gate.
- Skeptic: fingerprint descriptors may overfit because there are only 28 train non-control conditions; require a train-only descriptor audit and keep F094 raw/gated comparison.
- Methodologist: run F095 as an external validation rerun with identical splits, seeds, baselines, source contract, and pass/fail criteria; candidate selection must not use external metrics.
- Biologist: chemical structure fingerprints are a better perturbation prior than scalar properties, but mechanism gaps may remain for compounds whose phenotypic effects are target/pathway-mediated rather than structure-neighbor-mediated.
- Monitor: no hard escalation trigger is present; F094 used CUDA; no promotion; PLS/full-ridge remains only an audit floor.

## Steelman Of Opposing Proposals
- Architecture-repair steelman: cross-modal retrieval remains weak, so a representation-level repair may ultimately be required. It is premature because the action input is visibly underpowered and can be improved without changing architecture.
- Descriptor-skeptic steelman: adding many fingerprint bits could raise variance and make train-only gating more conservative. This is acceptable if F095 is framed as a falsifiable descriptor audit and not a promotion attempt.

## Three-Round Debate Summary
Round 1 recognized that calibration repair solved directional damage but not transition strength. Round 2 rejected immediate architecture redesign because it would confound the now-isolated action-capacity failure. Round 3 selected a descriptor upgrade with strict train-only scaling, no exact treatment one-hot, and the existing F094 gate.

## Scoring Table
High leakage-risk scores mean safer, lower leakage risk.

| proposal | novelty against current loop | feasibility | identity preservation | expected effect size | falsifiability | leakage risk | compute cost | average | minimum | next action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F095 PubChem fingerprint descriptor rerun | 0.78 | 0.78 | 0.95 | 0.70 | 0.90 | 0.82 | 0.75 | 0.811 | 0.70 | Add non-exact PubChem fingerprint descriptors and rerun the F094 gate on CUDA. |
| F096 cross-modal architecture repair | 0.82 | 0.45 | 0.70 | 0.70 | 0.75 | 0.58 | 0.40 | 0.629 | 0.40 | Defer until descriptor capacity is falsified. |
| Tune gate on external splits | 0.20 | 0.60 | 0.80 | 0.80 | 0.30 | 0.05 | 0.90 | 0.521 | 0.05 | Reject as leakage. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F095_NON_EXACT_PUBCHEM_FINGERPRINT_DESCRIPTOR_RERUN`.

## Next Experiment Command Or Implementation Target
Extend `scripts/run_f082_scgenescope_external_validation.py` with `--descriptor-mode pubchem_fingerprint`, keep `--gate-mode split_safe`, and rerun on CUDA with the same seeds, source-as-target, protected full-ridge floor, no-residual baseline, leakage checks, and identity checks.
