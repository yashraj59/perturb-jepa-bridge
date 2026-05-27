# Debate Council 124

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from F095 PubChem fingerprint descriptor gate

## Evidence Summary
F095 added non-exact PubChem 2D fingerprint bits to the scalar PubChem descriptor, raising the action dimension to `932` with zero missing non-control descriptor rows. The run used CUDA, reused the F082 condition cache, passed backed obs and pairing checks, and had clean identity/leakage flags.

The official F095 candidate was `F094_split_safe_jepa_calibration_gate`. It selected raw JEPA for all seeds. That candidate cleared transition and delta-cosine floor gaps on every split, but missed `alternate_test` recall by `-0.006061`, so the runner correctly returned `FAIL_EXTERNAL_NO_PROMOTION`.

The calibrated fingerprint row is scientifically important but not promotable from this run: it cleared all three floor-gap families on all external splits, but it was not the predeclared selected candidate. Selecting it now would use external validation feedback as model-selection signal.

## Independent Proposals
- Architect: freeze a fingerprint-calibrated JEPA candidate and run a new locked confirmation, preferably on a second external paired RNA+image validator or a scGeneScope protocol not used for tuning.
- Skeptic: do not promote anything from scGeneScope after iterative F082/F094/F095 repairs; the dataset has become part of the development loop.
- Methodologist: repair the gate by aligning train-only selection to the actual replicate-heldout contract, but treat that as a new non-promoting audit unless confirmed on a fresh external protocol.
- Biologist: fingerprints helped because perturbation mechanism is partly structure-linked, but target/pathway descriptors may still be required for robust biology.
- Monitor: no hard escalation trigger; no promotion; protected PLS raw-linear model of record remains active.

## Steelman Of Opposing Proposals
- Promotion steelman: the calibrated fingerprint row technically clears all registered floor gaps with clean identity and leakage checks. If it had been predeclared, it would be a plausible Tier 3 pass.
- No-promotion steelman: F095 was launched because F094 failed on the same external validator, and the calibrated row was discovered inside a repair loop. Calling it a pass would blur validation and development.

## Three-Round Debate Summary
Round 1 focused on the near-pass and whether the calibrated fingerprint row should count. Round 2 rejected post-hoc switching because the gate was predeclared and scGeneScope has now guided multiple repairs. Round 3 selected a frozen-candidate confirmation path: keep the useful descriptor result, but require a locked selector and another validation before any promotion language.

## Scoring Table
High leakage-risk scores mean safer, lower leakage risk.

| proposal | novelty against current loop | feasibility | identity preservation | expected effect size | falsifiability | leakage risk | compute cost | average | minimum | next action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F096 frozen fingerprint-calibrated confirmation protocol | 0.70 | 0.70 | 0.95 | 0.75 | 0.90 | 0.80 | 0.65 | 0.779 | 0.65 | Freeze the candidate and seek a locked external confirmation without further scGeneScope tuning. |
| F096 gate contract audit only | 0.60 | 0.85 | 0.95 | 0.55 | 0.90 | 0.90 | 0.85 | 0.800 | 0.55 | Audit why action-heldout CV rejected calibration despite external floor safety. |
| Promote calibrated F095 row now | 0.30 | 0.95 | 0.90 | 0.80 | 0.20 | 0.10 | 0.95 | 0.600 | 0.10 | Reject as post-hoc candidate switching. |
| New architecture immediately | 0.75 | 0.45 | 0.65 | 0.70 | 0.65 | 0.50 | 0.35 | 0.579 | 0.35 | Defer until selector and confirmation issues are resolved. |

## Monitor Decision
`COUNCIL_EXECUTE_METRIC_OR_INTERNAL_AUDIT`

## Exact Next Amendment
Execute `F096_FROZEN_SELECTOR_CONFIRMATION_OR_GATE_CONTRACT_AUDIT`.

## Next Experiment Command Or Implementation Target
Do not promote F095. Either freeze `pubchem_fingerprint + train-only delta calibration` before a genuinely new external confirmation, or run an internal gate-contract audit explaining why action-heldout train CV is misaligned with scGeneScope replicate-heldout validation.
