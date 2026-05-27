# Phase Closure Report 125

## Trigger
`PASS_EXTERNAL_TIER3_NON_PROMOTING` from `F096_FROZEN_FINGERPRINT_CALIBRATED_CANDIDATE`

## Interpretation
The immediate F082 external-validation failure is fixed as a non-promoting scGeneScope result. The frozen candidate uses ProgramBootstrapJEPA with PubChem scalar plus 2D fingerprint action descriptors and train-only delta calibration. It clears transition, delta-cosine, and recall floor gaps on validation, test, and alternate_test with clean identity and leakage flags.

No model is promoted because this pass occurred after scGeneScope-guided repair. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record until a fresh external Tier 3 confirmation passes.

## F096 Summary
| split | transition improvement | delta cosine | recall@1 | RNA->image recall@1 | image->RNA recall@1 | floor gap transition | floor gap delta cosine | floor gap recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_test | 0.494588 | 0.185750 | 0.127273 | 0.036364 | 0.036364 | 0.000619 | 0.002169 | 0.000000 |
| test | 0.646406 | 0.331194 | 0.428571 | 0.178571 | 0.119048 | 0.000709 | 0.001743 | 0.000000 |
| validation | 0.400167 | 0.277198 | 0.370370 | 0.123457 | 0.111111 | 0.000918 | 0.002177 | 0.000000 |

## Fixed Path
- `scripts/run_f082_scgenescope_external_validation.py`
- `--descriptor-mode pubchem_fingerprint`
- `--gate-mode calibrated`
- `--device cuda`
