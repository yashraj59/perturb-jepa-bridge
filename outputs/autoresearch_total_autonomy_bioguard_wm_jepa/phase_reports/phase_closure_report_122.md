# Phase Closure Report 122

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION`

## Interpretation
F082 external validation on scGeneScope ended as a candidate failure, not a project stop. The external validator path is usable: backed obs contracts, split mapping, RNA-image pairing, and identity/leakage checks passed. No model is promoted, and the protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

## External Validation Summary
| split | transition improvement | delta cosine | recall@1 | RNA->image recall@1 | image->RNA recall@1 | floor gap transition | floor gap delta cosine | floor gap recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_test | 0.507639 | 0.216546 | 0.127273 | 0.048485 | 0.030303 | -0.024421 | -0.092407 | 0.036364 |
| test | 0.648238 | 0.353785 | 0.404762 | 0.107143 | 0.095238 | 0.035921 | -0.042797 | 0.190476 |
| validation | 0.377733 | 0.279499 | 0.283951 | 0.111111 | 0.086420 | 0.056966 | -0.016252 | 0.135802 |

## Audit Closure
`F093_CALIBRATION_AND_DESCRIPTOR_REPAIR_REQUIRED` completed as an obs-free, artifact-only audit. It loaded only saved TSV/JSON validation artifacts and did not open raw H5AD matrices, fit a model, refit PCA, refit calibration, use held-out target statistics, or promote.

## Repair Direction
The next admissible repair is a JEPA-only split-safe calibration abstention/blend gate. The protected PLS/full-ridge floor remains an audit threshold only, not a candidate representation path or fallback output.
