# Phase Closure Report 126

## Trigger
`FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION` from cpg0003 Rosetta F097/F098

## Interpretation
The frozen F096/F082 path did not earn fresh external confirmation on cpg0003
Rosetta. F097 used compound-holdout splits and failed. F098 used a
same-condition replicate-holdout split that is closer to the scGeneScope
replicate contract, but it still failed the registered pass rule.

No model is promoted. The protected rank-3 train-split-only PLS raw-linear
readout remains the model of record.

## F098 Summary
| split | transition improvement | delta cosine | recall@1 | floor gap transition | floor gap delta cosine | floor gap recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_test | -0.639843 | 0.241802 | 0.002584 | 0.174012 | 0.142072 | 0.000000 |
| test | -0.678548 | 0.201629 | 0.000813 | 0.137905 | 0.099374 | -0.001626 |
| validation | -0.611366 | 0.261971 | 0.003500 | 0.232051 | 0.186954 | 0.000875 |

## Diagnosis
F098 improved over the full-ridge audit floor on transition and delta cosine,
but absolute transition improvement was negative on all splits and test recall
fell below the floor. This is a scientific failure, not a runtime failure.

## Next Step
Run an artifact-only F099 diagnostic before any new model changes. The diagnostic
should inspect Rosetta source/control geometry, source-vs-target cosines, full
ridge behavior, and whether L1000/Cell Painting profiles are already centered in
a way that makes `source_as_target` or transition-improvement semantics differ
from scGeneScope.
