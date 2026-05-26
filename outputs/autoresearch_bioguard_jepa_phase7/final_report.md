# BioGuard-JEPA Phase 7 Final Report

## Decision label
PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA

## Model of record status
Protected rank-3 train-split-only PLS raw-linear readout remains model of record.

## Experiments run
- BSG000 reproduction: completed before residual selection.
- BSG001 residual target/split audit: completed.
- BSG002-BSG004 residual candidates: evaluated through train-only cross-fitted selection.
- BSG005-BSG008: not run because no residual passed selection.

## Exact floor values used
- transition improvement: `0.0057`
- delta cosine: `0.3980`
- recall@1: `0.4815`
- delta rank: `10.2835`
- magnitude ratio: `0.7744`

## Candidate vs Floor Table
| experiment | mode | transition improvement | delta cosine | recall@1 | delta rank | floor gap transition | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BSG000 | reproduction | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.0000 | BSG000_PASS_REPRODUCED_PHASE6 |
| BSG001 | residual target/split audit | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.0000 | BSG001_PASS_RESIDUAL_TARGET_VALID |
| BSG002 | spectral residual CV selection | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.0000 | BSG002_DISCARD_SPECTRAL_RESIDUAL_FAILS_CV_GATE |
| BSG003 | kernel residual CV selection | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.0000 | BSG003_DISCARD_KERNEL_RESIDUAL_FAILS_CV_GATE |
| BSG004 | program residual CV selection | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.0000 | BSG004_DISCARD_PROGRAM_RESIDUAL_FAILS_CV_GATE |

BSG002-BSG004 report the deployed guarded output, which is the zero-residual floor fallback because no residual passed the cross-fitted gate.

## Train-Internal Cross-Fit Selection Table
| candidate_id | selected | cv_lcb_transition_gap | cv_lcb_recall_gap | mean_transition_gap | mean_recall_gap | residual_scale | action_negative_gap |
| ------------ | -------- | --------------------- | ----------------- | ------------------- | --------------- | -------------- | ------------------- |
| spectral     | False    | -0.000207             | -0.041111         | -0.000150           | -0.013889       | 0.000000       | -0.000150           |
| kernel       | False    | 0.000095              | -0.044454         | 0.000400            | 0.000000        | 0.000000       | 0.000400            |
| program      | False    | -0.000000             | 0.000000          | 0.000000            | 0.000000        | 0.000000       | 0.000000            |

## Calibration/Gating
Every evaluated residual defaulted to zero residual / floor fallback because the conservative train-internal gate did not pass.

## Leakage audit summary
PASS. Eval/test target rows were not used for fitting, whitening/statistics, residual calibration, residual selection, or candidate choice.

## JEPA identity status
No full BioGuard-JEPA candidate was trained. Operator-only probes cannot promote the model.

## Norman status
Norman was not run because synthetic BioGuard residual selection did not pass.

## Recommendation
Close Phase 7 under current data. The full-ridge floor is the safest transition operator until a residual has positive train-internal lower-confidence evidence.

## Notes
- No residual candidate passed train-internal cross-fitted selection. BSG005-BSG008 were not run.
