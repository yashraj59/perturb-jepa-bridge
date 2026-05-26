# BioOperator-JEPA Phase 5 Final Report

## Decision label

PHASE5_OPERATOR_FLOOR_REPRODUCED_BUT_NEURAL_FAILURE_CLOSE

## Model of record

Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass explicitly supersedes it.

## What was tested

- Stage A operator floor reproduction: passed (`PHASE5_OPERATOR_FLOOR_REPRODUCED`)
- Stage B sign/gradient/loss contracts: passed (`5 passed`)
- BOJ001 frozen neural ridge-equivalence: passed and matched the action-ridge floor
- BOJ002 frozen low-rank control-affine operator: failed below the ridge floor
- BOJ003 frozen-backbone integration: not run because BOJ002 fired stop condition
- BOJ004 end-to-end JEPA: not run
- BOJ005 Norman RNA-only diagnostic: not run

## Key metrics

| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |
|---|---:|---:|---:|---:|---:|---|
| BOJ001 | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | TIER1_KEEP_OPERATOR_MATCHES_FLOOR |
| BOJ002 | 0.0025 | 0.3603 | 0.3704 | 7.5812 | 0.7579 | TIER1_DISCARD_OPERATOR_BELOW_FLOOR |

## Floor comparison

- eval action_ridge_delta improvement = `+0.0057`
- eval action_ridge_delta delta_cosine = `0.3980`
- eval action_ridge_delta rank = `10.2835`
- BOJ001 floor gap = `-0.0000`
- BOJ002 floor gap = `-0.0032`

## What failed or passed

- Latent cache: passed; Phase 4 cached teacher latents loaded successfully.
- Metric reproduction: passed; BOJ000 reproduced action-ridge and low-rank ridge floors.
- Sign convention: passed contract test.
- Loss gradient: passed raw and whitened one-step gradient contracts.
- Ridge neuralization: passed; BOJ001 exactly reproduced the action-ridge floor with a neural linear head.
- Control-affine operator: failed; BOJ002 fell below action-ridge improvement, recall, and rank gates.
- Full JEPA wrapper: not tested because BOJ002 stopped the sequence.
- End-to-end encoder training: not tested.

## Recommendation

Retain BOJ001 only as a frozen operator-floor audit reference. Do not promote any Phase 5 Tier 1 result. The next technical work should debug why the structured low-rank control-affine constraint loses held-out rank/recall relative to the full ridge floor before attempting a full JEPA wrapper.
