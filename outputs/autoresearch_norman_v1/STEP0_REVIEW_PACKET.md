# Norman v1 Step 0 Review Packet

Status: paused at the mandatory Step 0 review gate. This is not a Tier 3 promotion or architecture-search closure.

## What Ran

- Built Norman 2019 condition-level loader and GEARS-compatible simulation split.
- Added split leakage test for perturbation-order aliases.
- Ran Step 0 recomputed baselines on CPU: global train mean, single additive, Family N condition-mean table, and closed-form PLS readout.
- Registered GEARS and CPA published values without modification.
- Did not start Family A or Family B.

## Best Recomputed Exact-Train-Combo Baseline

- Baseline: `single_perturbation_additive`
- Pearson delta all genes: `0.8981`
- Pearson delta DE20: `0.9652`
- Top-20 DE overlap: `0.5750`
- MSE delta all genes: `0.0016`

## Active Model Of Record

- Published GEARS remains active.
- Family N recomputed condition-mean table remains the carried train-only reference.
- No Tier 3 pass occurred.

## Required Next Action

Review Step 0 artifacts before authorizing Family A or Family B.

Runtime seconds: `23.45`.
