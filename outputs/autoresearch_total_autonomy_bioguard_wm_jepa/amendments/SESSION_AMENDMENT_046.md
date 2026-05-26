# Session Amendment 046: F019 Risk-Proxy Failure Localization

## Trigger
`F018_TRAIN_ONLY_RISK_PROXY_DISCARDED_HELDOUT_BELOW_FLOOR`

## Evidence
F018 selected a nonzero train-only support-gain proxy, but held-out seed0 lost condition recall. Under the floor-preservation contract, the candidate is discarded even though mean transition and mean delta improved.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Or Reopened Family
Family F: Metric And Data Redesign, risk-proxy failure-localization branch.

## Exact Next Experiment
`F019_RISK_PROXY_FAILURE_LOCALIZATION`

## Implementation Tasks
- Reconstruct the F018 selected rule.
- Identify safe vs unsafe selected rows per seed.
- Compare non-label feature distributions for safe and unsafe selected rows.
- Run an eval-oracle second-gate capacity audit to determine whether a simple additional feature threshold could repair seed0 without using it as a promoted rule.

## Gates
Diagnostic only. If a simple second gate has oracle capacity, the next step must be nested train-only calibration. If not, cool environment blending and pivot.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F019 risk-proxy failure localization
