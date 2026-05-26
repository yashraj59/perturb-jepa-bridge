# Session Amendment 045: F018 Environment Blend Risk Proxy

## Trigger
`F017_ORACLE_SAFE_ENVIRONMENT_BLEND_CAPACITY_EXISTS`

## Evidence
F017 found large eval-oracle safe blend capacity, but the oracle mask used held-out scoring labels and cannot be used as a candidate. The next step must test whether non-label train-only features can approximate the safe mask.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Family F diagnostic can promote.

## New Or Reopened Family
Family F: Metric And Data Redesign, train-only environment risk-proxy branch.

## Exact Next Experiment
`F018_ENVIRONMENT_BLEND_RISK_PROXY`

## Implementation Tasks
- Build train-only perturbation-group folds.
- Label safe activation only inside train folds by comparing blended and floor predictions.
- Select a simple threshold gate using non-label features only: action support, source support, delta support, and floor-vs-blend geometry.
- Apply the fixed selected rule to held-out eval rows for scoring only.

## Gates
Diagnostic only. A nonzero risk proxy must preserve the per-seed floor on held-out transition, delta cosine, and condition recall before it can justify a later JEPA mechanism.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F018 train-only environment risk proxy
