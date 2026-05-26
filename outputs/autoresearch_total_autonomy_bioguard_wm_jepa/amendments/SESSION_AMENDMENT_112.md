# Session Amendment 112: Tier 3 No-Regression Preflight

## Trigger
`F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`

## Evidence
F082 produced a fresh-seed non-promoting Tier 2-style pass. The autonomy protocol allows Tier 2 to justify Tier 3 design only; it cannot promote the candidate. Before launching a Tier 3 run, all no-regression gates must be made explicit, including the locked delta-rank floor and availability of a paired external validator.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Branch
`F083_TIER3_PREFLIGHT`

## Implementation Tasks
- Read F082 metrics.
- Compare transition, delta cosine, recall, delta rank, and magnitude ratio against locked floors.
- Check cross-modal retrieval gates.
- Check paired external validator availability; Norman RNA-only can be noted but cannot validate imaging.
- Lock candidate code path for any future Tier 3.
- Do not train and do not promote.

## Decision Use
If all gates pass, launch a locked low-compute Tier 3 validation. If rank or validator gates fail, pivot to the smallest targeted repair or validator discovery branch.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F083 Tier 3 no-regression preflight
