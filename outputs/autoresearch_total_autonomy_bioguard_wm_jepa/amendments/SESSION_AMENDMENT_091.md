# Session Amendment 091: Oracle Residual-Cone Capacity Diagnostic

## Trigger
`F061_PROJECTION_KILLS_DELTA_WITHOUT_STABILIZING_TRANSITION`

## Evidence
F061 closed pure floor-direction projection because it removed the delta-cosine gain and still left negative transition-source cosine gaps. The remaining question is whether an intermediate residual cone between unprojected and projected residuals has any safe nonzero held-out capacity at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F062 is an oracle diagnostic only and cannot promote.

## New Diagnostic Branch
`F062_ORACLE_RESIDUAL_CONE`

## Implementation Tasks
- Use fresh seeds 31, 32, and 33.
- Train the same low-compute descriptor-aligned ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- For each cone mix between unprojected and floor-direction-projected residuals, fit train-only row-wise abstention.
- Use held-out labels only to select the best cone mix diagnostically.
- Report whether safe nonzero residual geometry capacity exists.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F062 oracle residual-cone capacity diagnostic
