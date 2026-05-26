# Session Amendment 070: Constraint Ablation Diagnostic

## Trigger
`F040_RETRIEVAL_AWARE_ZERO_FALLBACK`

## Evidence
F038 found safe nonzero held-out residual scales, but F039 and F040 both selected zero under train-only residual calibration. The next step is not more training; it is to identify which train-only constraint blocks the oracle-safe scales.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F041_CONSTRAINT_ABLATION`

## Implementation Tasks
- Join the F039 train-only proxy scale grid with the F038 held-out oracle scale grid by seed and scale.
- Evaluate constraint ablations: full F039 proxy, drop q10, drop near-tie erosion, drop all margin constraints, continuous-only, and small-scale caps.
- Use held-out oracle outcomes only for diagnostic attribution.
- Do not train a new model, promote a candidate, or use held-out oracle outcomes as deployable calibration labels.

## Decision Use
If a simple train-only ablation recovers safe nonzero residuals, design a future validation branch with the rule pre-registered on fresh splits/seeds. If no ablation recovers safe nonzero residuals, pivot away from post-hoc residual scaling.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F041 constraint-ablation diagnostic
