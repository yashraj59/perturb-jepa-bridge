# Session Amendment 022: Cell-Line Source-Dominance Audit

## Trigger
`R001_RETRIEVAL_DOMINATED_BY_CELL_LINE_NOT_PERTURBATION`

## Evidence
R001 showed pseudoheldout retrieval is much stronger for cell-line labels than perturbation or condition labels. The next diagnostic must quantify whether latent endpoints and predicted deltas are dominated by source/cell-line structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending, residual-only transition heads, and metric relabeling are not promotable.

## New Or Reopened Family
Family C: Representation Repair, cell-line/source-dominance audit.

## Exact Next Experiment
`C004_CELL_LINE_SOURCE_DOMINANCE_AUDIT`

## Implementation Tasks
- Use train-only pseudoheldout predictions.
- Measure eta-squared and nearest-centroid predictability for cell line, perturbation, and batch across source, target, true delta, predicted endpoint, and predicted delta.
- Decide whether the next mechanism should remove source/cell-line dominance or redesign retrieval targets.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not fit or select using held-out eval rows.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C004 cell-line/source-dominance representation audit
