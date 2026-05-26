# Session Amendment 107: Source-Program Failure Localization

## Trigger
`F077_SOURCE_PROGRAM_ACTION_REPAIRS_STRUCTURE_BUT_WEAK`

## Evidence
F077 was the first branch to repair heldout recall and improve delta cosine relative to F076, but it still had very low transition improvement, low effective rank, and almost no perturbation-probe signal. Before adding capacity or descriptors, the loop must localize the remaining failure mode.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F078 is diagnostic and cannot promote.

## New Diagnostic Branch
`F078_SOURCE_PROGRAM_FAILURE_LOCALIZATION`

## Implementation Tasks
- Read F077 seed metrics and training traces.
- Compare candidate versus image teacher and observed RNA by policy and seed.
- Quantify recall gap, delta cosine gap, transition gap, effective-rank gap, perturbation-probe gap, batch-probe gap, and image-target cosine.
- Do not train, do not select a model, and do not promote.

## Decision Use
If recall is high but transition/action signal is low, pivot to delta/source-improvement objectives or richer non-exact action descriptors. If low-rank prototype collapse dominates, add rank/adversarial diversity constraints before a full JEPA wrapper.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F078 source-program failure localization
