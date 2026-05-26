# Session Amendment 069: Retrieval-Aware Transition Head

## Trigger
`F039_TRAIN_PROXY_ZERO_FALLBACK`

## Evidence
F038 proved nonzero safe residual capacity exists under oracle held-out scale selection, but F039's deployable train-only proxy selected the zero floor fallback. The next branch should change the residual target geometry during training instead of only changing post-hoc scale selection.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F040_RETRIEVAL_AWARE_TRANSITION_HEAD`

## Implementation Tasks
- Reuse the floor-initialized transition head.
- Add train-only paired-target retrieval margin preservation during residual-head training.
- Do not use condition keys as model inputs, held-out labels, or held-out oracle scales.
- After training, select scale with the F039 train-only safe-scale proxy.
- Score held-out rows only after train-only selection.

## Decision Use
If F040 preserves held-out floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero, pivot to representation/objective redesign. If it violates held-out recall, retire paired-target margin training under this benchmark.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F040 retrieval-aware transition-head training
