# C015 Source-State Preservation Decision Audit

## Decision
`C015_DO_NOT_REOPEN_TRAINING_SOURCE_STATE_PRESERVATION_INSUFFICIENT`

## Evidence
- Protected condition recall floor: `0.481481`.
- Current exact source-state nested recall: `0.480159`.
- Fresh exact source-state nested recall: `0.447090`.
- Current teacher-source neighborhood recall: `0.398148`.
- Current online-source neighborhood recall: `0.457672`.
- Fresh online-source neighborhood recall: `0.418651`.
- Online-minus-teacher source-state purity gain: `0.195547`.
- Online-minus-teacher residual cell-line eta^2: `0.567401`.

## Interpretation
Teacher target attenuation is real, but nested online-source-neighborhood retrieval remains below the protected floor and does not generalize to the fresh synthetic seed. This audit decides whether to reopen training now or continue with lower-compute diagnostics.
