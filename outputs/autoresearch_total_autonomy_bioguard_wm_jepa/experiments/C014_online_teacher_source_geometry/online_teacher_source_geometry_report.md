# C014 Online-Teacher Source Geometry Audit

## Decision
`C014_TEACHER_TARGET_ATTENUATES_SOURCE_STATE_STRUCTURE`

## Evidence
- online top-third same-cell-line purity: `0.911229`.
- teacher top-third same-cell-line purity: `0.715682`.
- online-minus-teacher purity gain: `0.195547`.
- online cell-line eta^2: `0.737945`.
- teacher cell-line eta^2: `0.475189`.
- online-teacher row cosine mean: `0.948731`.
- online-minus-teacher residual cell-line eta^2: `0.567401`.

## Interpretation
This audit tests whether the teacher target representation attenuates source-state structure relative to the online/context representation. It is diagnostic only and does not use source metadata as a model input.
