# Session Amendment 090: Projection Collapse Audit

## Trigger
`F060_FLOOR_DIRECTION_PROJECTION_SAFE_BUT_WEAK`

## Evidence
F060 preserved retrieval but was safe-but-weak. Positive floor-direction projection removed the delta-cosine gain and still produced negative transition-source cosine gaps on two of three seeds. This suggests pure projection is too restrictive and does not solve the F058 instability.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F061 is read-only diagnostics and cannot promote.

## New Diagnostic Branch
`F061_PROJECTION_COLLAPSE_AUDIT`

## Implementation Tasks
- Read F058, F059, and F060 artifacts.
- Compare unprojected row-wise residual signal with projected row-wise residual signal.
- Decide whether pure floor-direction projection should be cooled.
- Recommend the next branch without using eval targets for fitting or selection.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F061 projection collapse audit
