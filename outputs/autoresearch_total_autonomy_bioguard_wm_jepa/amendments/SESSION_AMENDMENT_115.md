# Session Amendment 115: Current-Registry Tier 3 Design And Validator Preflight

## Trigger
`F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE`

## Evidence
F085 showed that the F082 delta-calibrated JEPA wrapper preserves the current image-teacher latent floor under a representation-specific registry, while the old locked rank floor is not comparable to that target geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F086_CURRENT_REGISTRY_TIER3_DESIGN`

## Implementation Tasks
- Read F082 and F085 metrics.
- Lock corrected current-registry gates for transition, delta cosine, recall, and rank.
- Carry forward F082 cross-modal RNA->image and image->RNA retrieval gates.
- Check local paired scRNA plus imaging validator availability.
- Record external validator candidates without downloading large datasets.
- Do not train and do not promote.

## Decision Use
If F086 finds all synthetic gates pass and a paired validator is locally ready, launch a locked low-compute Tier 3 run. If only an external paired validator candidate exists, write an adapter/ingest preflight next. If no paired validator exists, pivot to validator discovery and keep Norman RNA-only as non-promoting diagnostics.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F086 current-registry Tier 3 design
