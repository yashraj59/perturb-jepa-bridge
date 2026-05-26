# Insight Brief 002: Post-C015/G002/F011/F012

## Current Best Result
No candidate superseded the protected rank-3 train-split-only PLS raw-linear readout. Exp/model promotion remains blocked.

## What Changed Understanding
C014 showed teacher z_bio attenuates source-state structure relative to online/context z_bio. C015 showed this is not enough to authorize source-state preservation training. G002 then showed online-source-neighborhood retrieval remains below the protected recall floor across fresh seeds 1-4. F011 showed a deeper issue: the seed0 recall floor is itself a seed-specific outlier, while transition improvement and delta cosine are stable or stronger on fresh seeds.

## Operational Lessons
Do not continue architecture search against the old seed0 recall gate. It is too likely to reward seed-specific retrieval structure. Do not silently change gates either; the correct move is a new named benchmark registry with multi-seed baselines.

## Cross-Family Findings
Representation repair has useful diagnostic signal but no deployable proxy that clears the floor. Metric/data redesign is now the active family.

## Next Best Experiments
1. Build `synthetic_genetic_anchor_lite_multiseed_v1` Step 0 benchmark registry using seeds 0-4.
2. Define stable primary and secondary gates from that registry without mutating the old benchmark.
3. Only then reopen architecture search, starting with mechanisms that improve continuous transition/delta geometry without hurting rank/magnitude.

## Surprising Finding
The old protected recall floor (`0.4815`) is not reproduced by fresh synthetic seeds (mean `0.2963`, max `0.3333`), while delta cosine is stronger on fresh seeds.
