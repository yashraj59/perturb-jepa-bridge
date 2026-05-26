# Norman v1 Final Report

Status: SEARCH_CLOSED_NO_NEW_BASELINE.

## Stop Condition

Five consecutive Tier 1 discards fired across Family A and Family B. The autonomous architecture loop is stopped pending explicit user instruction.

## Best Tier 1 Candidate

- Candidate: `B1_pure_additive_architecture`
- Exact-train-combo Pearson delta all genes: `0.8981`
- Tier 1 pass gate: `>0.9181`
- Exact top20 DE overlap: `0.5750`
- Exact MSE delta all genes: `0.0016`

## Model Of Record

- Published GEARS remains the active model of record.
- Family N condition-mean table remains the carried reference.
- The Step 0 single-additive baseline remains the strongest recomputed exact-combo comparator.
- No Tier 3 pass occurred.

## Result Rows

| experiment_num | family   | status                  | architectural_change                         |
| -------------- | -------- | ----------------------- | -------------------------------------------- |
| 7              | Family A | TIER1_DISCARD_NO_SIGNAL | Feature-conditioned perturbation bridge      |
| 8              | Family A | TIER1_DISCARD_NO_SIGNAL | Feature-conditioned low-rank operator bridge |
| 9              | Family B | TIER1_DISCARD_NO_SIGNAL | Pure additive composition architecture       |
| 10             | Family B | TIER1_DISCARD_NO_SIGNAL | Additive plus bounded MLP interaction        |
| 11             | Family B | TIER1_DISCARD_NO_SIGNAL | Additive plus low-rank interaction           |
