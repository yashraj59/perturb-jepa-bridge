# Norman Tier 1 Architecture Summary

- Device: `cuda`
- Seed: `11`
- Train fraction: `0.5`
- Steps per trained candidate: `300`
- Family N exact Pearson gate input: `0.5458`
- Single-additive exact Pearson gate input: `0.8981`
- Tier 1 primary pass gate: `>0.9181`
- Runtime seconds: `39.92`

## Exact-Train-Combo Metrics

| baseline                            | pearson_delta_all_mean | pearson_delta_de20_mean | top20_de_overlap_mean | mse_delta_all_mean | direction_accuracy_de20_mean | prediction_delta_variance_ratio | mean_collapse_flag |
| ----------------------------------- | ---------------------- | ----------------------- | --------------------- | ------------------ | ---------------------------- | ------------------------------- | ------------------ |
| A1_feature_bridge_mlp               | 0.6919                 | 0.7657                  | 0.4357                | 0.0021             | 0.9071                       | 0.7740                          | False              |
| A2_feature_bridge_rank2_operator    | 0.6566                 | 0.7306                  | 0.4000                | 0.0024             | 0.8750                       | 0.7442                          | False              |
| B1_pure_additive_architecture       | 0.8981                 | 0.9652                  | 0.5750                | 0.0016             | 1.0000                       | 1.4616                          | False              |
| B2_additive_bounded_interaction_mlp | 0.8922                 | 0.9657                  | 0.5893                | 0.0013             | 1.0000                       | 1.2790                          | False              |
| B3_additive_low_rank_interaction    | 0.8940                 | 0.9665                  | 0.6036                | 0.0015             | 1.0000                       | 1.4435                          | False              |

## Decisions

| experiment_num | family   | status                  | architectural_change                         |
| -------------- | -------- | ----------------------- | -------------------------------------------- |
| 7              | Family A | TIER1_DISCARD_NO_SIGNAL | Feature-conditioned perturbation bridge      |
| 8              | Family A | TIER1_DISCARD_NO_SIGNAL | Feature-conditioned low-rank operator bridge |
| 9              | Family B | TIER1_DISCARD_NO_SIGNAL | Pure additive composition architecture       |
| 10             | Family B | TIER1_DISCARD_NO_SIGNAL | Additive plus bounded MLP interaction        |
| 11             | Family B | TIER1_DISCARD_NO_SIGNAL | Additive plus low-rank interaction           |

Stop condition fired: 5 consecutive Tier 1 discards across families.
