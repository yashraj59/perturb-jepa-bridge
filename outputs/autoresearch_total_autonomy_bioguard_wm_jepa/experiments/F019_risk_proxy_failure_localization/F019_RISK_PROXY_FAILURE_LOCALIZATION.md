# F019 Risk Proxy Failure Localization

## Decision
`F019_NO_SIMPLE_SECOND_GATE_REPAIRS_F018_FAILURE`

## F018 Selected Rule Under Audit
- candidate: `pooled_full_ridge`
- alpha: `0.350000`
- feature: `support_gain`
- direction: `ge`
- threshold: `-0.024371`

## Seed-Level Failure
| seed | active fraction | unsafe active fraction | transition gap | delta gap | recall gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.851852 | 0.333333 | 0.005107 | 0.007226 | -0.111111 |
| 1 | 0.592593 | 0.259259 | 0.001690 | 0.005006 | 0.000000 |
| 2 | 0.814815 | 0.111111 | 0.003730 | 0.054832 | 0.037037 |
| 3 | 0.666667 | 0.148148 | 0.003464 | 0.017314 | 0.037037 |
| 4 | 0.703704 | 0.296296 | 0.002829 | 0.015244 | 0.037037 |

## Top Oracle Second Gates
| feature | direction | threshold | active fraction | mean transition gap | min transition gap | mean recall gap | min recall gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| action_support | ge | 0.500000 | 0.725926 | 0.003364 | 0.001690 | 0.000000 | -0.111111 |
| action_support | le | 0.500000 | 0.725926 | 0.003364 | 0.001690 | 0.000000 | -0.111111 |
| support_gain | ge | -0.030949 | 0.725926 | 0.003364 | 0.001690 | 0.000000 | -0.111111 |
| source_support | le | 0.997639 | 0.600000 | 0.003191 | 0.001134 | 0.000000 | -0.074074 |
| blend_minus_floor_norm | ge | 0.039866 | 0.577778 | 0.003173 | 0.001823 | 0.000000 | -0.111111 |
| blend_minus_floor_norm | ge | 0.044840 | 0.503704 | 0.003004 | 0.001736 | -0.014815 | -0.111111 |
| blend_delta_support | ge | 0.790476 | 0.614815 | 0.002980 | 0.001466 | -0.014815 | -0.111111 |
| floor_blend_delta_cosine | ge | 0.964308 | 0.570370 | 0.002802 | 0.001749 | 0.007407 | -0.111111 |
| blend_to_floor_norm_ratio | le | 0.972157 | 0.555556 | 0.002799 | 0.001547 | -0.007407 | -0.111111 |
| floor_delta_support | ge | 0.792214 | 0.562963 | 0.002775 | 0.001466 | -0.014815 | -0.111111 |

## Interpretation
F018 failed because the train-only support-gain proxy activated rows that were unsafe for seed0 retrieval. F019 is an eval-oracle localization diagnostic: if a simple second gate can repair the failure, the next step is a nested train-only two-feature proxy; otherwise the environment-blend family should be cooled.
