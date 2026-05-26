# C016 Online Latent Transition Audit

## Decision
`C016_ONLINE_LATENT_REPRESENTATION_HAS_REPAIR_SIGNAL`

## Active Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- condition recall@1 >= `0.255644`

## Summary
| space | seeds | mean transition | mean delta cosine | mean recall@1 | mean delta rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| online_action_ridge_floor | 5 | 0.007174 | 0.420718 | 0.362963 | 9.817766 |
| teacher_action_ridge_floor | 5 | 0.011281 | 0.451515 | 0.333333 | 10.245809 |

## Interpretation
This audit tests whether the online/source-state-preserving latent space carries a better action-conditioned transition signal than the teacher latent space under the active multiseed gates. It is diagnostic only and cannot promote a model.
