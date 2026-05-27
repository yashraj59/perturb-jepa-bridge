# Phase Closure Report 124

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from `F095_NON_EXACT_PUBCHEM_FINGERPRINT_DESCRIPTOR_RERUN`

## Interpretation
F095 ran on CUDA and completed as a non-promoting scientific near pass. PubChem fingerprint descriptors fixed descriptor coverage and greatly improved the floor comparison. The official candidate for F095 was still the split-safe gate, which selected raw JEPA output. Raw JEPA cleared transition and delta-cosine floors on all splits but missed `alternate_test` recall by `-0.006061`.

The calibrated fingerprint row would clear transition, delta-cosine, and recall floor gaps on all splits, but it is not promotable from this run because the predeclared F095 gate abstained and the calibrated row was not the selected candidate.

## F095 Official Candidate Summary
| split | transition improvement | delta cosine | recall@1 | floor gap transition | floor gap delta cosine | floor gap recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_test | 0.494546 | 0.203499 | 0.121212 | 0.000577 | 0.019918 | -0.006061 |
| test | 0.650817 | 0.351871 | 0.428571 | 0.005121 | 0.022421 | 0.000000 |
| validation | 0.415889 | 0.298418 | 0.382716 | 0.016640 | 0.023397 | 0.012346 |

## Repair Direction
Freeze the lesson: non-exact fingerprints are useful, but post-hoc candidate switching on scGeneScope is not a real Tier 3 promotion. The next step is to freeze a selector before another external confirmation or run a selector audit that does not use external metrics for candidate choice.
