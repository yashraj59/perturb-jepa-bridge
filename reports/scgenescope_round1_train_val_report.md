# scGeneScope Round1-Train-Val Result

Run date: 2026-05-18 UTC

This is the retained result for the clean branch. It uses the scGeneScope
precomputed scVI RNA embeddings and ViT-L Cell Painting embeddings, bridged at
condition level with `condition_key_scgenescope = perturbation`.

## Split

Prepared with:

```bash
uv run python scripts/prepare_scgenescope_pairing.py --split-policy round1_train_val
```

Split policy:

| split | source |
| --- | --- |
| train | round 1 replicates 3 and 5 |
| we_test | round 1 replicate 4 |
| he_test | round 2 replicates 1 and 2 |

## Design Audit

| split | conditions | batches | condition-batches min | condition-batches mean | single-batch condition fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 29 | 4 | 2 | 2.068966 | 0.000000 |
| we_test | 29 | 2 | 1 | 1.034483 | 0.965517 |
| he_test | 29 | 8 | 2 | 2.206897 | 0.000000 |

This is the cleanest scGeneScope split tested so far because both train and
held-out-experiment (`he_test`) have repeated technical batches per condition.

## Training

Fresh pretraining was run on train-only data. The final bridge log line was:

| step | align | image_batch_adv | rna_batch_adv | image_perturbation_cls | rna_perturbation_cls | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2950 | 0.6101 | 1.0379 | 0.9765 | 0.1820 | 0.6655 | 3.8004 |

The bridge learned alignment, but the final checkpoint remained batch-decodable,
especially for image embeddings.

## Retrieval

| split | mAP | R@1 | R@5 | R@10 | median rank | batch-only R@5 | learned / batch R@5 | same-perturbation enrich@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WE | 0.421082 | 0.275862 | 0.551724 | 0.862069 | 4.0 | 0.344828 | 1.600000 | 2.5 |
| HE | 0.218707 | 0.103448 | 0.275862 | 0.448276 | 12.0 | 0.413793 | 0.666667 | 1.3 |

Centroid controls:

| split | centroid mAP | centroid R@5 | label-shuffle mAP | label-shuffle R@5 | mAP margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| WE | 0.421082 | 0.551724 | 0.170242 | 0.206897 | 0.250840 |
| HE | 0.218707 | 0.275862 | 0.158163 | 0.206897 | 0.060544 |

## Batch Probes

| split | RNA probe resub | RNA probe balanced | image probe resub | image probe balanced |
| --- | ---: | ---: | ---: | ---: |
| WE | 0.758621 | 0.685714 | 0.931034 | 0.688095 |
| HE | 0.827586 | n/a | 0.896552 | n/a |

## Interpretation

The fix path improved split design and produced a strong within-round result:

- WE passes the `1.5 * batch_only` R@5 threshold:
  `0.551724 > 1.5 * 0.344828`.
- WE clearly beats label shuffle.

HE still fails:

- learned HE R@5 is `0.275862`
- batch-only HE R@5 is `0.413793`
- image batch probe remains high at `0.896552`

The model is not broken as an alignment machine; it can fit within-round
cross-modal retrieval when train and eval are close. The failure is round/domain
transfer: round 2 shifts the image/RNA embedding geometry enough that the
learned bridge does not carry biology across it.

The data is useful for debugging and within-domain retrieval, but the effective
biological sample size is 29 condition bags, not the full profile count. More
independent perturbations or explicit domain-disentanglement are needed before
claiming robust held-out-domain biological transfer.
