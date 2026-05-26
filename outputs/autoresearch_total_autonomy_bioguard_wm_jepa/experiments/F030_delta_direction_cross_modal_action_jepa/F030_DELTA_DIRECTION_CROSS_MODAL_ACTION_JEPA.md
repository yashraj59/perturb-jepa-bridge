# F030 Delta-Direction Cross-Modal/Action JEPA

## Decision
`F030_DELTA_DIRECTION_CROSS_MODAL_ACTION_JEPA_TIER1_PASS`

## Purpose
F029 trained a real cross-modal/action JEPA but missed the delta-cosine gate. F030 keeps the same JEPA identity and adds only targeted transition-direction pressure: a stop-gradient teacher-delta cosine loss and a source-improvement hinge.

## Active Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- condition recall@1 >= `0.255644`

## Mean Metrics
- direct JEPA transition improvement: `0.244565`
- direct JEPA delta cosine: `0.875045`
- direct JEPA recall@1: `0.839506`
- delta-cosine gain vs F029: `0.643023`
- transition gain vs F029: `0.178186`
- representation ridge floor transition: `0.258531`
- RNA->image recall@1: `0.728395`
- image->RNA recall@1: `0.814815`
- action-negative transition gap: `0.026288`

## Seed Rows
```tsv
seed	direct_transition_improvement	direct_delta_cosine	direct_recall_at_1	direct_delta_rank	direct_magnitude_ratio	ridge_floor_transition_improvement	ridge_floor_delta_cosine	ridge_floor_recall_at_1	ridge_floor_delta_rank	rna_to_image_recall_at_1	image_to_rna_recall_at_1	action_negative_transition_gap	target_effective_rank	image_target_effective_rank	identity_violation	leakage_flag
0	0.2339459239544114	0.8728201551106585	0.9259259259259259	9.848354092935379	0.7470709619822151	0.25248784653105394	0.9446918874852716	0.8888888888888888	5.991014252378266	0.7777777777777778	0.8888888888888888	0.019098889298998167	5.431093099382405	6.332271290158552	0.0	0.0
1	0.29772026237201926	0.8848916375753341	0.6666666666666666	9.606944044700622	0.8385655580630453	0.2971144006335742	0.8665828633046387	0.7407407407407407	6.490953344496809	0.6666666666666666	0.6666666666666666	0.05050511884816958	5.7914686381140665	7.695449192243398	0.0	0.0
2	0.20202796321530925	0.8674240524266841	0.9259259259259259	10.678530273393276	0.733892778070482	0.22599143284394246	0.939539965801192	0.9259259259259259	6.763852122613254	0.7407407407407407	0.8888888888888888	0.009259062564726334	5.802090897044281	7.450356362663291	0.0	0.0

```

## Train Trace
```tsv
seed	loss	transition_loss	delta_direction_loss	source_improvement_hinge	rna_to_image_loss	image_to_rna_loss	pca_anchor_loss	action_negative_loss	variance_loss	mean_positive_transition_cosine	mean_negative_transition_cosine
0	0.12838214635849	0.03142554312944412	0.04181993007659912	0.0	0.0071952324360609055	0.01012422889471054	0.3156030476093292	0.020891744643449783	0.0007370617240667343	0.9746823310852051	0.8415267467498779
1	0.16421817243099213	0.05136023834347725	0.05618488788604736	0.0006736000068485737	0.008147129788994789	0.01364117581397295	0.31673169136047363	0.024656957015395164	0.00047400096082128584	0.9604076743125916	0.848053514957428
2	0.20435382425785065	0.05768632888793945	0.07269322872161865	0.00020641705486923456	0.010339522734284401	0.013924619182944298	0.45706692337989807	0.011010449379682541	0.0	0.9517887830734253	0.8027885556221008

```

## Identity And Leakage
- online/context encoders present: `True`
- EMA target encoders present: `True`
- stop-gradient latent targets: `True`
- query-conditioned predictors: `True`
- cross-modal RNA/image JEPA losses: `True`
- action-conditioned transition JEPA loss: `True`
- action input: non-exact program descriptor only
- targeted repair: delta-direction loss plus source-improvement hinge
- eval rows: scoring only
- no model promotion from this Tier 1 probe
