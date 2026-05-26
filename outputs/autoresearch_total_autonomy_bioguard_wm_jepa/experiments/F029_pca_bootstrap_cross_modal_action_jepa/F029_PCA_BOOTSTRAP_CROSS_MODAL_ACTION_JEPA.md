# F029 PCA-Bootstrap Cross-Modal/Action JEPA

## Decision
`F029_REPRESENTATION_FLOOR_GOOD_DIRECT_JEPA_UNDER_FLOOR`

## Purpose
F028 proved that a neural RNA encoder can preserve the repaired program-action floor when bootstrapped from train-only PCA. F029 is the first small real JEPA probe on the repaired benchmark: it uses online RNA/image encoders, EMA target encoders, stop-gradient latent targets, query-conditioned predictors, RNA->image and image->RNA latent losses, and control RNA + non-exact program action -> perturbed RNA transition loss.

## Active Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- condition recall@1 >= `0.255644`

## Mean Metrics
- direct JEPA transition improvement: `0.066379`
- direct JEPA delta cosine: `0.232022`
- direct JEPA recall@1: `0.716049`
- representation ridge floor transition: `0.090390`
- RNA->image recall@1: `0.740741`
- image->RNA recall@1: `0.827160`
- action-negative transition gap: `0.025565`

## Seed Rows
```tsv
seed	direct_transition_improvement	direct_delta_cosine	direct_recall_at_1	direct_delta_rank	direct_magnitude_ratio	ridge_floor_transition_improvement	ridge_floor_delta_cosine	ridge_floor_recall_at_1	ridge_floor_delta_rank	rna_to_image_recall_at_1	image_to_rna_recall_at_1	action_negative_transition_gap	target_effective_rank	image_target_effective_rank	identity_violation	leakage_flag
0	0.06782742380317341	0.2641602930641669	0.8148148148148148	6.491847495608478	1.2832085209787845	0.0964963193118168	0.8739639175133193	0.7777777777777778	5.520607315771544	0.7777777777777778	0.8888888888888888	0.019971702666985704	4.70653769072506	5.792151142083556	0.0	0.0
1	0.0793712546495117	0.2387544414789651	0.5925925925925926	8.04046342021349	1.231406344982869	0.09592202339664185	0.7865866830967958	0.48148148148148145	7.24954734827331	0.7037037037037037	0.7777777777777778	0.04640115915614602	5.072083648051685	7.29032668002509	0.0	0.0
2	0.05193819095378255	0.1931524136791129	0.7407407407407407	8.446496459288806	0.8637088021116889	0.07875272394884514	0.8043211603593633	1.0	6.771451934265017	0.7407407407407407	0.8148148148148148	0.010320671122301207	5.864110599831982	7.193856859600664	0.0	0.0

```

## Train Trace
```tsv
seed	loss	transition_loss	rna_to_image_loss	image_to_rna_loss	pca_anchor_loss	action_negative_loss	variance_loss	mean_positive_transition_cosine	mean_negative_transition_cosine
0	0.07337111234664917	0.01282503642141819	0.003942805342376232	0.007897979579865932	0.2348262518644333	0.0172317735850811	0.0003372585924807936	0.9896605014801025	0.8571949005126953
1	0.08962474763393402	0.01866811141371727	0.006927721202373505	0.012655777856707573	0.24298037588596344	0.027770616114139557	0.0	0.9843196272850037	0.9078200459480286
2	0.1079815998673439	0.022773582488298416	0.005072675179690123	0.009262376464903355	0.3491092622280121	0.010511077009141445	0.0	0.981257438659668	0.869148850440979

```

## Identity And Leakage
- online/context encoders present: `True`
- EMA target encoders present: `True`
- stop-gradient latent targets: `True`
- query-conditioned predictors: `True`
- cross-modal RNA/image JEPA losses: `True`
- action-conditioned transition JEPA loss: `True`
- action input: non-exact program descriptor only
- PCA anchor: auxiliary bootstrap, not the only objective
- eval rows: scoring only
- no model promotion from this Tier 1 probe
