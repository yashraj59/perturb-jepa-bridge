# Architectural Changes Log

## Change 001: variance_0p001

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6352.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 002: variance_0p002

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.002`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6379.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 003: variance_0p005

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6420.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 004: variance_0p01

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6485.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 005: variance_0p02

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6610.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 006: variance_0p05

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7034.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 007: variance_0p1

### Family
A

### Exact mechanism
shared_variance_weight: `--shared-variance-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7742.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 008: covariance_0p001

### Family
G

### Exact mechanism
shared_covariance_weight: `--shared-covariance-weight 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5999.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 009: covariance_0p005

### Family
G

### Exact mechanism
shared_covariance_weight: `--shared-covariance-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0000; collapse True; RNA latent R2 -0.6190.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 010: covariance_0p01

### Family
G

### Exact mechanism
shared_covariance_weight: `--shared-covariance-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7126.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 011: covariance_0p02

### Family
G

### Exact mechanism
shared_covariance_weight: `--shared-covariance-weight 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8195.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 012: covariance_0p05

### Family
G

### Exact mechanism
shared_covariance_weight: `--shared-covariance-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8871.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 013: crosscorr_0p001

### Family
G

### Exact mechanism
cross_correlation_weight: `--cross-correlation-weight 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6347.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 014: crosscorr_0p005

### Family
G

### Exact mechanism
cross_correlation_weight: `--cross-correlation-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6514.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 015: crosscorr_0p01

### Family
G

### Exact mechanism
cross_correlation_weight: `--cross-correlation-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7052.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 016: crosscorr_0p02

### Family
G

### Exact mechanism
cross_correlation_weight: `--cross-correlation-weight 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8087.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 017: crosscorr_0p05

### Family
G

### Exact mechanism
cross_correlation_weight: `--cross-correlation-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0312; collapse True; RNA latent R2 -1.0302.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 018: var_0p001_cov_0p001

### Family
G

### Exact mechanism
variance_covariance_combo: `--shared-variance-weight 0.001 --shared-covariance-weight 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6012.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 019: var_0p005_cov_0p005

### Family
G

### Exact mechanism
variance_covariance_combo: `--shared-variance-weight 0.005 --shared-covariance-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0000; collapse True; RNA latent R2 -0.6247.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 020: var_0p01_cov_0p005

### Family
G

### Exact mechanism
variance_covariance_combo: `--shared-variance-weight 0.01 --shared-covariance-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0000; collapse True; RNA latent R2 -0.6314.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 021: var_0p005_cov_0p01

### Family
G

### Exact mechanism
variance_covariance_combo: `--shared-variance-weight 0.005 --shared-covariance-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7167.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 022: var_0p02_cov_0p01

### Family
G

### Exact mechanism
variance_covariance_combo: `--shared-variance-weight 0.02 --shared-covariance-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7273.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 023: ema_0p9

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.9`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6327.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 024: ema_0p95

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.95`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6331.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 025: ema_0p98

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.98`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6333.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 026: ema_0p99

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.99`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6334.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 027: ema_0p992

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.992`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6334.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 028: ema_0p999

### Family
A

### Exact mechanism
ema_decay: `--ema-decay 0.999`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 029: mean_p1

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator mean --num-bag-prototypes 1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse False; RNA latent R2 -1.0266.

### Verdict
TIER1_DISCARD_BATCH_LEAKAGE.

## Change 030: mean_p2

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator mean --num-bag-prototypes 2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse False; RNA latent R2 -1.0266.

### Verdict
TIER1_DISCARD_BATCH_LEAKAGE.

## Change 031: mean_p4

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator mean --num-bag-prototypes 4`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse False; RNA latent R2 -1.0266.

### Verdict
TIER1_DISCARD_BATCH_LEAKAGE.

## Change 032: attention_p1

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator attention --num-bag-prototypes 1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.4051.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 033: attention_p3

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator attention --num-bag-prototypes 3`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6567.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 034: attention_p4

### Family
B

### Exact mechanism
bag_aggregator_capacity: `--bag-aggregator attention --num-bag-prototypes 4`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8196.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 035: dropout_0p0

### Family
H

### Exact mechanism
dropout: `--dropout 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8166.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 036: dropout_0p02

### Family
H

### Exact mechanism
dropout: `--dropout 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5111.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 037: dropout_0p1

### Family
H

### Exact mechanism
dropout: `--dropout 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6786.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 038: dropout_0p2

### Family
H

### Exact mechanism
dropout: `--dropout 0.2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.9073.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 039: lr_0p0003

### Family
H

### Exact mechanism
lr: `--lr 0.0003`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6635.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 040: lr_0p0005

### Family
H

### Exact mechanism
lr: `--lr 0.0005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6988.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 041: lr_0p002

### Family
H

### Exact mechanism
lr: `--lr 0.002`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5677.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 042: lr_0p003

### Family
H

### Exact mechanism
lr: `--lr 0.003`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5341.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 043: wd_0p0

### Family
H

### Exact mechanism
weight_decay: `--weight-decay 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6336.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 044: wd_0p001

### Family
H

### Exact mechanism
weight_decay: `--weight-decay 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6336.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 045: wd_0p05

### Family
H

### Exact mechanism
weight_decay: `--weight-decay 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6329.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 046: wd_0p1

### Family
H

### Exact mechanism
weight_decay: `--weight-decay 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6323.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 047: align_0p0

### Family
C

### Exact mechanism
align_weight: `--align-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -1.0465.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 048: align_0p25

### Family
C

### Exact mechanism
align_weight: `--align-weight 0.25`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0000; collapse True; RNA latent R2 -0.9517.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 049: align_0p5

### Family
C

### Exact mechanism
align_weight: `--align-weight 0.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8268.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 050: align_1p5

### Family
C

### Exact mechanism
align_weight: `--align-weight 1.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.4933.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 051: align_2p0

### Family
C

### Exact mechanism
align_weight: `--align-weight 2.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.3676.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 052: temp_0p05

### Family
C

### Exact mechanism
temperature: `--temperature 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.2024.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 053: temp_0p2

### Family
C

### Exact mechanism
temperature: `--temperature 0.2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0312; collapse True; RNA latent R2 -0.8876.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 054: temp_0p5

### Family
C

### Exact mechanism
temperature: `--temperature 0.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -1.0185.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 055: multipos_a0p25_t0p2

### Family
C

### Exact mechanism
multi_positive_alignment: `--multi-positive-alignment --align-weight 0.25 --temperature 0.2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -1.0350.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 056: multipos_a0p5_t0p2

### Family
C

### Exact mechanism
multi_positive_alignment: `--multi-positive-alignment --align-weight 0.5 --temperature 0.2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -1.0080.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 057: multipos_a1p0_t0p2

### Family
C

### Exact mechanism
multi_positive_alignment: `--multi-positive-alignment --align-weight 1.0 --temperature 0.2`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0312; collapse True; RNA latent R2 -0.9409.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 058: multipos_a0p5_t0p05

### Family
C

### Exact mechanism
multi_positive_alignment: `--multi-positive-alignment --align-weight 0.5 --temperature 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.4538.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 059: multipos_a1p5_t0p05

### Family
C

### Exact mechanism
multi_positive_alignment: `--multi-positive-alignment --align-weight 1.5 --temperature 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.1619.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 060: cf_0p005

### Family
D

### Exact mechanism
counterfactual_weight: `--counterfactual-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 061: cf_0p01

### Family
D

### Exact mechanism
counterfactual_weight: `--counterfactual-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 062: cf_0p02

### Family
D

### Exact mechanism
counterfactual_weight: `--counterfactual-weight 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 063: cf_0p05

### Family
D

### Exact mechanism
counterfactual_weight: `--counterfactual-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 064: cf_0p01_cycle_0p0

### Family
D

### Exact mechanism
counterfactual_cycle_balance: `--counterfactual-weight 0.01 --cycle-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6291.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 065: cf_0p01_cycle_0p1

### Family
D

### Exact mechanism
counterfactual_cycle_balance: `--counterfactual-weight 0.01 --cycle-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6372.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 066: cf_0p02_cycle_0p0

### Family
D

### Exact mechanism
counterfactual_cycle_balance: `--counterfactual-weight 0.02 --cycle-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6291.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 067: cf_0p02_cycle_0p1

### Family
D

### Exact mechanism
counterfactual_cycle_balance: `--counterfactual-weight 0.02 --cycle-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6372.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 068: cf_bottleneck_0p0

### Family
D

### Exact mechanism
response_bottleneck: `--counterfactual-weight 0.01 --response-bottleneck-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6337.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 069: cf_bottleneck_0p001

### Family
D

### Exact mechanism
response_bottleneck: `--counterfactual-weight 0.01 --response-bottleneck-weight 0.001`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6335.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 070: cf_bottleneck_0p02

### Family
D

### Exact mechanism
response_bottleneck: `--counterfactual-weight 0.01 --response-bottleneck-weight 0.02`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6324.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 071: batch_adv_0p0

### Family
E

### Exact mechanism
batch_adv_weight: `--batch-adv-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6327.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 072: batch_adv_0p005

### Family
E

### Exact mechanism
batch_adv_weight: `--batch-adv-weight 0.005`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6325.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 073: batch_adv_0p01

### Family
E

### Exact mechanism
batch_adv_weight: `--batch-adv-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6328.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 074: batch_adv_0p05

### Family
E

### Exact mechanism
batch_adv_weight: `--batch-adv-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6342.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 075: batch_adv_0p1

### Family
E

### Exact mechanism
batch_adv_weight: `--batch-adv-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6375.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 076: perturb_cls_0p0

### Family
E

### Exact mechanism
perturbation_cls_weight: `--perturbation-cls-weight 0.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6336.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 077: perturb_cls_0p01

### Family
E

### Exact mechanism
perturbation_cls_weight: `--perturbation-cls-weight 0.01`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6330.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 078: perturb_cls_0p1

### Family
E

### Exact mechanism
perturbation_cls_weight: `--perturbation-cls-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6343.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 079: batch_0p0_align_0p25

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.0 --align-weight 0.25`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0000; collapse True; RNA latent R2 -0.9532.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 080: batch_0p0_align_0p5

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.0 --align-weight 0.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8275.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 081: batch_0p05_align_0p25

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.05 --align-weight 0.25`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0312; collapse True; RNA latent R2 -0.9496.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 082: batch_0p05_align_0p5

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.05 --align-weight 0.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8238.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 083: batch_0p1_align_0p25

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.1 --align-weight 0.25`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0312; collapse True; RNA latent R2 -0.9465.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 084: batch_0p1_align_0p5

### Family
E

### Exact mechanism
batch_alignment_tradeoff: `--batch-adv-weight 0.1 --align-weight 0.5`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.8260.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 085: sched_w5_f0p0

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 5 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.0 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 086: sched_w5_f0p2

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 5 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.2 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 087: sched_w10_f0p0

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 10 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.0 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 088: sched_w10_f0p2

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 10 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.2 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 089: sched_w20_f0p0

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 20 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.0 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 090: sched_w20_f0p2

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 20 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.2 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 091: sched_w30_f0p0

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 30 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.0 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 092: sched_w30_f0p2

### Family
F

### Exact mechanism
objective_schedule: `--schedule-reconstruction-warmup-steps 30 --schedule-reconstruction-anneal-steps 20 --schedule-reconstruction-final-scale 0.2 --schedule-warmup-non-reconstruction-scale 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7732.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 093: mask_0p05

### Family
F

### Exact mechanism
mask_weight: `--rna-mask-weight 0.05 --image-mask-weight 0.05`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5366.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 094: mask_0p1

### Family
F

### Exact mechanism
mask_weight: `--rna-mask-weight 0.1 --image-mask-weight 0.1`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5912.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 095: mask_0p4

### Family
F

### Exact mechanism
mask_weight: `--rna-mask-weight 0.4 --image-mask-weight 0.4`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6614.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 096: mask_0p8

### Family
F

### Exact mechanism
mask_weight: `--rna-mask-weight 0.8 --image-mask-weight 0.8`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.7296.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 097: jepa_0p0_align_1p0

### Family
I

### Exact mechanism
minimal_objective_ablation: `--jepa-weight 0.0 --align-weight 1.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5994.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 098: jepa_0p25_align_1p0

### Family
I

### Exact mechanism
minimal_objective_ablation: `--jepa-weight 0.25 --align-weight 1.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6048.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 099: jepa_0p5_align_1p0

### Family
I

### Exact mechanism
minimal_objective_ablation: `--jepa-weight 0.5 --align-weight 1.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.6161.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 100: jepa_2p0_align_1p0

### Family
I

### Exact mechanism
minimal_objective_ablation: `--jepa-weight 2.0 --align-weight 1.0`

### Why this preserves JEPA
The RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.

### Compute cost estimate
CPU-only micro screen; no model width, dataset size, or image size increase.

### Observed effect
Recall@1 0.0625; collapse True; RNA latent R2 -0.5755.

### Verdict
TIER1_DISCARD_MODE_COLLAPSE.

## Change 101: raw_mlp_pls_distilled_head

### Family
J

### Exact mechanism
Added separate nonlinear raw MLP student outputs `rna_distilled_shared` and `image_distilled_shared`; trained only those heads against the frozen rank-3 PLS raw-linear readout for 150 CPU steps.

### Why this preserves JEPA
The protected retrieval path remains the frozen raw-linear PLS bridge readout. The JEPA/scRNA bridge forward contract remains intact, and the student head is evaluated separately before any promotion.

### Compute cost estimate
CPU-only Tier 2 check on `synth_micro` seeds 0/1/2; no model width, dataset size, image size, or real data increase.

### Observed effect
Mean recall@1 0.0938; RNA latent R2 0.2729; batch balanced accuracy 0.4688; no collapse; protected PLS drift 0.0.

### Verdict
TIER2_FAIL_SIGNAL_INCONSISTENT.

## Change 102: linear_clone_pls_distilled_head

### Family
J

### Exact mechanism
Added separate trainable linear student heads `rna_distilled_linear_projection` and `image_distilled_linear_projection`; cloned the train-split-only rank-3 PLS map into those heads with `install_prefit_pls_distillation_head`.

### Why this preserves JEPA
The frozen PLS retrieval path remains untouched. The cloned student is a separate module that can be serialized, trained, and evaluated independently before replacing retrieval.

### Compute cost estimate
CPU-only Tier 2 check on `synth_micro` seeds 0/1/2, 10 steps per seed, no auxiliary drift pressure, no real data.

### Observed effect
Mean recall@1 0.2396; RNA latent R2 0.5929; image latent R2 0.9134; batch balanced accuracy 0.4792; student-teacher MSE 0.0; protected PLS drift 0.0.

### Verdict
TIER2_PASS_CLEAN_ENGINEERING_BASELINE.

## Change 103: residual_mlp_pls_clone_adapter

### Family
J

### Exact mechanism
Added zero-initialized residual student outputs around the PLS linear clone: `rna_distilled_residual_shared` and `image_distilled_residual_shared`. The cloned linear heads stay frozen; only residual MLP heads and scalar gates train.

### Why this preserves JEPA
At initialization the residual student exactly equals the PLS clone. The protected frozen PLS retrieval path remains untouched, and the residual is evaluated independently.

### Compute cost estimate
CPU-only Tier 2 check on `synth_micro` seeds 0/1/2, 50 steps per seed, no real data.

### Observed effect
Mean recall@1 0.2396; RNA latent R2 0.5929; image latent R2 0.9134; batch balanced accuracy 0.4792; mean RNA residual scale 0.0044; mean image residual scale 0.0182; protected PLS drift 0.0.

### Verdict
TIER2_PASS_CLEAN_NO_IMPROVEMENT_DO_NOT_PROMOTE.

## Change 104: clone_frozen_counterfactual_decoder

### Family
K

### Exact mechanism
Added a clone-frozen counterfactual training script that freezes the protected PLS retrieval and exact linear clone, then trains only state/response/delta/RNA decoder heads on synthetic control-to-treated pairs.

### Why this preserves JEPA
The JEPA/scRNA bridge and shared retrieval geometry remain active and frozen. The change targets only the counterfactual transition head.

### Compute cost estimate
CPU-only seed-0 probes; no real data, no GPU, no model width or dataset increase.

### Observed effect
Generative RNA decoding improved training loss but did not pass counterfactual gates. Letting the perturbation encoder train did not fix the failure.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_FAIL.

## Change 105: residual_rna_counterfactual_decoder

### Family
K

### Exact mechanism
Added `counterfactual_rna_residual`, where counterfactual RNA is source pseudobulk plus a learned delta. The residual delta decoder is zero-initialized in the training script.

### Why this preserves JEPA
The protected PLS shared geometry remains frozen. The residual decoder starts as source-as-target and must earn any perturbation delta improvement.

### Compute cost estimate
CPU-only Tier 2 check on `synth_micro` seeds 0/1/2, 25 steps per seed, no real data.

### Observed effect
No protected retrieval/R2/batch regression. Direction, logFC, pseudobulk, and top-DE overlap improved in all seeds, but program-level effect recovery regressed in seeds 1 and 2.

### Verdict
TIER2_FAIL_PROGRAM_RECOVERY_INCONSISTENT.

## Change 107: program_factorized_residual_counterfactual_decoder

### Family
K

### Exact mechanism
Pre-registered a program-factorized counterfactual RNA decoder. The decoder predicts one delta per synthetic generator program, broadcasts by `gene_program_assignment`, and can add a zero-initialized within-program residual with each program's mean removed.

### Why this preserves JEPA
The protected PLS retrieval path and exact linear clone remain frozen. The new parameterization is behind explicit config/script flags and is initialized to source-as-target when residual RNA prediction is enabled.

### Compute cost estimate
CPU-only `synth_micro` Tier 1 seed 0, then seeds 0/1/2 only if Tier 1 passes. No real data, no real marker/pathway resources, no GPU-heavy work.

### Observed effect
Tier 2 completed for the no-within-residual 25-step decoder. Protected geometry was unchanged in all seeds. Seed 0 passed the counterfactual gate, but seeds 1 and 2 failed program recovery. Mean final delta/target ratio was 0.0202; mean program contribution ratio was 0.9600; cap-hit fraction was 0.0. The within-program residual audit also failed seeds 1 and 2 and let residual contributions dominate the program head.

### Verdict
TIER2_FAIL_PROGRAM_RECOVERY_INCONSISTENT.

## Change 110: direct_biological_metadata_program_context

### Family
K

### Exact mechanism
Pre-registered a direct no-batch metadata context for the program-factorized decoder. The program head may receive perturbation ID one-hot, cell-line ID one-hot, dose, time, and source program means.

### Why this preserves JEPA
The protected shared geometry is unchanged. The direct context contains biological condition metadata already available to the perturbation encoder and excludes technical batch metadata to avoid leakage.

### Compute cost estimate
CPU-only `synth_micro` Tier 1 seed 0 at 100 steps, then seeds 0/1/2 only if Tier 1 passes. No real data or external biological resources.

### Observed effect
Tier 2 completed for the source-program-context plus direct no-batch metadata decoder, no within-program residual, 100 steps. Protected geometry was unchanged in all seeds. Seed 0 passed, seed 1 improved program recovery only weakly, and seed 2 regressed. Mean program recovery was 0.1181 with high cross-seed variance; final delta/target ratio rose to 0.1227 without cap hits.

### Verdict
TIER2_FAIL_PROGRAM_RECOVERY_INCONSISTENT.

## Change 109: source_conditioned_program_factorized_decoder

### Family
K

### Exact mechanism
Pre-registered source-program conditioning for the factorized counterfactual decoder. The bridge pools source RNA into synthetic program means, concatenates those means to `z_counterfactual`, predicts program deltas, and broadcasts them back to genes.

### Why this preserves JEPA
The protected shared retrieval geometry is unchanged and frozen. The new signal is source RNA already provided to the counterfactual path, aggregated only by synthetic generator program assignment.

### Compute cost estimate
CPU-only `synth_micro` Tier 1 seed 0 at 100 steps, then seeds 0/1/2 only if Tier 1 passes. No real data or external biological resources.

### Observed effect
Tier 2 completed for source-program context, no within-program residual, 100 steps. Protected geometry was unchanged. Seed 0 program recovery improved strongly, while seeds 1 and 2 failed the full counterfactual gate. Mean program recovery was 0.0391 and final delta/target ratio was 0.0569. The within-program residual source-context audit increased amplitude but did not fix seed 2.

### Verdict
TIER2_FAIL_PROGRAM_RECOVERY_INCONSISTENT.

## Change 108: delta_mse_program_factorized_training

### Family
K

### Exact mechanism
Pre-registered an explicit `--delta-mse` objective option for the clone-frozen counterfactual decoder. The option trains residual counterfactual output against target delta rather than absolute target RNA.

### Why this preserves JEPA
It changes only the low-compute synthetic counterfactual repair script. The protected PLS retrieval path, exact linear clone, synthetic data split, and evaluation gates remain unchanged.

### Compute cost estimate
CPU-only `synth_micro` Tier 1 seed 0, then seeds 0/1/2 only if Tier 1 passes. No real data or external biological resources.

### Observed effect
Tier 1 seed 0 completed. Delta-MSE preserved geometry and reproduced the seed-0 counterfactual pass, but the final delta/target ratio stayed low at 0.0301. It was not advanced because adjacent program-factorized Tier 2 checks exposed the same seed 1/2 program-recovery failure pattern.

### Verdict
TIER1_KEEP_COUNTERFACTUAL_SIGNAL_NO_TIER2.

## Change 106: synthetic_program_delta_loss

### Family
K

### Exact mechanism
Added `--program-weight` to the clone-frozen counterfactual decoder script. The auxiliary loss averages predicted and target RNA deltas by synthetic generator program and penalizes program-delta MSE.

### Why this preserves JEPA
The loss uses only synthetic generator metadata and does not alter shared retrieval geometry. Real marker/pathway data remains unused.

### Compute cost estimate
CPU-only Tier 2 check on `synth_micro` seeds 0/1/2, 25 steps per seed, no real data.

### Observed effect
Gate passes stayed at 1/3. Mean program-level recovery was 0.0326 versus 0.0356 for the unweighted residual-delta run.

### Verdict
TIER2_FAIL_PROGRAM_RECOVERY_INCONSISTENT.

## Change 111: seed2_meta_src_dw0p05_pw0p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0610; direction delta 0.5127; logFC delta 0.1774; protected geometry True; final delta/target ratio 0.0174.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 220: seed2_train_only_condition_mean_table

### Family
N

### Exact mechanism
`python scripts/run_family_n_distillation.py --dataset synth_micro --seed 2 --rank 3 --device cpu --output-dir outputs/autoresearch_synth_lite/diagnostics/FAMILY_N_DISTILLATION`

Candidate A stores train-split perturbed means keyed only by `perturbation_id`, `cell_line_id`, `dose`, and `time`; `batch_id` is excluded. Unseen-key fallback uses nearest same perturbation/cell-line train key, then train perturbation mean, then global train mean.

### Why this preserves JEPA
No bridge parameters or protected PLS readouts are trained. This is a train-only counterfactual teacher/reference table for the separate counterfactual surface.

### Compute cost estimate
CPU deterministic table build/evaluation; no real data, marker/pathway resources, pretrained assets, or GPU.

### Observed effect
Exact train-key coverage 1.0000; program 0.7520; direction 0.6899; logFC 0.7502; pseudobulk 0.8725; top50 0.5683; protected retrieval/R2/batch metrics unchanged.

### Verdict
REFERENCE_TEACHER_COMPLETE_MATCHES_EXP213.

## Change 221: seed2_distilled_linear_condition_mean

### Family
N

### Exact mechanism
Closed-form ridge linear student trained from no-batch biological/source-program features to the train-only matched-mean teacher target.

### Why this preserves JEPA
No bridge weights are modified; the learned object is a separate conditional mean predictor using synthetic train-pair statistics only.

### Compute cost estimate
CPU closed-form solve after Family N data preparation.

### Observed effect
Leakage gate true; train teacher MSE 6.56e-14; test program 0.7353; direction 0.6854; logFC 0.7261; pseudobulk 0.8506; top50 0.5583.

### Verdict
TIER1_DISTILLATION_COMPLETE_UNDER_TEACHER.

## Change 222: seed2_distilled_mlp_condition_mean

### Family
N

### Exact mechanism
Small two-layer MLP student trained on the same no-batch features and train-only teacher targets for 1200 CPU full-batch steps.

### Why this preserves JEPA
The model is an isolated conditional mean student; protected bridge geometry remains frozen by construction.

### Compute cost estimate
CPU Tier 1 synthetic run; no external assets.

### Observed effect
Leakage gate true; train teacher MSE 2.61e-06; test program 0.7033; direction 0.6738; logFC 0.7286; pseudobulk 0.8574; top50 0.5525.

### Verdict
TIER1_DISTILLATION_COMPLETE_UNDER_TEACHER.

## Change 223: seed2_linear_condition_mean_hybrid_alpha0p1

### Family
N

### Exact mechanism
Shrinkage sweep `prediction = alpha * learned_model + (1 - alpha) * train_only_table` over linear and MLP students. Recorded best nonzero hybrid by program/logFC ordering: linear `alpha = 0.1`.

### Why this preserves JEPA
No protected bridge weights are touched; fallback and learned predictions are fit exclusively from train split statistics/features.

### Compute cost estimate
CPU vectorized sweep over saved Family N predictions.

### Observed effect
Leakage gate true; program 0.7526; direction 0.6882; logFC 0.7500; pseudobulk 0.8717; top50 0.5658. The alpha-0.25 linear hybrid reached top50 0.5700 but lost more logFC/pseudobulk.

### Verdict
TIER1_HYBRID_SWEEP_COMPLETE_NO_PROMOTION.

## Change 213: seed2_no_batch_matched_perturbed_mean

### Family
M

### Exact mechanism
Added `scripts/run_family_m_transport_baselines.py` and evaluated a formal direct matched perturbed-mean baseline. Matching key is exactly `perturbation_id`, `cell_line_id`, `dose`, and `time`; `batch_id` is excluded.

### Why this preserves JEPA
No JEPA/scRNA bridge parameters are changed. The frozen PLS geometry is only read for protected metric reporting.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation; no real data, marker/pathway resources, pretrained biological assets, or network installs.

### Observed effect
Exact key coverage 1.0000; program 0.7520; direction 0.6899; logFC 0.7502; pseudobulk 0.8725; top50 0.5683; delta/target ratio 0.7400. Protected geometry unchanged.

### Verdict
BASELINE_COMPLETE_STRONG_EXACT_CONDITION_MATCH.

## Change 214: seed2_no_batch_residualized_matching

### Family
M

### Exact mechanism
Residualized matching baseline: `x_cf = x_control + mean_delta[perturbation_id, cell_line_id, dose, time]`, with deltas fit from train pairs and aggregated across batches.

### Why this preserves JEPA
No model parameters are changed; evaluation is a deterministic baseline.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation.

### Observed effect
Exact key coverage 1.0000; program 0.3502; direction 0.5312; logFC 0.1268; pseudobulk 0.7491; top50 0.4150; delta/target ratio 0.3421.

### Verdict
BASELINE_COMPLETE_RESIDUALIZED_MATCHING_REFERENCE.

## Change 215: seed2_knn_residual_transport_program_k1

### Family
M

### Exact mechanism
No-batch kNN residual transport in source program space. Each train control cell contributes `target_train_mean - source_train_cell` within the matched biological condition; the nearest source cell residual is applied to each test control.

### Why this preserves JEPA
Deterministic nonparametric counterfactual predictor; frozen bridge metrics are unchanged.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation.

### Observed effect
Program 0.5146; direction 0.5161; logFC 0.0513; pseudobulk 0.6216; top50 0.4075; delta/target ratio 0.7829.

### Verdict
TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE.

## Change 216: seed2_knn_residual_transport_program_k3

### Family
M

### Exact mechanism
No-batch source program kNN residual transport with `k=3`.

### Why this preserves JEPA
Deterministic nonparametric counterfactual predictor; frozen bridge metrics are unchanged.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation.

### Observed effect
Program 0.5293; direction 0.5488; logFC 0.0623; pseudobulk 0.6954; top50 0.4308; delta/target ratio 0.5087.

### Verdict
TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE.

## Change 217: seed2_knn_residual_transport_program_k5

### Family
M

### Exact mechanism
No-batch source program kNN residual transport with `k=5`.

### Why this preserves JEPA
Deterministic nonparametric counterfactual predictor; frozen bridge metrics are unchanged.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation.

### Observed effect
Program 0.5145; direction 0.5450; logFC -0.0038; pseudobulk 0.7136; top50 0.4217; delta/target ratio 0.4262.

### Verdict
TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE.

## Change 218: seed2_knn_residual_transport_program_k8

### Family
M

### Exact mechanism
No-batch source program kNN residual transport with `k=8`.

### Why this preserves JEPA
Deterministic nonparametric counterfactual predictor; frozen bridge metrics are unchanged.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation.

### Observed effect
Program 0.4593; direction 0.5433; logFC 0.0795; pseudobulk 0.7361; top50 0.4167; delta/target ratio 0.3825.

### Verdict
TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE.

## Change 219: seed2_sinkhorn_residual_transport_program_eps0p5

### Family
M

### Exact mechanism
In-repo entropic Sinkhorn transport in source program space, condition-specific on `perturbation_id`, `cell_line_id`, `dose`, and `time`, with `batch_id` excluded. Test controls receive kernel-interpolated barycentric residuals.

### Why this preserves JEPA
Deterministic nonparametric counterfactual predictor; no bridge weights are updated.

### Compute cost estimate
CPU deterministic seed-2 synthetic evaluation; no POT/GeomLoss/network install.

### Observed effect
Program 0.4211; direction 0.5343; logFC 0.0657; pseudobulk 0.7290; top50 0.4217; delta/target ratio 0.3827; mean support 12 train cells per condition.

### Verdict
TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE.

## Change 211: seed2_sparse_residual_rank4_topde50

### Family
L

### Exact mechanism
Added a separate sparse perturbation residual head in `perturb_jepa/models/sparse_perturbation_residual.py` and trained it with `scripts/train_sparse_perturbation_residual_head.py`. The protected bridge is wrapped with frozen PLS raw-linear readouts and frozen clone features. The sparse head builds `z_pert` from source program means and no-batch biological metadata, predicts only a delta RNA residual, and combines a program-factorized term with a learned rank-4 gene dictionary.

### Losses and gates
Tier-1 loss used low global delta MSE weight `0.05`, top-DE weight `8.0`, active-program gene weight `4.0`, program direction weight `1.0`, program sign weight `0.25`, outside-effect sparsity `0.02`, and z_inv/z_pert decorrelation `0.01`. Protected retrieval/R2/batch gates were reused exactly through `evaluate_step0` and the existing `_counterfactual_gate_pass` helper.

### Why this preserves JEPA
The JEPA/scRNA bridge and protected PLS/linear clone geometry remain active and frozen. No shared retrieval encoder/readout code was modified; the new head is an isolated synthetic counterfactual residual branch.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 100 steps; no real data, real marker/pathway resources, pretrained assets, or biological graph priors.

### Observed effect
Program recovery delta `+0.2877`; direction delta `+0.5793`; logFC delta `-0.0830`; top50 delta `0.0000`; protected geometry `True`; final delta/target ratio `0.0150`. Matching-mean baseline achieved program recovery `0.3502`, logFC `0.1268`, and top50 `0.4150`.

### Verdict
TIER1_DISCARD_STRONG_BASELINE_UNDERPERFORM.

## Change 212: seed2_sparse_residual_amp_rank8_topde50

### Family
L

### Exact mechanism
Amplitude-focused variant of the isolated sparse perturbation residual head: rank-8 low-rank gene dictionary, 150 CUDA steps, `lr=0.003`, global delta weight `0.5`, top-DE weight `12.0`, program-gene weight `2.0`, sparsity `0.002`, and decorrelation `0.001`.

### Why this preserves JEPA
Same as Change 211: the protected JEPA/scRNA bridge, PLS raw-linear retrieval path, and clone geometry are frozen. Only the synthetic counterfactual residual head is trained.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 150 steps; no real data, real marker/pathway resources, pretrained assets, or biological graph priors.

### Observed effect
Program recovery `0.2750`; direction accuracy `0.5622`; logFC `0.1088`; top50 `0.3917`; protected geometry `True`; final sparse delta/target ratio `0.4116`. Matching mean remained stronger on program recovery `0.3502`, logFC `0.1268`, and top50 `0.4150`.

### Verdict
TIER1_DISCARD_STRONG_BASELINE_UNDERPERFORM.

## Post-210 Architectural Synthesis

The strict GPU continuation tested 100 additional Family K trials without changing the protected PLS retrieval/readout path or using real data/resources.

- MLP program-decoder loss rebalancing and delta-MSE variants were not sufficient.
- Context ablations showed source program context and no-batch biological metadata are both useful; removing either worsened seed-2 program recovery.
- Within-program residuals did not repair seed 2.
- Shallow linear program decoders improved seed 2 from regression to weak positive recovery.
- Train-split-only prefit ridge initialization for the linear program decoder produced the strongest architectural signal, with seed-2 program recovery around `+0.25`.

Observed remaining failure: top-50 DE overlap did not clear the full counterfactual gate. The ridge-initialized linear program decoder is a near-miss diagnostic candidate, not a promotable architecture.

## Change 112: seed2_meta_src_dw0p05_pw0p1

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0608; direction delta 0.5127; logFC delta 0.1755; protected geometry True; final delta/target ratio 0.0180.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 113: seed2_meta_src_dw0p05_pw0p25

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0606; direction delta 0.5127; logFC delta 0.1730; protected geometry True; final delta/target ratio 0.0187.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 114: seed2_meta_src_dw0p05_pw0p5

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0603; direction delta 0.5127; logFC delta 0.1690; protected geometry True; final delta/target ratio 0.0197.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 115: seed2_meta_src_dw0p05_pw1p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0601; direction delta 0.5203; logFC delta 0.1629; protected geometry True; final delta/target ratio 0.0212.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 116: seed2_meta_src_dw0p05_pw2p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.05 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0604; direction delta 0.5216; logFC delta 0.1556; protected geometry True; final delta/target ratio 0.0227.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 117: seed2_meta_src_dw0p1_pw0p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0591; direction delta 0.5299; logFC delta 0.1859; protected geometry True; final delta/target ratio 0.0148.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 118: seed2_meta_src_dw0p1_pw0p1

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0597; direction delta 0.5227; logFC delta 0.1848; protected geometry True; final delta/target ratio 0.0153.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 119: seed2_meta_src_dw0p1_pw0p25

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0602; direction delta 0.5227; logFC delta 0.1830; protected geometry True; final delta/target ratio 0.0160.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 120: seed2_meta_src_dw0p1_pw0p5

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0605; direction delta 0.5137; logFC delta 0.1803; protected geometry True; final delta/target ratio 0.0170.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 121: seed2_meta_src_dw0p1_pw1p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0604; direction delta 0.5127; logFC delta 0.1759; protected geometry True; final delta/target ratio 0.0185.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 122: seed2_meta_src_dw0p1_pw2p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.1 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0605; direction delta 0.5127; logFC delta 0.1689; protected geometry True; final delta/target ratio 0.0202.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 123: seed2_meta_src_dw0p2_pw0p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0608; direction delta 0.5302; logFC delta 0.1940; protected geometry True; final delta/target ratio 0.0120.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 124: seed2_meta_src_dw0p2_pw0p1

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0595; direction delta 0.5312; logFC delta 0.1935; protected geometry True; final delta/target ratio 0.0124.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 125: seed2_meta_src_dw0p2_pw0p25

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0581; direction delta 0.5299; logFC delta 0.1924; protected geometry True; final delta/target ratio 0.0130.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 126: seed2_meta_src_dw0p2_pw0p5

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0574; direction delta 0.5299; logFC delta 0.1904; protected geometry True; final delta/target ratio 0.0139.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 127: seed2_meta_src_dw0p2_pw1p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0586; direction delta 0.5299; logFC delta 0.1868; protected geometry True; final delta/target ratio 0.0153.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 128: seed2_meta_src_dw0p2_pw2p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0600; direction delta 0.5227; logFC delta 0.1802; protected geometry True; final delta/target ratio 0.0172.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 129: seed2_meta_src_dw0p4_pw0p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0671; direction delta 0.5312; logFC delta 0.1912; protected geometry True; final delta/target ratio 0.0095.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 130: seed2_meta_src_dw0p4_pw0p1

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0667; direction delta 0.5312; logFC delta 0.1923; protected geometry True; final delta/target ratio 0.0098.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 131: seed2_meta_src_dw0p4_pw0p25

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0659; direction delta 0.5312; logFC delta 0.1936; protected geometry True; final delta/target ratio 0.0102.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 132: seed2_meta_src_dw0p4_pw0p5

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0644; direction delta 0.5312; logFC delta 0.1946; protected geometry True; final delta/target ratio 0.0108.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 133: seed2_meta_src_dw0p4_pw1p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0613; direction delta 0.5312; logFC delta 0.1943; protected geometry True; final delta/target ratio 0.0120.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 134: seed2_meta_src_dw0p4_pw2p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.4 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0579; direction delta 0.5299; logFC delta 0.1907; protected geometry True; final delta/target ratio 0.0139.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 135: seed2_meta_src_dw0p8_pw0p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0662; direction delta 0.5299; logFC delta 0.1806; protected geometry True; final delta/target ratio 0.0082.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 136: seed2_meta_src_dw0p8_pw0p1

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0666; direction delta 0.5299; logFC delta 0.1821; protected geometry True; final delta/target ratio 0.0084.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 137: seed2_meta_src_dw0p8_pw0p25

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0670; direction delta 0.5271; logFC delta 0.1841; protected geometry True; final delta/target ratio 0.0085.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 138: seed2_meta_src_dw0p8_pw0p5

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0674; direction delta 0.5275; logFC delta 0.1870; protected geometry True; final delta/target ratio 0.0089.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 139: seed2_meta_src_dw0p8_pw1p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0671; direction delta 0.5312; logFC delta 0.1912; protected geometry True; final delta/target ratio 0.0095.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 140: seed2_meta_src_dw0p8_pw2p0

### Family
K

### Exact mechanism
seed2_program_loss_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --direction-weight 0.8 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0644; direction delta 0.5312; logFC delta 0.1946; protected geometry True; final delta/target ratio 0.0108.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 141: seed2_delta_meta_src_dw0p05_pw0p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0610; direction delta 0.5127; logFC delta 0.1774; protected geometry True; final delta/target ratio 0.0174.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 142: seed2_delta_meta_src_dw0p05_pw0p1

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0608; direction delta 0.5127; logFC delta 0.1755; protected geometry True; final delta/target ratio 0.0180.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 143: seed2_delta_meta_src_dw0p05_pw0p25

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0606; direction delta 0.5127; logFC delta 0.1730; protected geometry True; final delta/target ratio 0.0187.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 144: seed2_delta_meta_src_dw0p05_pw0p5

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0603; direction delta 0.5127; logFC delta 0.1690; protected geometry True; final delta/target ratio 0.0197.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 145: seed2_delta_meta_src_dw0p05_pw1p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0601; direction delta 0.5203; logFC delta 0.1629; protected geometry True; final delta/target ratio 0.0212.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 146: seed2_delta_meta_src_dw0p05_pw2p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.05 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0604; direction delta 0.5216; logFC delta 0.1556; protected geometry True; final delta/target ratio 0.0227.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 147: seed2_delta_meta_src_dw0p1_pw0p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0591; direction delta 0.5299; logFC delta 0.1859; protected geometry True; final delta/target ratio 0.0148.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 148: seed2_delta_meta_src_dw0p1_pw0p1

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0597; direction delta 0.5227; logFC delta 0.1848; protected geometry True; final delta/target ratio 0.0153.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 149: seed2_delta_meta_src_dw0p1_pw0p25

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0602; direction delta 0.5227; logFC delta 0.1830; protected geometry True; final delta/target ratio 0.0160.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 150: seed2_delta_meta_src_dw0p1_pw0p5

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0605; direction delta 0.5137; logFC delta 0.1803; protected geometry True; final delta/target ratio 0.0170.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 151: seed2_delta_meta_src_dw0p1_pw1p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0604; direction delta 0.5127; logFC delta 0.1759; protected geometry True; final delta/target ratio 0.0185.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 152: seed2_delta_meta_src_dw0p1_pw2p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.1 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0605; direction delta 0.5127; logFC delta 0.1689; protected geometry True; final delta/target ratio 0.0202.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 153: seed2_delta_meta_src_dw0p2_pw0p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0608; direction delta 0.5302; logFC delta 0.1940; protected geometry True; final delta/target ratio 0.0120.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 154: seed2_delta_meta_src_dw0p2_pw0p1

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0595; direction delta 0.5312; logFC delta 0.1935; protected geometry True; final delta/target ratio 0.0124.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 155: seed2_delta_meta_src_dw0p2_pw0p25

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0581; direction delta 0.5299; logFC delta 0.1924; protected geometry True; final delta/target ratio 0.0130.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 156: seed2_delta_meta_src_dw0p2_pw0p5

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0574; direction delta 0.5299; logFC delta 0.1904; protected geometry True; final delta/target ratio 0.0139.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 157: seed2_delta_meta_src_dw0p2_pw1p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0586; direction delta 0.5299; logFC delta 0.1868; protected geometry True; final delta/target ratio 0.0153.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 158: seed2_delta_meta_src_dw0p2_pw2p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0600; direction delta 0.5227; logFC delta 0.1802; protected geometry True; final delta/target ratio 0.0172.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 159: seed2_delta_meta_src_dw0p4_pw0p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0671; direction delta 0.5312; logFC delta 0.1912; protected geometry True; final delta/target ratio 0.0095.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 160: seed2_delta_meta_src_dw0p4_pw0p1

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 0.1`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0667; direction delta 0.5312; logFC delta 0.1923; protected geometry True; final delta/target ratio 0.0098.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 161: seed2_delta_meta_src_dw0p4_pw0p25

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0659; direction delta 0.5312; logFC delta 0.1936; protected geometry True; final delta/target ratio 0.0102.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 162: seed2_delta_meta_src_dw0p4_pw0p5

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0644; direction delta 0.5312; logFC delta 0.1946; protected geometry True; final delta/target ratio 0.0108.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 163: seed2_delta_meta_src_dw0p4_pw1p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0613; direction delta 0.5312; logFC delta 0.1943; protected geometry True; final delta/target ratio 0.0120.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 164: seed2_delta_meta_src_dw0p4_pw2p0

### Family
K

### Exact mechanism
seed2_delta_mse_program_grid: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --delta-mse --direction-weight 0.4 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0579; direction delta 0.5299; logFC delta 0.1907; protected geometry True; final delta/target ratio 0.0139.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 165: seed2_linear_meta_src_dw0p1_pw0p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.1 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0103; direction delta 0.5457; logFC delta 0.0433; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 166: seed2_linear_meta_src_dw0p1_pw0p25

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.1 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0092; direction delta 0.5416; logFC delta 0.0430; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 167: seed2_linear_meta_src_dw0p1_pw1p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.1 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0075; direction delta 0.5416; logFC delta 0.0428; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 168: seed2_linear_meta_src_dw0p1_pw4p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.1 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0056; direction delta 0.5385; logFC delta 0.0428; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 169: seed2_linear_meta_src_dw0p2_pw0p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0146; direction delta 0.5391; logFC delta 0.0453; protected geometry True; final delta/target ratio 0.1490.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 170: seed2_linear_meta_src_dw0p2_pw0p25

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0131; direction delta 0.5436; logFC delta 0.0444; protected geometry True; final delta/target ratio 0.1492.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 171: seed2_linear_meta_src_dw0p2_pw1p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0103; direction delta 0.5457; logFC delta 0.0433; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 172: seed2_linear_meta_src_dw0p2_pw4p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0069; direction delta 0.5385; logFC delta 0.0427; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 173: seed2_linear_meta_src_dw0p4_pw0p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.4 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0206; direction delta 0.5364; logFC delta 0.0503; protected geometry True; final delta/target ratio 0.1478.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 174: seed2_linear_meta_src_dw0p4_pw0p25

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.4 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0185; direction delta 0.5398; logFC delta 0.0484; protected geometry True; final delta/target ratio 0.1483.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 175: seed2_linear_meta_src_dw0p4_pw1p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.4 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0146; direction delta 0.5391; logFC delta 0.0453; protected geometry True; final delta/target ratio 0.1490.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 176: seed2_linear_meta_src_dw0p4_pw4p0

### Family
K

### Exact mechanism
seed2_linear_program_decoder: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --direction-weight 0.4 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.0092; direction delta 0.5416; logFC delta 0.0430; protected geometry True; final delta/target ratio 0.1494.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE.

## Change 177: seed2_prefitridge_a1em06_pw0p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 1e-06 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2512; direction delta 0.5608; logFC delta 0.0488; protected geometry True; final delta/target ratio 0.3387.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 178: seed2_prefitridge_a1em06_pw0p25

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 1e-06 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2503; direction delta 0.5635; logFC delta 0.0463; protected geometry True; final delta/target ratio 0.3399.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 179: seed2_prefitridge_a1em06_pw1p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 1e-06 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2515; direction delta 0.5591; logFC delta 0.0453; protected geometry True; final delta/target ratio 0.3404.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 180: seed2_prefitridge_a1em06_pw4p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 1e-06 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2523; direction delta 0.5598; logFC delta 0.0492; protected geometry True; final delta/target ratio 0.3399.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 181: seed2_prefitridge_a0p0001_pw0p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.0001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2512; direction delta 0.5608; logFC delta 0.0489; protected geometry True; final delta/target ratio 0.3387.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 182: seed2_prefitridge_a0p0001_pw0p25

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.0001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2503; direction delta 0.5635; logFC delta 0.0464; protected geometry True; final delta/target ratio 0.3399.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 183: seed2_prefitridge_a0p0001_pw1p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.0001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2514; direction delta 0.5591; logFC delta 0.0455; protected geometry True; final delta/target ratio 0.3404.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 184: seed2_prefitridge_a0p0001_pw4p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.0001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2522; direction delta 0.5598; logFC delta 0.0494; protected geometry True; final delta/target ratio 0.3399.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 185: seed2_prefitridge_a0p001_pw0p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2507; direction delta 0.5608; logFC delta 0.0505; protected geometry True; final delta/target ratio 0.3387.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 186: seed2_prefitridge_a0p001_pw0p25

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2499; direction delta 0.5635; logFC delta 0.0480; protected geometry True; final delta/target ratio 0.3399.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 187: seed2_prefitridge_a0p001_pw1p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2511; direction delta 0.5635; logFC delta 0.0474; protected geometry True; final delta/target ratio 0.3402.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 188: seed2_prefitridge_a0p001_pw4p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.001 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2518; direction delta 0.5598; logFC delta 0.0511; protected geometry True; final delta/target ratio 0.3397.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 189: seed2_prefitridge_a0p01_pw0p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.01 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2466; direction delta 0.5577; logFC delta 0.0591; protected geometry True; final delta/target ratio 0.3389.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 190: seed2_prefitridge_a0p01_pw0p25

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.01 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2470; direction delta 0.5615; logFC delta 0.0581; protected geometry True; final delta/target ratio 0.3388.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 191: seed2_prefitridge_a0p01_pw1p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.01 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2486; direction delta 0.5615; logFC delta 0.0605; protected geometry True; final delta/target ratio 0.3385.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 192: seed2_prefitridge_a0p01_pw4p0

### Family
K

### Exact mechanism
seed2_prefit_program_ridge: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --linear-program-decoder --prefit-program-ridge --prefit-program-ridge-alpha 0.01 --prefit-program-ridge-repeats 4 --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta 0.2494; direction delta 0.5587; logFC delta 0.0636; protected geometry True; final delta/target ratio 0.3378.

### Verdict
TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE.

## Change 193: seed2_context_src1_meta0_pw0p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1727; direction delta 0.5309; logFC delta 0.1406; protected geometry True; final delta/target ratio 0.0191.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 194: seed2_context_src1_meta0_pw0p5

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --direction-weight 0.2 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1615; direction delta 0.5309; logFC delta 0.1316; protected geometry True; final delta/target ratio 0.0224.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 195: seed2_context_src1_meta0_pw2p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1181; direction delta 0.5309; logFC delta 0.1156; protected geometry True; final delta/target ratio 0.0280.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 196: seed2_context_src0_meta1_pw0p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-metadata-context --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1333; direction delta 0.5268; logFC delta 0.0585; protected geometry True; final delta/target ratio 0.0136.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 197: seed2_context_src0_meta1_pw0p5

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-metadata-context --direction-weight 0.2 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1089; direction delta 0.5302; logFC delta 0.0588; protected geometry True; final delta/target ratio 0.0161.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 198: seed2_context_src0_meta1_pw2p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-metadata-context --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0971; direction delta 0.5247; logFC delta 0.0597; protected geometry True; final delta/target ratio 0.0208.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 199: seed2_context_src0_meta0_pw0p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --direction-weight 0.2 --program-weight 0.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0790; direction delta 0.5309; logFC delta 0.1023; protected geometry True; final delta/target ratio 0.0109.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 200: seed2_context_src0_meta0_pw0p5

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --direction-weight 0.2 --program-weight 0.5`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0785; direction delta 0.5309; logFC delta 0.1019; protected geometry True; final delta/target ratio 0.0128.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 201: seed2_context_src0_meta0_pw2p0

### Family
K

### Exact mechanism
seed2_context_ablation: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0789; direction delta 0.5330; logFC delta 0.0921; protected geometry True; final delta/target ratio 0.0169.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 202: seed2_wres_abs_pw1p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0549; direction delta 0.5199; logFC delta 0.1589; protected geometry True; final delta/target ratio 0.0394.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 203: seed2_wres_delta_pw1p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 1.0 --delta-mse`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0549; direction delta 0.5199; logFC delta 0.1589; protected geometry True; final delta/target ratio 0.0394.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 204: seed2_wres_abs_pw2p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 2.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0570; direction delta 0.5237; logFC delta 0.1607; protected geometry True; final delta/target ratio 0.0400.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 205: seed2_wres_delta_pw2p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 2.0 --delta-mse`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0570; direction delta 0.5237; logFC delta 0.1607; protected geometry True; final delta/target ratio 0.0400.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 206: seed2_wres_abs_pw4p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 4.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0586; direction delta 0.5254; logFC delta 0.1621; protected geometry True; final delta/target ratio 0.0406.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 207: seed2_wres_delta_pw4p0

### Family
K

### Exact mechanism
seed2_within_program_residual_control: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --within-program-residual --direction-weight 0.2 --program-weight 4.0 --delta-mse`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0586; direction delta 0.5254; logFC delta 0.1621; protected geometry True; final delta/target ratio 0.0406.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 208: seed2_lr0p0003_pw0p25

### Family
K

### Exact mechanism
seed2_optimizer_sensitivity: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --lr 0.0003 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0928; direction delta 0.5333; logFC delta 0.1742; protected geometry True; final delta/target ratio 0.0026.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 209: seed2_lr0p0003_pw1p0

### Family
K

### Exact mechanism
seed2_optimizer_sensitivity: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --lr 0.0003 --direction-weight 0.2 --program-weight 1.0`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.1011; direction delta 0.5326; logFC delta 0.1788; protected geometry True; final delta/target ratio 0.0028.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 210: seed2_lr0p0005_pw0p25

### Family
K

### Exact mechanism
seed2_optimizer_sensitivity: `/usr/local/bin/python scripts/train_clone_counterfactual_decoder.py --dataset synth_micro --seed 2 --rank 3 --steps 50 --device cuda --output-dir outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER --residual-rna --program-factorized-rna --program-condition-source --program-metadata-context --lr 0.0005 --direction-weight 0.2 --program-weight 0.25`

### Why this preserves JEPA
The JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.

### Compute cost estimate
GPU Tier 1 seed-2 synthetic run, 50 steps; no real data, real marker/pathway resources, or pretrained assets.

### Observed effect
Program delta -0.0945; direction delta 0.5312; logFC delta 0.1844; protected geometry True; final delta/target ratio 0.0048.

### Verdict
TIER1_DISCARD_SEED2_PROGRAM_REGRESSION.

## Change 224: seed2_count_audit_source_as_target

### Family
O

### Exact mechanism
`python scripts/run_family_o_count_likelihood.py --dataset synth_micro --seed 2 --rank 3 --device cpu --append-results`, Candidate A source-as-target count audit.

### Why this preserves JEPA
No bridge weights are trained. The change adds a synthetic-only diagnostic path over `SyntheticBiologyLiteDataset.observed_counts`.

### Compute cost estimate
CPU Tier 1 seed-2 diagnostic; no real data, marker/pathway resources, pretrained assets, or batch metadata features.

### Observed effect
Raw counts available `true`; pseudo-count path used `false`; zero fraction `0.2026`; Poisson NLL `513.3838`; program `-0.0393`; logFC `0.1191`; top50 `0.2925`.

### Verdict
COUNT_AUDIT_BASELINE_COMPLETE.

## Change 225: seed2_train_global_count_mean_poisson_baseline

### Family
O

### Exact mechanism
Train-only global perturbed count mean baseline within `scripts/run_family_o_count_likelihood.py`.

### Why this preserves JEPA
No bridge weights are trained. The baseline uses train target count statistics only and excludes `batch_id`.

### Compute cost estimate
CPU Tier 1 seed-2 diagnostic; synthetic-only.

### Observed effect
Poisson NLL `63.2684`; program `0.4624`; logFC `0.7444`; top50 `0.6017`; direction `0.6854`; pseudobulk `0.8681`.

### Verdict
COUNT_AUDIT_BASELINE_COMPLETE.

## Change 226: seed2_poisson_train_only_count_mean_table

### Family
O

### Exact mechanism
Train-only condition count mean table keyed by `perturbation_id`, `cell_line_id`, `dose`, and `time`, evaluated with Poisson NLL.

### Why this preserves JEPA
No bridge weights are trained. Target means come from train split only; `batch_id` is excluded from the key/features.

### Compute cost estimate
CPU Tier 1 seed-2 diagnostic; synthetic-only.

### Observed effect
Poisson NLL `48.4387`; program `0.7433`; logFC `0.7562`; top50 `0.6392`; direction `0.7679`; pseudobulk `0.8815`; exact train-key coverage `1.0000`.

### Verdict
TIER1_POISSON_COUNT_TABLE_COMPLETE.

## Change 227: seed2_nb_train_only_count_mean_table

### Family
O

### Exact mechanism
Same train-only condition count mean table as Change 226 with train-only gene-wise NB dispersion calibration.

### Why this preserves JEPA
No bridge weights are trained. Dispersion is fit only on train target counts and remains positive through softplus/clamping.

### Compute cost estimate
CPU Tier 1 seed-2 diagnostic; synthetic-only.

### Observed effect
NB NLL `4.9933`; perturbation metrics match Change 226; learned dispersion median `1.6888`, mean `1.7681`.

### Verdict
TIER1_NB_COUNT_TABLE_COMPLETE.

## Change 228: seed2_poisson_mlp_no_batch_condition_source

### Family
O

### Exact mechanism
Small no-batch condition/source-feature MLP trained with Poisson NLL to predict clamped positive count means.

### Why this preserves JEPA
The protected bridge is untouched. Inputs exclude `batch_id`; training uses train target counts only.

### Compute cost estimate
CPU Tier 1 seed-2 full-batch MLP, 1200 steps; synthetic-only.

### Observed effect
Train Poisson NLL `37.8105`; held-out Poisson NLL `133.1362`; program `0.4855`; logFC `0.4293`; top50 `0.5433`.

### Verdict
TIER1_POISSON_COUNT_MLP_COMPLETE.

## Change 229: seed2_nb_mlp_no_batch_condition_source

### Family
O

### Exact mechanism
Small no-batch condition/source-feature MLP trained with NB NLL and learned positive gene-wise dispersion.

### Why this preserves JEPA
The protected bridge is untouched. Inputs exclude `batch_id`; training uses train target counts only.

### Compute cost estimate
CPU Tier 1 seed-2 full-batch MLP, 1400 steps; synthetic-only.

### Observed effect
Train NB NLL `4.9025`; held-out NB NLL `8.1260`; program `0.5770`; logFC `0.4057`; top50 `0.5342`; dispersion median `1.6111`.

### Verdict
TIER1_NB_COUNT_MLP_COMPLETE.
