# Insight Brief 219: Family M Transport Baselines

## Scope

Experiments 213-219 started Family M: deterministic no-batch matching and conditional transport for `synth_micro` seed 2. All runs were synthetic-only, CPU-only, and excluded `batch_id` from matching/features.

## Main Result

Direct no-batch matched perturbed mean is the new exact-condition ceiling:

- program recovery `0.7520`;
- logFC correlation `0.7502`;
- top50 DE overlap `0.5683`;
- direction accuracy `0.6899`;
- pseudobulk correlation `0.8725`.

The previously cited residualized matching mean is still reproduced:

- program recovery `0.3502`;
- logFC correlation `0.1268`;
- top50 DE overlap `0.4150`.

## Transport Result

Nonparametric residual transport did not clear the baseline discipline test.

- kNN source-program residual transport best variant: `k=3`, program `0.5293`, top50 `0.4308`, but logFC only `0.0623`.
- Sinkhorn source-program residual transport: program `0.4211`, top50 `0.4217`, logFC `0.0657`.

## Interpretation

CellOT, CINEMA-OT, scDRP, and Conditional Monge Gap motivate conditional transport, but this exact-condition synthetic split is dominated by empirical perturbed means. Systema-style baseline discipline says not to claim neural progress unless direct matching and residualized matching are beaten.

## Decision

Do not launch optional neural transport on this split. The next useful benchmark change is to remove exact-condition target support, for example held-out perturbations, doses, cell lines, or condition keys, while preserving the no-batch constraint.
