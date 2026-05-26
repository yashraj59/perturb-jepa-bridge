# Insight Brief 223: Train-Only Distillation Confirms Matching-Dominated Split

## Scope

Family N tested train-only matched-mean distillation on `synth_micro` seed 2 after Family M established the no-batch matched perturbed mean as the strongest exact-condition baseline.

## Main Result

The train-only conditional mean table exactly reproduced Experiment 213 without test target leakage:

- biological key: `perturbation_id`, `cell_line_id`, `dose`, `time`;
- `batch_id` excluded from keys and features;
- teacher target rows from test: `0`;
- exact train-key coverage on test: `1.0000`;
- program `0.7520`, logFC `0.7502`, top50 `0.5683`, direction `0.6899`, pseudobulk `0.8725`.

## Learned Students

The linear and MLP students safely distilled the teacher but did not beat it.

- Linear: program `0.7353`, logFC `0.7261`, top50 `0.5583`.
- MLP: program `0.7033`, logFC `0.7286`, top50 `0.5525`.
- Best nonzero shrinkage hybrid: program `0.7526`, logFC `0.7500`, top50 `0.5658`.

All passed leakage diagnostics and preserved protected geometry because no bridge weights were trained.

## Interpretation

The blocker is no longer a missing neural architecture on this split. The seed-2 exact-condition task is dominated by train-split perturbed means. Learned conditional mean models are useful as leakage-safe distillations, but they mostly reparameterize the table and introduce small smoothing errors.

## Decision

No Family N promotion. Do not run more exact-condition scalar/model sweeps. Reopen learned counterfactual modeling only after changing the benchmark to remove exact biological key support or to test held-out perturbation/dose/cell-line generalization under the same train-only, no-batch safeguards.
