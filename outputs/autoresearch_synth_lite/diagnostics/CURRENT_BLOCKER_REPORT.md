# Current Blocker Report

## Status

Resolved for the low-compute synthetic loop by a train-split-only closed-form rank-3 PLS linear readout.

The neural JEPA bridge still should not be considered fixed by SGD training. The resolved point is narrower and important: the condition-level RNA/image shared geometry can be made non-collapsed, retrieval-positive, biologically informative, and batch-safe without real data or GPU-heavy training.

## Previous Blocker

The synthetic target was learnable, but the current JEPA bridge could not produce a safe RNA-image shared condition embedding under low-compute synthetic checks.

Evidence:

- true latent oracle retrieval on `synth_micro` is perfect;
- raw ridge RNA-to-image retrieval beats random;
- image shared embeddings repeatedly recover biological latent structure;
- RNA shared embeddings either collapse, have negative biological latent R2, or both;
- repairs that remove collapse do not improve retrieval;
- repairs that add RNA biological signal collapse the RNA shared representation.

## What Failed

- 100-experiment broad search: no promoted candidate.
- objective ablation: train-batch traces were misleading because eval-mode collection still collapses.
- no-dropout variance/covariance: can clear hard collapse but retrieval remains random.
- token/patch pooling: no retrieval improvement.
- RNA pseudobulk readout: positive RNA latent R2 but severe RNA collapse.
- stronger pseudobulk variance and unnormalized pseudobulk projection: no pass.

## Safe State

- No real data used.
- No GPU-heavy work used.
- No random neural student promoted.
- Full test suite passes: `137 passed, 29 warnings`.

## Breakthrough

The direct RNA pseudobulk-to-shared whitening/probe diagnostic passed, and the resulting closed-form PLS readout was installed as bridge raw-linear readout heads.

The rank-3 prefit PLS candidate passed:

- Tier 2 `synth_micro`, seeds `0/1/2`;
- Tier 3 `synth_easy_lite`, seed `0`;
- Tier 3 `synth_batch_confound_lite`, seed `0`.

Summary: `outputs/autoresearch_synth_lite/diagnostics/PREFIT_PLS_READOUT/TIER2_TIER3_SUMMARY.md`.

## Next Risk

The next risk is counterfactual program consistency. The exact linear PLS clone is now the trainable handoff baseline, and the residual adapter is safe but not better. Residual RNA counterfactual decoding improves per-gene delta metrics without shared-geometry regression, but it fails Tier 2 because program-level effect recovery regresses in two seeds.

Program-aggregated loss weighting was tested and did not fix the issue. Next repair should change parameterization to a program-factorized residual decoder. Real marker/pathway data remains forbidden.

## Updated Blocker After Experiments 107-110

The program-factorized family narrowed but did not resolve the blocker.

- Program-factorized decoding at 25 steps preserved protected geometry and passed seed 0, but seeds 1 and 2 failed program recovery.
- Delta-MSE reproduced the seed-0 pass only.
- Source program context at 100 steps improved seed 0 strongly but failed seeds 1 and 2.
- Direct no-batch biological metadata plus source program context at 100 steps produced the strongest mean program recovery so far, but still failed Tier 2: seed 0 program delta `+0.3994`, seed 1 `+0.0422`, seed 2 `-0.0663`.

Current blocker: seed-2 counterfactual program-level effect recovery, not shared-geometry safety. Protected retrieval/R2/batch metrics remain preserved, so the next continuation should target seed-2-specific synthetic program recovery under the same no-real-data and no-batch-metadata constraints.

## Updated Blocker After Experiment 211

The first sparse perturbation-specific residual head changed the blocker interpretation.

- Protected PLS/linear clone geometry is still safe: frozen readout drift `0.00000000`, protected geometry `True`.
- A separate perturbation-only residual channel can recover seed-2 program structure: program recovery improved from `0.0000` to `0.2877`.
- The learned sparse residual does not yet recover the gene-level effect amplitude: logFC correlation was `-0.0830`, top50 overlap stayed flat at `0.4058`, and mean predicted delta/target delta ratio was only `0.0149`.
- The no-batch matching-mean baseline is strong on the same seed-2 task: program recovery `0.3502`, logFC `0.1268`, top50 overlap `0.4150`.

Current blocker: seed-2 top-DE/logFC amplitude recovery that beats matching mean while preserving the protected PLS geometry. Sparse residual heads are not promoted unless they beat that baseline, not merely source-as-target.

## Updated Blocker After Experiment 212

The amplitude-focused sparse residual follow-up did not clear the blocker.

- Protected geometry remained safe.
- Delta scale increased substantially: mean sparse delta/target ratio `0.4116`.
- LogFC improved to `0.1088`, but still trailed matching mean `0.1268`.
- Top50 overlap regressed to `0.3917`, below source-as-target/matching mean levels.
- Program recovery was `0.2750`, below matching mean `0.3502`.

Current blocker: not merely sparse residual capacity or amplitude. The learned residual head is not ranking the synthetic top-DE genes as well as a simple no-batch matching-mean transport. Sparse residual Family L should cool down; the next synthetic-only family should treat control-to-perturbed distribution matching directly, using conditional OT/Monge-style transport conditioned on perturbation, cell line, dose, and time while excluding batch metadata.

## Updated Blocker After Strict GPU Experiments 111-210

The strict GPU continuation completed 100 new synthetic trials, all on CUDA after a GPU-free check. No protected geometry stop fired, and no model-of-record update is allowed.

The blocker changed:

- MLP program decoders with source program context and direct no-batch metadata still regressed seed-2 program recovery across the loss grid.
- A shallow linear program decoder without ridge initialization produced weak positive seed-2 program recovery, but stayed below gate.
- Train-split-only prefit ridge initialization for the linear program decoder consistently produced strong seed-2 program recovery around `+0.25`.
- Ridge variants still failed the full gate because top-50 DE overlap remained below the required `+0.03` improvement. The best top-50 delta was `+0.0200` in Experiment 192. Some ridge variants also missed logFC by a small margin, but top-50 overlap is the consistent blocker.

Current blocker: converting ridge-initialized program-level recovery into top-DE overlap without real marker/pathway resources, technical batch metadata, or loss terms that abandon the JEPA/scRNA perturbation bridge identity.

## Updated Blocker After Family M Experiments 213-219

Family M clarified that the seed-2 split has a very strong exact-condition matching ceiling when train/test share the same biological condition keys.

- Direct no-batch matched perturbed mean, using only `perturbation_id`, `cell_line_id`, `dose`, and `time` and explicitly excluding `batch_id`, achieved program `0.7520`, logFC `0.7502`, top50 `0.5683`, direction `0.6899`, and pseudobulk `0.8725`.
- Residualized matching reproduced the earlier matching-mean reference: program `0.3502`, logFC `0.1268`, top50 `0.4150`.
- kNN residual transport in source program space improved program recovery over residualized matching for `k=1/3/5`, with best top50 at `k=3` (`0.4308`), but all kNN variants failed the joint comparison because logFC remained weak.
- In-repo Sinkhorn residual transport was smooth but still incomplete: program `0.4211`, logFC `0.0657`, top50 `0.4217`.
- Protected retrieval/R2/batch metrics were unchanged because these are deterministic baselines with no bridge parameter updates.

Current blocker: the exact-condition train perturbed mean is a strong baseline/ceiling, and residual transport does not explain the remaining gap. Optional neural transport is not justified on this split. The next meaningful counterfactual question should remove exact-condition target support, for example by evaluating held-out biological conditions/perturbations/doses, or explicitly frame this as a matching-baseline-dominated split rather than a learned counterfactual modeling benchmark.

## Updated Blocker After Family N Experiments 220-223

Family N confirmed the leakage-safe interpretation of the matched-mean ceiling and tested whether a learnable conditional mean can distill it.

- The train-only condition mean table used only `perturbation_id`, `cell_line_id`, `dose`, and `time`; `batch_id` was excluded from keys/features, and teacher construction used `0` test target rows.
- Exact train-key coverage on the seed-2 test pairs was `1.0000`, so the train-only table exactly reproduced Experiment 213: program `0.7520`, logFC `0.7502`, top50 `0.5683`, direction `0.6899`, pseudobulk `0.8725`.
- The linear student passed leakage diagnostics and approximated the teacher well, but remained below it on seed-2 test: program `0.7353`, logFC `0.7261`, top50 `0.5583`.
- The MLP student also passed leakage diagnostics but underfit relative to the linear student/table: program `0.7033`, logFC `0.7286`, top50 `0.5525`.
- Shrinkage hybrids produced only small tradeoffs. The best nonzero linear hybrid slightly increased program recovery to `0.7526`, but logFC/top50/pseudobulk did not beat the train-only table jointly.

Current blocker: this exact-condition synthetic split is now clearly matching-baseline dominated. A learned model can safely distill the table, but it does not improve the benchmark unless exact train target support is removed or made incomplete. Do not promote Family N; the next meaningful step is a held-out/no-exact-biological-key evaluation that keeps the same no-batch and train-only safeguards.

## Updated Blocker After Family O Experiments 224-229

Family O started a separate synthetic-only count-likelihood path without replacing Family N.

- The synthetic-lite generator does contain raw count-like RNA values: `observed_counts` from gamma-Poisson sampling with dropout. The pseudo-count fallback was not used.
- Count diagnostics on `synth_micro` seed 2: zero fraction `0.2026`, mean `107.9808`, variance `49344.0223`, log mean-variance correlation `0.9593`, MoM dispersion median `1.4854`.
- The train-only count mean table with Poisson/NB scoring improved top-DE and effect-amplitude metrics versus Family N's expression table: logFC `0.7562` vs `0.7502`, top50 `0.6392` vs `0.5683`, direction `0.7679` vs `0.6899`, pseudobulk `0.8815` vs `0.8725`.
- The same count table slightly regressed program recovery: `0.7433` vs Family N `0.7520`.
- Learned count decoders did not close the gap. The Poisson MLP reached program/logFC/top50 `0.4855/0.4293/0.5433`; the NB MLP reached `0.5770/0.4057/0.5342`, both below the train-only count table despite leakage gates passing.
- Protected geometry is unchanged because Family O did not train bridge weights.

Current blocker: count likelihood improves count-scale calibration and top-DE recovery only when paired with the exact train-only condition table. Learned count decoders are not yet competitive, and Family N remains the independent expression-space train-only matched-mean reference. The next Family O step, if any, should be one targeted hybrid NB + program/top-DE auxiliary trial or a no-exact-key count benchmark; do not run a broad count-MLP sweep on the current exact-condition split.
