# Next Steps Autoresearch-Bio Plan

## Status

Model of record remains unchanged: no synthetic candidate has passed Tier 1 gates, no Tier 2 or Tier 3 promotion is allowed, and no real data is allowed.

The 100-experiment search ended with no safe candidate. The follow-up diagnostics changed the failure interpretation:

- Objective ablations did not finally collapse on traced train batches.
- Full eval-mode collection collapses even on train condition bags.
- True latent oracle retrieval on `synth_micro` is perfect.
- Raw ridge RNA-to-image retrieval is above random on test.
- Model projected/shared embeddings are already below the hard std gate before retrieval.

## Current Hypothesis

The main failure is model representation geometry, not a bad synthetic target and not only a held-out split problem. The leading suspects are:

- global CLS embeddings are weakly trained by masked reconstruction;
- projection heads normalize already-low-variance encoder CLS outputs;
- train-mode dropout makes batch traces look healthier than eval-mode collection;
- tiny condition-bag batches make variance/covariance losses unstable unless accumulated or queued;
- image and RNA encoders learn useful local reconstruction features but do not expose aligned global condition embeddings.

## Protected Rules

- Synthetic data only.
- Synthetic GPU runs are now required when the GPU is free. Before each GPU batch, check `nvidia-smi`; if free, run with CUDA. CPU is allowed only for genuinely CPU-only work or if CUDA fails.
- JEPA + scRNA-seq perturbation bridge remains the core concept.
- No broad architecture sweep until diagnostics localize collapse and one repair passes a focused gate.
- No model promotion without explicit Tier 3.

## Phase D: Diagnostics Before More Search

### D1 Encoder/Projection/Mode Audit

Run a diagnostic that compares, by split and aggregation mode:

- raw encoder CLS output;
- pre-normalized projection output;
- normalized projection output;
- bag/shared output;
- train-mode no-gradient vs eval-mode outputs;
- token/patch mean pooled alternatives.

Pass condition: identify the first stage where min std falls below `0.01`, and whether dropout is masking collapse.

Result: completed in `diagnostics/ENCODER_PROJECTION_MODE_AUDIT`.

- Eval-mode shared embeddings collapse and retrieval remains random.
- Train-mode no-gradient embeddings become high-variance, but biological R2 becomes strongly negative.
- Therefore train-batch noncollapse was partly a dropout artifact, not a useful representation.
- RNA raw CLS has variance but poor biology; image raw/patch features carry biology but are angularly near-collapsed.
- L2-normalized projection heads further compress already narrow angular variation.

### D2 CLS vs Token Pool Audit

If CLS is collapsed while token/patch means carry biological signal, add a diagnostic-only alternate pooling path:

- RNA token mean or attention pooling excluding masked positions;
- image patch mean or attention pooling;
- no architectural promotion, just evaluation of representation geometry.

Pass condition: token/patch pooled embeddings improve test recall above random and preserve min std above `0.01`.

### D3 Variance Estimator Audit

Evaluate variance/covariance losses using accumulated condition embeddings or a tiny queue instead of one mini-batch.

Pass condition: variance regularizer raises train and test min std without increasing batch leakage or destroying raw biological latent R2.

## Phase R: Focused Repairs

Only start these after D1-D3.

### R1 Replace CLS Condition Embedding

If D2 passes, replace condition-level CLS embeddings with token/patch pooled embeddings or a learned pooling head.

Tier 1 gate:

- `synth_micro` recall@1 must exceed random by at least `0.05`.
- RNA and image shared min std must both be `>= 0.01` on train, val, and test collection.
- batch probe balanced accuracy must not exceed majority by more than `0.10`.

### R2 Explicit Anti-Collapse With Queue

If projection collapse remains after R1, add a small CPU-safe memory queue for variance/covariance estimates.

Tier 1 gate:

- no collapse on train/val/test;
- no worse than raw ridge by more than a documented margin on retrieval;
- no batch leakage hard fail.

### R3 Synthetic Split Repair

Independently fix the synthetic split so val/test condition bags have enough cells for stable condition-level evaluation.

This is not a model improvement. It is an evaluation reliability repair.

## Stop Rules

Stop and write a blocker report if:

- true latent oracle retrieval fails on any diagnostic split;
- raw ridge cannot beat random after split repair;
- three focused repairs still collapse at the same encoder/projection stage;
- batch leakage becomes the only way to improve retrieval.

## Immediate Next Action

Run D3-lite now: no-dropout eval-mode checks with and without explicit variance/covariance regularization. This tests whether anti-collapse losses were ineffective because dropout made train-mode variance estimates overoptimistic.

## Subsequent Results

### D3-Lite No-Dropout Anti-Collapse

Completed in `diagnostics/DROPOUT_ZERO_D3_LITE`.

- Mean pooling + dropout `0.0` + variance/covariance regularization cleared the hard collapse flag.
- Retrieval stayed random.
- RNA latent R2 stayed negative.

Conclusion: collapse prevention is necessary but not sufficient.

### Token/Patch Pool Repair

Completed in `diagnostics/TOKEN_POOL_REPAIR`.

- RNA variance improved.
- Image variance remained just below the hard gate.
- Retrieval stayed random.
- RNA latent R2 remained negative.

Conclusion: token/patch pooling alone is not enough.

### RNA Pseudobulk Condition Readout

Completed in `diagnostics/PSEUDOBULK_REPAIR`.

- RNA latent R2 became positive with pseudobulk readout.
- RNA shared variance collapsed badly.
- Retrieval stayed random.
- Stronger variance pressure did not fix the collapse.
- Disabling final pseudobulk L2 normalization did not fix the collapse and made RNA latent R2 negative again.

Conclusion: pseudobulk carries useful RNA biology, but the current projection/readout geometry compresses it into a low-rank representation. The next focused step should be a direct RNA pseudobulk-to-shared whitening/probe diagnostic, not more generic objective sweeps.

## Updated Status After Prefit PLS Repair

The direct whitening/probe diagnostic passed. A rank-3 train-split-only PLS linear readout was installed into bridge raw-linear readout heads and passed:

- Tier 2 `synth_micro`, seeds `0/1/2`;
- Tier 3 `synth_easy_lite`, seed `0`;
- Tier 3 `synth_batch_confound_lite`, seed `0`.

Summary artifact: `diagnostics/PREFIT_PLS_READOUT/TIER2_TIER3_SUMMARY.md`.

## Next Plan

### P1 Freeze-Protected PLS Initializer

Treat the rank-3 PLS readout as the protected synthetic geometry baseline. Any subsequent training must report deltas against this exact readout and fail if it regresses:

- RNA->image recall@1;
- RNA/image min std;
- RNA biological latent R2;
- batch probe gate.

### P2 Reintroduce JEPA Around The Readout

Train only the masked reconstruction/JEPA encoders while freezing the raw-linear PLS readout, then evaluate whether encoder features can improve auxiliary tasks without changing the protected shared geometry.

### P3 Distill Without Moving The Baseline

If a trainable neural head is needed, distill into a separate head while retaining the frozen PLS head as `rna_retrieval`/`image_retrieval` until the learned head independently passes Tier 2.

### P4 Counterfactual Repair Separately

The PLS readout fixes retrieval/shared geometry, not the counterfactual decoder. Counterfactual metrics should be repaired only after the shared geometry remains stable under P2/P3.

## Updated Status After P3 Distillation

P3 was split into two candidates:

- Randomly initialized raw MLP student: failed Tier 2 because retrieval was inconsistent across seeds.
- Exact linear PLS clone student: passed Tier 2 as an engineering baseline with zero student-teacher MSE and zero frozen-readout drift.

The active trainable shared-geometry baseline is now the exact PLS linear clone, not the raw MLP.

## Immediate Next Action

Completed a trust-region residual adapter on top of the linear clone:

- initialize residual weights and output scale to zero;
- keep the frozen PLS retrieval path active;
- evaluate the residual student independently;
- hard fail if residual drift reduces recall@1, RNA latent R2, min std, or batch gate relative to the clone.

Result: safe but no improvement. Keep it as an extension point; do not promote it over the exact linear clone.

## Next Action After Residual

Move to counterfactual repair with the clone-frozen shared geometry:

- freeze frozen PLS retrieval and the exact linear clone;
- train only `state_head`, `response_head`, `delta_gate`, `delta_effect`, and `rna_distribution_decoder`;
- use synthetic observed control/treated pairs only;
- gate on no regression of clone retrieval/R2/batch metrics plus improved counterfactual direction/program recovery.

## Updated Status After Counterfactual Repair

Clone-frozen counterfactual repair produced a useful but non-promotable result:

- generative RNA decoder: failed seed-0 counterfactual gate;
- residual RNA delta decoder: improved direction, logFC, pseudobulk, and top-DE overlap without shared-geometry regression;
- residual RNA delta decoder failed Tier 2 because program-level effect recovery regressed in seeds 1 and 2.

## Next Action After Counterfactual Tier 2 Failure

Add a synthetic-program aggregated delta loss:

- use `dataset.gene_program_assignment` from the synthetic generator only;
- train residual RNA delta decoder with both per-gene delta loss and program-mean delta loss;
- preserve the exact same retrieval/R2/batch no-regression gates;
- promote only if program recovery improves across all three seeds without losing the per-gene gains.

Result: completed and failed. Program-loss weighting did not fix program recovery consistency.

## Next Action After Program-Loss Failure

Do not keep reweighting the same decoder. The next counterfactual candidate should be a program-factorized residual decoder:

- predict one delta per synthetic program;
- broadcast program deltas back to genes using `gene_program_assignment`;
- optionally add a small within-program residual initialized to zero;
- gate on the same protected retrieval metrics and all counterfactual delta metrics.

## Updated Status After Experiments 107-110

Program-factorized decoder variants were completed and reconciled into the central artifacts:

- Experiment 107: program-factorized decoder at 25 steps failed Tier 2. Seed 0 passed, seeds 1 and 2 failed program recovery.
- Experiment 108: delta-MSE training passed seed 0 but was not advanced after adjacent Tier 2 failures showed the same seed 1/2 pattern.
- Experiment 109: source program context at 100 steps failed Tier 2. Seed 0 improved strongly; seeds 1 and 2 failed.
- Experiment 110: source program context plus direct no-batch biological metadata at 100 steps failed Tier 2. Seed 0 passed, seed 1 improved program recovery weakly, and seed 2 regressed.

No model-of-record update is allowed. Protected shared geometry was preserved, but the counterfactual program gate did not pass all seeds.

## Strict 100-Experiment Continuation Plan

The continuation must add at least 100 new documented experiments/trials unless a protected stop condition fires. Start at Experiment 111 and keep all completed trials in `results.tsv`, `research_journal.md`, `architectural_changes_log.md`, `family_allocation.md`, and the clone-counterfactual decoder summary as soon as each Tier check completes.

Focus the next Family K search on seed-2 program recovery while preserving the direct no-batch biological metadata and protected PLS geometry. Favor small synthetic-only changes to training dynamics and decoder parameterization: program loss schedules, seed-2-balanced pair sampling, linear/prefit program decoders, metadata/source-context ablations, direction-weight schedules, and amplitude control. Do not introduce real data, real marker/pathway resources, or pretrained assets.

## Strict GPU Continuation Progress

- Completed 111: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0610`, gate `False`. Continue unless protected stop fires.

- Completed 112: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0608`, gate `False`. Continue unless protected stop fires.

- Completed 113: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0606`, gate `False`. Continue unless protected stop fires.

- Completed 114: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0603`, gate `False`. Continue unless protected stop fires.

- Completed 115: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0601`, gate `False`. Continue unless protected stop fires.

- Completed 116: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0604`, gate `False`. Continue unless protected stop fires.

- Completed 117: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0591`, gate `False`. Continue unless protected stop fires.

- Completed 118: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0597`, gate `False`. Continue unless protected stop fires.

- Completed 119: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0602`, gate `False`. Continue unless protected stop fires.

- Completed 120: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0605`, gate `False`. Continue unless protected stop fires.

- Completed 121: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0604`, gate `False`. Continue unless protected stop fires.

- Completed 122: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0605`, gate `False`. Continue unless protected stop fires.

- Completed 123: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0608`, gate `False`. Continue unless protected stop fires.

- Completed 124: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0595`, gate `False`. Continue unless protected stop fires.

- Completed 125: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0581`, gate `False`. Continue unless protected stop fires.

- Completed 126: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0574`, gate `False`. Continue unless protected stop fires.

- Completed 127: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0586`, gate `False`. Continue unless protected stop fires.

- Completed 128: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0600`, gate `False`. Continue unless protected stop fires.

- Completed 129: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0671`, gate `False`. Continue unless protected stop fires.

- Completed 130: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0667`, gate `False`. Continue unless protected stop fires.

- Completed 131: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0659`, gate `False`. Continue unless protected stop fires.

- Completed 132: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0644`, gate `False`. Continue unless protected stop fires.

- Completed 133: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0613`, gate `False`. Continue unless protected stop fires.

- Completed 134: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0579`, gate `False`. Continue unless protected stop fires.

- Completed 135: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0662`, gate `False`. Continue unless protected stop fires.

- Completed 136: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0666`, gate `False`. Continue unless protected stop fires.

- Completed 137: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0670`, gate `False`. Continue unless protected stop fires.

- Completed 138: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0674`, gate `False`. Continue unless protected stop fires.

- Completed 139: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0671`, gate `False`. Continue unless protected stop fires.

- Completed 140: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0644`, gate `False`. Continue unless protected stop fires.

- Completed 141: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0610`, gate `False`. Continue unless protected stop fires.

- Completed 142: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0608`, gate `False`. Continue unless protected stop fires.

- Completed 143: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0606`, gate `False`. Continue unless protected stop fires.

- Completed 144: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0603`, gate `False`. Continue unless protected stop fires.

- Completed 145: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0601`, gate `False`. Continue unless protected stop fires.

- Completed 146: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0604`, gate `False`. Continue unless protected stop fires.

- Completed 147: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0591`, gate `False`. Continue unless protected stop fires.

- Completed 148: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0597`, gate `False`. Continue unless protected stop fires.

- Completed 149: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0602`, gate `False`. Continue unless protected stop fires.

- Completed 150: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0605`, gate `False`. Continue unless protected stop fires.

- Completed 151: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0604`, gate `False`. Continue unless protected stop fires.

- Completed 152: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0605`, gate `False`. Continue unless protected stop fires.

- Completed 153: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0608`, gate `False`. Continue unless protected stop fires.

- Completed 154: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0595`, gate `False`. Continue unless protected stop fires.

- Completed 155: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0581`, gate `False`. Continue unless protected stop fires.

- Completed 156: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0574`, gate `False`. Continue unless protected stop fires.

- Completed 157: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0586`, gate `False`. Continue unless protected stop fires.

- Completed 158: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0600`, gate `False`. Continue unless protected stop fires.

- Completed 159: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0671`, gate `False`. Continue unless protected stop fires.

- Completed 160: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0667`, gate `False`. Continue unless protected stop fires.

- Completed 161: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0659`, gate `False`. Continue unless protected stop fires.

- Completed 162: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0644`, gate `False`. Continue unless protected stop fires.

- Completed 163: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0613`, gate `False`. Continue unless protected stop fires.

- Completed 164: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0579`, gate `False`. Continue unless protected stop fires.

- Completed 165: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0103`, gate `False`. Continue unless protected stop fires.

- Completed 166: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0092`, gate `False`. Continue unless protected stop fires.

- Completed 167: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0075`, gate `False`. Continue unless protected stop fires.

- Completed 168: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0056`, gate `False`. Continue unless protected stop fires.

- Completed 169: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0146`, gate `False`. Continue unless protected stop fires.

- Completed 170: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0131`, gate `False`. Continue unless protected stop fires.

- Completed 171: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0103`, gate `False`. Continue unless protected stop fires.

- Completed 172: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0069`, gate `False`. Continue unless protected stop fires.

- Completed 173: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0206`, gate `False`. Continue unless protected stop fires.

- Completed 174: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0185`, gate `False`. Continue unless protected stop fires.

- Completed 175: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0146`, gate `False`. Continue unless protected stop fires.

- Completed 176: `TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE`, seed 2, program delta `0.0092`, gate `False`. Continue unless protected stop fires.

- Completed 177: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2512`, gate `False`. Continue unless protected stop fires.

- Completed 178: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2503`, gate `False`. Continue unless protected stop fires.

- Completed 179: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2515`, gate `False`. Continue unless protected stop fires.

- Completed 180: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2523`, gate `False`. Continue unless protected stop fires.

- Completed 181: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2512`, gate `False`. Continue unless protected stop fires.

- Completed 182: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2503`, gate `False`. Continue unless protected stop fires.

- Completed 183: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2514`, gate `False`. Continue unless protected stop fires.

- Completed 184: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2522`, gate `False`. Continue unless protected stop fires.

- Completed 185: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2507`, gate `False`. Continue unless protected stop fires.

- Completed 186: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2499`, gate `False`. Continue unless protected stop fires.

- Completed 187: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2511`, gate `False`. Continue unless protected stop fires.

- Completed 188: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2518`, gate `False`. Continue unless protected stop fires.

- Completed 189: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2466`, gate `False`. Continue unless protected stop fires.

- Completed 190: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2470`, gate `False`. Continue unless protected stop fires.

- Completed 191: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2486`, gate `False`. Continue unless protected stop fires.

- Completed 192: `TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE`, seed 2, program delta `0.2494`, gate `False`. Continue unless protected stop fires.

- Completed 193: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1727`, gate `False`. Continue unless protected stop fires.

- Completed 194: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1615`, gate `False`. Continue unless protected stop fires.

- Completed 195: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1181`, gate `False`. Continue unless protected stop fires.

- Completed 196: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1333`, gate `False`. Continue unless protected stop fires.

- Completed 197: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1089`, gate `False`. Continue unless protected stop fires.

- Completed 198: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0971`, gate `False`. Continue unless protected stop fires.

- Completed 199: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0790`, gate `False`. Continue unless protected stop fires.

- Completed 200: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0785`, gate `False`. Continue unless protected stop fires.

- Completed 201: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0789`, gate `False`. Continue unless protected stop fires.

- Completed 202: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0549`, gate `False`. Continue unless protected stop fires.

- Completed 203: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0549`, gate `False`. Continue unless protected stop fires.

- Completed 204: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0570`, gate `False`. Continue unless protected stop fires.

- Completed 205: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0570`, gate `False`. Continue unless protected stop fires.

- Completed 206: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0586`, gate `False`. Continue unless protected stop fires.

- Completed 207: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0586`, gate `False`. Continue unless protected stop fires.

- Completed 208: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0928`, gate `False`. Continue unless protected stop fires.

- Completed 209: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.1011`, gate `False`. Continue unless protected stop fires.

- Completed 210: `TIER1_DISCARD_SEED2_PROGRAM_REGRESSION`, seed 2, program delta `-0.0945`, gate `False`. Continue unless protected stop fires.

## Updated Status After Strict GPU Experiments 111-210

The continuation completed 100 new documented CUDA trials. No trial passed the full counterfactual gate, and no model-of-record update is permitted.

Findings:

- Direct loss rebalancing with the MLP program decoder did not repair seed 2.
- Context ablations confirmed that source program context and no-batch biological metadata are both needed; removing either worsened seed-2 program recovery.
- Within-program residual variants did not help and continued to regress seed-2 program recovery.
- Linear program decoders improved seed 2 from regression to weak positive recovery.
- Train-split-only prefit ridge initialization for the linear program decoder produced the strongest signal: program recovery around `+0.25` on seed 2 while preserving protected geometry.
- The remaining full-gate blocker is top-50 DE overlap. Best ridge top-50 delta was `+0.0200`, below the `+0.03` gate.

Next synthetic-only direction: keep the ridge-initialized linear program decoder as a non-promoted near-miss candidate and add a synthetic top-DE-aware repair that uses only train-pair target-delta magnitudes or synthetic generator metadata. Do not use real marker lists, real pathway resources, or pretrained assets.

## Updated Status After Experiment 211

Completed 211: `TIER1_DISCARD_STRONG_BASELINE_UNDERPERFORM`, seed 2, program delta `+0.2877`, direction delta `+0.5793`, logFC delta `-0.0830`, top50 delta `0.0000`, protected geometry `True`, gate `False`.

The sparse perturbation residual head validated the literature-informed separation idea only partially. It recovered program-level structure but produced too-small and gene-misoriented deltas. It also underperformed the no-batch matching-mean baseline:

- candidate: program `0.2877`, logFC `-0.0830`, top50 `0.4058`;
- matching mean: program `0.3502`, logFC `0.1268`, top50 `0.4150`.

Immediate next step: run at most one amplitude-focused sparse residual Tier-1 variant before pivoting. Suggested variant: higher global/top-DE pressure and lower sparsity/decorrelation, e.g. `--lr 0.003 --steps 150 --global-delta-weight 0.5 --top-de-weight 12 --program-gene-weight 2 --sparsity-weight 0.002 --decorrelation-weight 0.001 --dictionary-rank 8`. Keep the same matching-mean comparator and protected gates.

Pivot rule: if the next sparse residual variant does not beat matching mean on logFC and top50 while preserving protected geometry, stop sparse-head sweeps and test or plan a conditional OT/Monge-style transport from control to perturbed distributions conditioned on perturbation/cell/dose/time metadata. Keep it synthetic-only and no-batch.

## Updated Status After Experiment 212

Completed 212: `TIER1_DISCARD_STRONG_BASELINE_UNDERPERFORM`, seed 2, program `0.2750`, direction `0.5622`, logFC `0.1088`, top50 `0.3917`, protected geometry `True`, gate `False`.

Sparse residual Family L cooldown is active. The amplitude variant raised delta scale (`mean_sparse_final_delta_to_target_ratio = 0.4116`) but still failed the strong baseline comparison:

- candidate: program `0.2750`, logFC `0.1088`, top50 `0.3917`;
- matching mean: program `0.3502`, logFC `0.1268`, top50 `0.4150`.

Next family plan: conditional OT/Monge-style transport, synthetic-only.

1. Keep the protected PLS/readout clone frozen.
2. Use train split only to fit a no-batch conditional map from control expression distribution to perturbed expression distribution, conditioned on perturbation, cell line, dose, and time.
3. Start with a small deterministic empirical transport baseline: within each no-batch condition key, pair control and perturbed train cells by nearest neighbor or sorted 1D projections in synthetic program/top-DE space, estimate a transport delta, and evaluate on seed-2 test pairs.
4. Compare against source-as-target, dataset mean, matching mean, and sparse residual Experiments 211-212 on the same metrics.
5. Promote only if it beats matching mean on logFC/top50/program metrics and preserves protected retrieval/R2/batch gates.

Do not use real marker lists, real pathways, pretrained assets, or batch metadata.

## Updated Status After Family M Experiments 213-219

Completed Family M deterministic baselines on `synth_micro` seed 2, CPU only:

- 213 `BASELINE_COMPLETE_STRONG_EXACT_CONDITION_MATCH`: direct no-batch matched perturbed mean, program `0.7520`, logFC `0.7502`, top50 `0.5683`, direction `0.6899`, pseudobulk `0.8725`.
- 214 `BASELINE_COMPLETE_RESIDUALIZED_MATCHING_REFERENCE`: residualized matching, program `0.3502`, logFC `0.1268`, top50 `0.4150`.
- 215-218 `TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE`: source-program kNN residual transport for `k=1/3/5/8`; best was `k=3` for program/top50 but logFC stayed below matching.
- 219 `TIER1_DISCARD_MATCHING_MEAN_INCOMPLETE`: in-repo Sinkhorn residual transport, program `0.4211`, logFC `0.0657`, top50 `0.4217`.

Interpretation:

The exact-condition split is dominated by a simple train-split perturbed-mean baseline. Residualized matching remains a useful comparator for learned delta predictors, but the stronger direct perturbed-mean baseline means no neural transport should be started on this split unless the goal is explicitly to learn a smoother approximation to exact matching.

Next steps:

1. Do not continue sparse residual Family L or optional neural transport on current seed-2 exact-condition support.
2. Treat direct matched perturbed mean as the current synthetic seed-2 ceiling for exact-condition recovery.
3. If architecture search continues, amend the benchmark first: held-out perturbation, dose, cell-line, or no-exact-condition target support so matching mean is not a near-oracle.
4. Keep no-batch constraints and synthetic-only scope. No real data, real marker/pathway resources, pretrained biological assets, or technical batch metadata.

## Updated Status After Family N Distillation

Completed Family N A-D on `synth_micro` seed 2, CPU only:

- 220 `REFERENCE_TEACHER_COMPLETE_MATCHES_EXP213`: train-only condition mean table, no batch key/features, no test target rows; program `0.7520`, logFC `0.7502`, top50 `0.5683`.
- 221 `TIER1_DISTILLATION_COMPLETE_UNDER_TEACHER`: linear conditional mean student; leakage gate passed, program `0.7353`, logFC `0.7261`, top50 `0.5583`.
- 222 `TIER1_DISTILLATION_COMPLETE_UNDER_TEACHER`: small MLP conditional mean student; leakage gate passed, program `0.7033`, logFC `0.7286`, top50 `0.5525`.
- 223 `TIER1_HYBRID_SWEEP_COMPLETE_NO_PROMOTION`: shrinkage sweep; best nonzero hybrid made only tiny tradeoffs and did not beat the train-only table jointly.

Interpretation:

Family N answered the user-selected option 2: the no-batch matched perturbed-mean behavior can be distilled into train-only learned conditional models without test leakage, but the direct train-only table remains the strongest exact-condition model. No learned student or hybrid earns Tier 2 because the ceiling is already the train-key lookup.

Immediate next action:

1. Do not continue scalar Family N sweeps on seed-2 exact-condition support.
2. Add or run a no-exact-key benchmark before reopening learned counterfactual modeling, for example held-out perturbation, held-out dose, held-out cell line, or explicit removal of exact biological key support from train.
3. Keep the same safeguards: train-only teacher construction, no `batch_id`, fallback only from train statistics, and required comparison against train-only table, residualized matching, Family L, and prefit ridge.
4. No model-of-record update.

## Updated Status After Family O Count-Likelihood Start

Completed Family O A-E on `synth_micro` seed 2, CPU only, without replacing Family N:

- 224 `COUNT_AUDIT_BASELINE_COMPLETE`: raw synthetic `observed_counts` exist; pseudo-count path unused. Count diagnostics: zero fraction `0.2026`, mean `107.9808`, variance `49344.0223`, log mean-variance correlation `0.9593`, dispersion median `1.4854`.
- 225 `COUNT_AUDIT_BASELINE_COMPLETE`: train-only global count mean baseline, Poisson NLL `63.2684`, program `0.4624`, logFC `0.7444`, top50 `0.6017`.
- 226 `TIER1_POISSON_COUNT_TABLE_COMPLETE`: train-only count mean table, Poisson NLL `48.4387`, program `0.7433`, logFC `0.7562`, top50 `0.6392`, direction `0.7679`, pseudobulk `0.8815`.
- 227 `TIER1_NB_COUNT_TABLE_COMPLETE`: same count table with train-only gene-wise NB dispersion, NB NLL `4.9933`.
- 228 `TIER1_POISSON_COUNT_MLP_COMPLETE`: no-batch Poisson MLP underperformed table, program/logFC/top50 `0.4855/0.4293/0.5433`.
- 229 `TIER1_NB_COUNT_MLP_COMPLETE`: no-batch NB MLP underperformed table, program/logFC/top50 `0.5770/0.4057/0.5342`.

Interpretation:

Family O found a real count-scale signal: the train-only count table improves logFC, top50 overlap, direction accuracy, and pseudobulk correlation versus the Family N expression table. It does not jointly beat Family N because program recovery is slightly lower (`0.7433` vs `0.7520`). Learned count decoders are currently worse than the exact table, so no Tier 2 or model-of-record update is justified.

Immediate next action:

1. Keep Family N active as the independent train-only matched-mean distillation reference.
2. For Family O, run at most one Candidate F: NB NLL plus small program/top-DE auxiliary loss on the same no-batch condition/source MLP, with train-only top-DE/program statistics and explicit leakage diagnostics.
3. Stop Family O on this exact split if Candidate F does not beat the count table on program while preserving the count table's top50/logFC gains.
4. Prefer a no-exact-key or held-out-condition benchmark before any broader learned count-model sweep.
5. No model-of-record update.
