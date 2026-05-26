# perturb_jepa Autoresearch: Compositional Generalization on Norman 2019

## Project Context

The synthetic-only `autoresearch/synthetic-lite-v1` loop closed without a neural Tier 3 win. The exact-condition synthetic split is matching-baseline dominated and cannot test the mechanisms this codebase was built for. This phase moves the same architecture skeleton onto Norman et al. 2019 K562 single + double CRISPRi perturbation data, where compositional generalization is a real, measurable property and where published baselines exist.

This is a focused two-week sprint, not an open-ended search.

## Model Of Record

- Active baseline: published GEARS numbers on the canonical Norman 2019 held-out-combo split.
- Carried-over reference: train-only condition-mean table (Family N pattern from synthetic loop) recomputed on the Norman split.
- Rebasing rule: only a Tier 3 pass that beats GEARS on the no-exact subset by margin > seed std can rebase. No neural candidate is the model of record by default.

## Role

You are an autonomous research agent running a bounded, hypothesis-driven architectural search on real perturbation data. You must preserve the model of record, document every experiment, evaluate on exact and no-exact subsets separately, and stop when stop conditions fire.

## Setup

1. Branch: create `autoresearch/norman-v1` from current main.
2. Read fully before any code change:
   - `outputs/autoresearch_synth_lite/model_of_record.md`
   - `outputs/autoresearch_synth_lite/diagnostics/FAMILY_N_DISTILLATION/REPORT.md`
   - `perturb_jepa/models/bridge.py`
   - `perturb_jepa/models/perturbation_encoder.py`
   - `scripts/run_family_n_distillation.py`
3. Add data loader: `perturb_jepa/data/norman2019.py`
   - Provenance: log Norman 2019 source URL, version, license, gene-symbol resolution map, and ortholog mapping (if any) in `external_resources.md`.
   - Output a `NormanDataset` object exposing per-condition pseudobulk delta logFC, raw counts, single-vs-double label, perturbation IDs as a tuple, and a canonical GEARS-compatible held-out-combo split.
4. Initialize logs in `outputs/autoresearch_norman_v1/`:
   `results.tsv`, `research_journal.md`, `architectural_changes_log.md`, `family_allocation.md`, `BASELINE_REGISTRY.md`, `papers_consulted.md`, `external_resources.md`, `identity_violations_considered.md`.
5. Verify split fidelity: held-out combos must not appear in train under any aliasing of perturbation order (A+B vs B+A). Add a unit test that fails loudly on leakage.

## Step 0 Baselines (Stage A)

Before any architectural experiment, register baselines for the Norman split:

| baseline | source | metric | required |
|---|---|---|---|
| Global train mean | recompute | pseudobulk Pearson delta, top-20 DE overlap, MSE | yes |
| Single-perturbation additive | recompute: delta(A+B) := delta(A) + delta(B) from train singles | same | yes |
| Family N condition-mean table | port `scripts/run_family_n_distillation.py` to Norman | same | yes |
| Closed-form PLS readout | port `perturb_jepa/training/prefit_readout.py` to Norman | same | yes |
| GEARS published numbers | Roohani et al. 2024, exact split | same | yes |
| CPA published numbers | Lotfollahi et al. 2023, exact split | same | yes |

Splits to report separately:
- `exact_train_combo`: held-out combos whose component singles are both in train (the actual hard case)
- `unseen_single`: held-out perturbations not seen as singles in train (sanity check, smaller)

Single-perturbation additive baseline is the most important baseline here. If your model cannot beat it, no mechanism is doing real work — it is the dumbest possible compositional rule.

Write `BASELINE_REGISTRY.md` with provenance, mean/std, per-seed values, source script, and known caveats. Resolve any conflicts by reading raw per-seed files, not summary markdown.

Stop sign: do not start Family A or B until Step 0 is complete and reviewed.

## Identity

### Keep
- Bridge architecture skeleton (`PerturbJEPABridge`).
- Protected closed-form PLS readout install path.
- Family N condition-mean table reference.
- Norman 2019 GEARS-compatible split definition.
- Evaluation scripts (write once, do not touch during search).

### Can Modify
- Perturbation encoder (featurization replacing one-hot lookup).
- Delta predictor and its parameterization.
- Counterfactual decoder structure (program-factorized path is already optional).
- Loss weights, schedules, contribution caps.

### Cannot Modify
- Norman split definition.
- Evaluator code.
- Gene set / DE definitions.
- Published baseline numbers.

Any change to a "Cannot Modify" item requires an entry in `identity_violations_considered.md` and explicit user approval.

## Search Targets

- Primary: pseudobulk Pearson delta on `exact_train_combo` subset, mean over seeds.
- Secondary: top-20 DE overlap on `exact_train_combo`; MSE on `exact_train_combo`.
- Protected: no regression on `unseen_single` vs Family N baseline; no regression on direction accuracy (signed DE agreement) by more than 5%.
- Catastrophic fail: prediction collapses to mean on `exact_train_combo` (variance over perturbations < 10% of train target variance).

## Architectural Families

### Family A: Featurized Perturbation + Low-Rank Operator Delta

**Motivation.** The current `PerturbationEncoder` uses `nn.Embedding(num_perturbations, dim)`. On `unseen_single`, an unseen perturbation has no embedding at all. The current `delta_effect = nn.Linear(perturbation_dim, shared_dim)` is rank-1 with no state interaction.

**Hypothesis.** Featurized perturbation embeddings make unseen perturbations representable. A small low-rank operator acting on the program-factorized state lets perturbation effects depend on cell state in a structurally constrained way that the lookup table cannot match.

**Suggested experiments (in order):**

1. Replace `nn.Embedding(num_perturbations, dim)` with `f(perturbation_features) → dim`, where features are the target gene's mean log-expression vector across train cells. Smallest change. Initialize so that on a seen perturbation it reproduces the previous embedding to within tolerance.
2. Add ESM2 or Geneformer embedding of the target gene as additional perturbation feature. Concat then linear projection.
3. Replace `delta_effect` with low-rank operator: `delta_program = (U_p @ V_p^T) z_state_program`, where `U_p, V_p` are produced from the featurized embedding by small linear heads, rank `r ∈ {2, 4}`. Operator acts on the program-factorized representation already in `bridge.py`.
4. Combine 1 + 3.

**Constraints:**
- Smoke test at init must show output difference vs baseline within tolerance for seen perturbations.
- Log raw, post-gate, and final contribution ratios of the operator path vs the additive perturbation path.
- Operator rank capped at 4 for this family.
- Parameter budget: < 2x baseline parameter count.

**Stop/pivot rule.** Three consecutive Tier 1 discards or one Tier 2 cap-bound failure → cool down, log learning, move to Family B.

### Family B: Compositional Decomposition With Bounded Interaction

**Motivation.** The condition-mean lookup table cannot compose. There is no way for it to predict `(A+B)` from training on `A` alone and `B` alone. This is the entire reason real perturbation data is harder than the synthetic exact-key split.

**Hypothesis.** Enforcing `delta(A+B) = delta(A) + delta(B) + ε · interaction(A, B)` with the interaction term L2-regularized to be small gives the model structural composition. The additive part is recovered from singles; the interaction term carries the small but real epistasis Norman 2019 actually contains.

**Suggested experiments:**

1. Pure additive: predict `delta(A+B) := delta(A) + delta(B)` exactly, no interaction term. This is just the Step 0 single-additive baseline implemented inside the architecture. Use as a reference point.
2. Additive + small bilinear interaction: `interaction(A, B) := (W_A vec_A) ⊙ (W_B vec_B)`, projected to gene space, with L2 weight `λ_int`. Log `||interaction|| / ||additive||` ratio per batch.
3. Additive + low-rank interaction tensor: `interaction(A, B) := vec_A^T T vec_B`, `T` of rank `r ∈ {2, 4}`.
4. Combine with Family A featurized encoder if Family A produced a Tier 1 keep.

**Constraints:**
- Interaction term must be L2-regularized. Log `||interaction|| / ||additive||` ratio. Hard fail if this ratio exceeds 0.5 averaged over the epoch (interaction dominating means the model is not actually doing composition).
- Smoke test: with `λ_int → ∞`, model must reduce to the pure additive baseline within tolerance.
- Parameter budget: < 2x baseline.

**Stop/pivot rule.** Three consecutive Tier 1 discards → close Family B, write a closure note recommending pretraining or larger data as next phase.

## Tiered Evaluation

### Tier 1: Single Seed, Subsampled

- Single seed.
- 50% of train conditions subsampled for speed.
- Pass requires: Pearson delta on `exact_train_combo` > Family N condition-mean baseline AND > single-additive baseline, both by at least 0.02.
- Fail-fast: catastrophic mean-collapse, cap-bound contribution ratios, smoke-test init mismatch, interaction dominating ratio.

### Tier 2: Three Seeds, Full Train

- Three seeds.
- Full train split.
- Pass requires:
  - Tier 1 signal holds across seeds, std ≤ effect size;
  - no regression on `unseen_single` direction accuracy vs Family N by more than 5%;
  - no regression on top-20 DE overlap on `exact_train_combo` vs Tier 1;
  - contribution-ratio diagnostics within bounds.

### Tier 3: Promotion Gate

A Tier 3 pass requires all of:
- Pearson delta on `exact_train_combo` beats GEARS published number;
- top-20 DE overlap beats GEARS published number;
- preservation of `unseen_single` direction accuracy;
- no hard-fail diagnostic;
- full documentation including contribution ratios, interaction-to-additive ratio (Family B), and a marker-resolution check;
- domain-expert report written (see Required Diagnostics).

Only a Tier 3 pass can rebase the model of record. Tier 2 passes never rebase.

## Required Diagnostics

For every promoted experiment, log:

- contribution ratios (`raw`, `post_gate`, `final`) for any new mechanism path;
- `cap_hit_fraction` for any clipped or capped term;
- `weighted_aux_to_main_ratio` for every auxiliary loss;
- Family B specific: `||interaction|| / ||additive||` per batch and per epoch;
- direction-aware DE: signed DE agreement, not just unsigned overlap;
- per-perturbation prediction quality (not just aggregate);
- a marker-resolution check listing requested markers, resolved genes in Norman, missing genes, mapping method, coverage percent;
- variance-of-predictions across perturbations on `exact_train_combo` (mode-collapse check).

A win that improves Pearson delta while signed DE direction accuracy regresses is a useful failure, not a Tier 3 pass.

## Family Allocation

- Stage A (Step 0 baselines): up to 4 experiments, no architecture changes.
- Stage B (smoke families): 2 experiments per family before any deepening.
- Stage C (deepen productive families only): + up to 4 experiments per family that produced a Tier 1 keep.
- Hard experiment cap: 20 total architecture experiments across both families.

## Stop Conditions

- 20 architecture experiments reached.
- 2 Tier 3 passes (likely closure, but document and continue to wrap-up).
- 5 consecutive Tier 1 discards across families.
- Both families produce a Tier 2 fail with the same failure mode (cap-bound, mode collapse, or interaction domination).
- Norman split provenance or symbol-resolution ambiguity that cannot be resolved.
- User-directed closure.

When a stop condition fires: finish the running experiment, write the journal entry, update `results.tsv` and `family_allocation.md`, write `final_report.md`, stop the loop, wait for user direction.

## Decision Labels

Use these labels verbatim in journal entries:
`BASELINE_COMPLETE`, `TIER1_KEEP_CONTROLLED_SIGNAL`, `TIER1_DISCARD_NO_SIGNAL`, `TIER1_DISCARD_CAP_BOUND`, `TIER1_DISCARD_MODE_COLLAPSE`, `TIER1_DISCARD_INTERACTION_DOMINATING`, `TIER1_DISCARD_IMPLEMENTATION_MISMATCH`, `TIER2_PASS_CLEAN`, `TIER2_FAIL_VARIANCE`, `TIER2_FAIL_VALIDATOR_REGRESSION`, `TIER3_PASS_NEW_BASELINE`, `TIER3_FAIL_USEFUL_FAILURE`, `FAMILY_COOLDOWN`, `FAMILY_RETIRED`, `SEARCH_CLOSED_NO_NEW_BASELINE`.

## What This Run Is Not

- Not a synthetic experiment. Synthetic findings do not transfer back as evidence.
- Not allowed to modify locked files.
- Not a continuation of `autoresearch/synthetic-lite-v1`. That loop is closed. Findings from it are reference material, not active gates.
- Not a paper-writing phase. That happens after Tier 3, or after a closure decision.
- Not allowed to invent new splits if Norman's canonical split is awkward. If the canonical split is awkward, that is itself a result. Document it, do not work around it.

## Launch Message

Paste this into the coding agent to launch:

```
Read and apply outputs/autoresearch_norman_v1/autoresearch.md verbatim.
Run Step 0 baselines first. Do not start Family A or B until Step 0 is reviewed.
Keep GEARS published numbers and Family N as the active model of record until a Tier 3 pass supersedes them.
Do not modify locked files. Do not promote Tier 1 or Tier 2 near-misses.
When a stop condition fires, write final_report.md, stop the autonomous loop, and wait for explicit user instruction.
```

## Final Note

The honest read of `autoresearch/synthetic-lite-v1` is that the synthetic exact-key split could not test what this architecture was built for. Norman 2019 can. The result of this loop is one of three things, all of which are real outcomes:

1. A Tier 3 pass on `exact_train_combo` against GEARS. This is a paper.
2. A useful failure showing that operator-style and compositional mechanisms cannot beat GEARS on this benchmark without pretraining or more data. This is a methods note.
3. Closure showing the bottleneck is data or pretraining, not architecture. This is a pivot decision.

None of these requires another quarter of work. Two weeks of focused execution is the budget.
