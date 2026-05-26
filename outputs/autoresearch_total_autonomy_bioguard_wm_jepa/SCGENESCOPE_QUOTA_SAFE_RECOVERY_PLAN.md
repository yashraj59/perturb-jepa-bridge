# scGeneScope Quota-Safe Recovery Plan

## Status
`HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED` is active.

The autonomous research loop must not launch more scGeneScope payload downloads, Tier 3 runs, JEPA training, whitening, calibration, residual selection, or target-statistic fitting while this hard escalation remains unresolved.

## Trigger Evidence
F092 attempted the F091-approved smallest paired feature dry run:

```tsv
modality	repo_path	bytes
rna	features/rnaseq/scvi/n200/round_2.h5ad	2565764148
image	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503
```

The paired footprint was `13589664651` bytes. The files downloaded, but status writing failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

The generated payload directory `data/raw/scgenescope` was removed after the failure.

## What Is Proven
- The scGeneScope adapter contract is valid at the Croissant metadata level.
- The replicate split contract maps `3=train`, `5=validation`, `4=test`, and `1/2=alternate_test`.
- The smallest paired feature candidate is identifiable from saved Hugging Face metadata.
- The actual payload attempt exceeded the effective workspace quota even though ordinary `df` free-space estimates looked sufficient.
- The F092 retry guard now blocks automatic retry while `HARD_ESCALATION_REPORT.md` exists.

## What Is Not Proven
- The real feature H5AD `obs` contract has not been validated.
- The real RNA/image condition-pair table has not been built from payload metadata.
- No scGeneScope Tier 3 scoring can be launched.
- No BioGuard/BioGuard-WM candidate can be promoted from scGeneScope evidence.

## Allowed Work While Hard Escalation Is Active
- Edit tests, docs, and safeguards.
- Inspect existing metadata artifacts already present under `outputs/`.
- Prepare local-only code paths for a user-supplied smaller manifest or subset.
- Run repo tests that do not download payloads or train large models.

## Forbidden Until Recovery Evidence Exists
- Re-run F092 without explicit override.
- Download scGeneScope feature or measured payloads.
- Train, tune, whiten, calibrate, or fit target statistics on scGeneScope.
- Treat Norman RNA-only diagnostics as cross-modal validation.
- Promote any candidate.

## Recovery Option A: Smaller Manifest-Backed Subset
Before resuming the external-validator branch, provide or build a local manifest that references a smaller paired feature subset.

Minimum requirements:
- both RNA and image modality rows are present;
- `Treatment`, `Replicate`, `batch`, and `Group` are recoverable;
- split mapping follows the Croissant contract;
- the paired local payload footprint is below the effective quota with room for status artifacts;
- the subset is declared diagnostic-only unless it preserves held-out condition semantics.

## Recovery Option B: Explicit Quota-Safe Retry
Before retrying the F091-approved 13.59 GB pair, prove all of:

```text
HARD_ESCALATION_REPORT.md reviewed by user or operator
--acknowledge-hard-escalation-retry intentionally supplied
data/raw/scgenescope absent or intentionally preserved
status-write reserve succeeds
payload target directory has quota headroom for at least 2x paired payload plus artifacts
download cache location is known and quota-safe
```

The first resumed run must remain obs-only/backed metadata inspection. It must not train or fit anything.

## Recovery Option C: Alternative External Validator
If scGeneScope remains quota-blocked, the next amendment should search for a smaller condition-paired scRNA plus imaging perturbation validator. Any replacement must still satisfy:

- real or condition-paired scRNA plus imaging;
- perturbation/control labels;
- held-out perturbation or held-out experiment split;
- no use of exact target-key one-hot features;
- no eval target means or pooled train+test statistics.

## Current Model Status
The protected rank-3 train-split-only PLS raw-linear readout remains the model of record. The protected full-ridge transition floor remains active. No Phase 8/F091/F092 result promotes a JEPA candidate.
