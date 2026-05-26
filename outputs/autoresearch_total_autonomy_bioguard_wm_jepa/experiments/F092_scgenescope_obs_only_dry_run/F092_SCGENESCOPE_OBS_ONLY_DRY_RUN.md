# F092 scGeneScope Obs-Only Backed Dry Run

## Decision
`HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED`

## Purpose
F091 approved a smallest paired scGeneScope feature dry run under storage/RAM/backed-IO gates. F092 attempted to download only those two approved feature files and then open H5AD `obs` metadata in backed mode.

## Result
The two approved feature files downloaded, but the run failed while writing `download_file_status.tsv`:

```text
OSError: [Errno 122] Disk quota exceeded
```

This is a hard escalation under the active protocol. The downloaded payload directory was removed to recover quota:

```text
data/raw/scgenescope
```

## Payloads Before Cleanup
```tsv
modality	path	bytes
rna	data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad	2565764148
image	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad	11023900503
```

## Controls
- No model was trained.
- No JEPA encoder, transition head, residual gate, whitening, target mean, or calibration model was fit.
- No model was promoted.
- The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

## Required Next Step
Do not retry F092 automatically. Use a smaller manifest-backed feature subset or allocate explicit workspace quota for at least the paired feature payload plus output artifacts before another real scGeneScope payload attempt.
