# F097 cpg0003 Rosetta Fresh Confirmation Preflight

## Decision
`F097_CPG0003_ROSETTA_FRESH_PREFLIGHT_READY_FOR_RUNNER_IMPLEMENTATION`

No model is promoted. F096 remains a non-promoting scGeneScope repair-loop pass,
and the protected rank-3 train-split-only PLS raw-linear readout remains the
model of record until a fresh external confirmation actually passes.

## Purpose
F096 fixed the F082 scGeneScope external-validation failure with frozen PubChem
fingerprint descriptors plus train-only delta-calibrated ProgramBootstrapJEPA.
Because scGeneScope guided F093/F094/F095 repairs, F096 cannot promote. F097
starts a genuinely fresh external confirmation track.

## Dataset Candidate
- source: `cpg0003-rosetta/broad/workspace/preprocessed_data/`
- panel: `CDRPBIO-BBBC036-Bray`
- modalities: Cell Painting morphology profiles plus L1000 expression profiles
- cell line: `U2OS`
- raw files are local only under ignored `data/raw/cpg0003_rosetta/`
- raw files must not be committed or pushed

## Files Inspected
```text
data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz
data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/L1000/replicate_level_l1k.csv.gz
```

## Contract Findings
```text
Cell Painting rows = 21122
Cell Painting columns = 627
L1000 rows = 6929
L1000 columns = 988
shared exact compound+dose pairs = 1469
Cell Painting replicate count per shared pair = min 4, median 8, max 16
L1000 replicate count per shared pair = min 1, median 2, max 2
controls = DMSO/negcon present in both modalities
shared-pair SMILES missing = 0
```

## Required Runner
Implement the next runner as a fresh confirmation of the frozen F096 path, not
a new architecture:

```text
ProgramBootstrapJEPA
SMILES/PubChem-derived non-exact chemical descriptors plus dose
train-only source/control centroids
train-only Cell Painting PCA target space
train-only delta calibration
source-as-target baseline
protected full-ridge audit floor
no-residual/raw JEPA baseline
transition improvement
delta cosine
recall@1
L1000->Cell Painting and Cell Painting->L1000 retrieval
delta rank
identity checks
leakage checks
```

## Leakage Controls
- Do not use `condition_key`, `biological_key`, exact treatment one-hot, held-out
  target means, pooled train+test statistics, or floor predictions as candidate
  model inputs.
- Use compound IDs only for pairing, grouping, and retrieval relevance labels.
- Use SMILES/PubChem-derived descriptors and dose as action inputs.
- Fit PCA, scaling, baselines, and calibration on train split only.
- Keep PLS/full-ridge as an audit floor only, not as the JEPA representation path.

## Caveat
cpg0003 Rosetta is L1000 expression plus Cell Painting morphology, not scRNA.
A pass is a fresh external perturbational transcriptomics+morphology
confirmation/stress test. If strict paired scRNA plus imaging is required for
promotion, a later fresh scRNA+imaging protocol is still required.

## Resume Command Skeleton
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest \
  tests/test_bioguard_wm_calibration.py \
  tests/test_program_bootstrap_jepa.py \
  tests/test_scgenescope_adapter.py -q
```

Then implement/run the F097 cpg0003 Rosetta external confirmation runner on GPU
unless the GPU is unavailable or already occupied.
