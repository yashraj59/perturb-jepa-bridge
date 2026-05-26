# Frozen PLS JEPA Tier 2 Summary

## Candidate

- Mechanism: rank-3 prefit PLS readout, frozen during JEPA/reconstruction training.
- Dataset: `synth_micro`.
- Seeds: `0/1/2`.
- Steps per seed: `50`.
- Real data used: `false`.
- GPU used: `false`.
- Checkpoint serialization: each run writes `frozen_pls_jepa.pt` with `metadata.prefit_readout`, `metadata.prefit_readout_path`, protected metric deltas, and frozen-readout drift.

## Results

| seed | protected geometry preserved | readout drift | RNA->image R@1 | RNA->image R@5 | RNA latent R2 | image latent R2 | batch bal acc | batch majority |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.00000000 | 0.2188 | 0.5000 | 0.4155 | 0.8876 | 0.5000 | 0.5000 |
| 1 | 1 | 0.00000000 | 0.2188 | 0.6875 | 0.7563 | 0.9379 | 0.5625 | 0.5000 |
| 2 | 1 | 0.00000000 | 0.2812 | 0.8125 | 0.6069 | 0.9146 | 0.3750 | 0.5000 |

Tier 2 means:

- RNA->image recall@1: `0.2396 +/- 0.0295`
- RNA->image recall@5: `0.6667 +/- 0.1284`
- RNA latent R2: `0.5929 +/- 0.1395`
- Image latent R2: `0.9134 +/- 0.0206`
- Batch balanced accuracy: `0.4792 +/- 0.0780`
- Wallclock minutes per seed: `1.2468 +/- 0.0410`

## Interpretation

Freezing the closed-form PLS readout successfully protects the shared RNA/image condition geometry while JEPA and reconstruction losses train around it. This is the right safety boundary for the next loop: the frozen readout is a non-regression retrieval baseline, and learned heads must earn promotion independently.

## Next Step

Add a separate distillation head that learns from the frozen PLS readout while the retrieval path still uses the frozen readout. Promote the learned head only after it passes the same Tier 2 gates without using the frozen readout as `rna_retrieval`/`image_retrieval`.
