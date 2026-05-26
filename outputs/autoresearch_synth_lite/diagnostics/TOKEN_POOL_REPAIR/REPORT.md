# Token/Patch Pool Repair Report

## Setup

Focused repair check:

- RNA encoder global readout: `mean_tokens`
- image encoder global readout: `mean_patches`
- bag aggregator: `mean`
- dropout: `0.0`
- shared variance weight: `0.2`
- shared covariance weight: `0.01`
- dataset: `synth_micro`
- seed: `0`
- steps: `30`
- device: CPU

## Result

- Collapse flag: `1`
- RNA shared min std: `0.031627`
- image shared min std: `0.009962`
- RNA-to-image recall@1: `0.0625`
- random recall@1: `0.0625`
- RNA biological latent R2: `-0.3632`
- image biological latent R2: `0.9210`
- batch probe balanced accuracy: `0.46875`

## Interpretation

Token/patch pooling improved RNA variance and perturbation-direction cosine but did not pass the hard collapse gate because image min std stayed just below `0.01`. More importantly, retrieval stayed exactly at random and RNA latent R2 remained negative.

This repair does not solve the core problem. The model needs a stronger RNA condition-level biological readout or auxiliary target, not just a different global token pooling rule.

## Decision

Discard as a Tier 1 candidate. Keep the encoder pooling option in code because it is useful for diagnostics, but do not promote it.
