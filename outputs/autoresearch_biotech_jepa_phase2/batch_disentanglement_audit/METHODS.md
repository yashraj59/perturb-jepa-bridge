# Methods

The Stage 1 audit uses synthetic datasets generated locally by `synthetic_biology_lite`; no real data is used.

Batch confounding is measured at condition-bag level with contingency tables, Cramer's V, and normalized mutual information.

Raw/protected probes are trained on condition-level train split embeddings and evaluated on the requested eval split. Probe labels are batch labels used only for diagnostics.

Protected PLS latents are rank-3 PLS readouts fit on train RNA and image pseudobulks.

BioAction zero-step and Phase 1 checkpoint probes freeze the model and train only diagnostic batch probes on collected latent states.

Split-half ceilings split cells within each condition bag into two halves and evaluate same-bag/same-biological retrieval plus batch decodability.
