# Papers Consulted

## What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

- Local file: `papers/2512.24497v3.pdf`
- Title from local PDF metadata: `What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?`

Extracted implementation lessons:
- JEPA-WMs are predictive latent dynamics models.
- The predictor/dynamics model is central, not just the encoder.
- AdaLN + RoPE is a strong action-conditioning architecture.
- Multistep rollout can help robustness but needs correct target contracts.
- Train/eval context length must match.
- L2/cosine latent endpoint costs are natural for calibration.
- Deterministic predictors can average multimodal futures; residuals need uncertainty or abstention.

Mapping to this project:
- Build an action-AdaLN + RoPE residual predictor.
- Enforce fixed context contract.
- Add biological two-step rollout only after one-step gates pass.
- Keep floor-preserving residual and train-only calibration.

## Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics

- Local file: `papers/2602.02093v1.pdf`
- Title from local PDF metadata: `Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics`
- arXiv/DOI from metadata: `https://arxiv.org/abs/2602.02093v1` / `https://doi.org/10.48550/arXiv.2602.02093`
- License from metadata: `http://creativecommons.org/licenses/by/4.0/`

Extracted implementation lessons:
- Cell-JEPA uses a student RNA encoder on masked expression values and an EMA teacher RNA encoder on the unmasked view.
- The latent target is a stop-gradient teacher cell embedding, and a predictor maps the student cell embedding to that target.
- Expression-value masking is used while gene identities remain visible by default.
- Per-cell quantile binning and random expressed-gene subsampling are part of the RNA input recipe.
- A reconstruction term is retained only as an auxiliary anchor; the JEPA representation loss should dominate.
- The perturbation section is a warning for this project: better absolute post-perturbation state prediction did not translate into better delta/effect-size estimation.

Mapping to this project:
- Add a representation-only RNA warmstart branch before another residual/risk-gate operator.
- Keep delta/effect-size metrics protected and do not promote vanilla Cell-JEPA as a perturbation-transition solution.
- After warmstart, rerun the protected train-only action-ridge transition floor on frozen `z_bio`.


## External Primary References Checked

- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA feature prediction: https://arxiv.org/abs/2404.08471
- JEPA-WM code/paper artifacts: https://github.com/facebookresearch/jepa-wms
- CellOT neural optimal transport: https://www.nature.com/articles/s41592-023-01969-x
- GEARS perturbation prediction: https://www.nature.com/articles/s41587-023-01905-6
- CPA perturbation autoencoder: https://github.com/theislab/CPA
- scVI batch-aware single-cell latent model: https://www.nature.com/articles/s41592-018-0229-2
- VICReg collapse avoidance: https://arxiv.org/abs/2105.04906
- Barlow Twins redundancy reduction: https://arxiv.org/abs/2103.03230
- BYOL online/target SSL: https://arxiv.org/abs/2006.07733
- DINO self-distillation: https://arxiv.org/abs/2104.14294
- Conformal risk control: https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/conformal-risk.pdf
- Safe policy improvement: https://jmlr.org/papers/v22/19-707.html

Outcome for this run:
- Use JEPA-WM predictor primitives for Phase 8 v3.
- Use Cell-JEPA only as a representation warmstart diagnostic unless it preserves delta/effect-size transition floors.
- If residual calibration fails again, prefer metric/data redesign or safe abstention over forcing residual scale.
