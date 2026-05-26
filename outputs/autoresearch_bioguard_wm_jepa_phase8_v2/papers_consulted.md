# Papers Consulted

## What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

- Paper: What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?
- Local path: `papers/2512.24497v3.pdf`
- Evidence read from local PDF: `Published in Transactions on Machine Learning Research (05/2026) What Drives Success in Physical Planning with Joint- Embedding Predictive World Models? Basile Terver basile.terver@ens.psl.edu Meta F AIR Inria Paris Tsung-Yen Yangjimmytyyang@meta.com Meta F AIR Jean Ponce jean.ponce@ens.fr Ecole normale supérieure / PSL New York University Adrien Bardes abardes@meta.com Meta F AIR Yann Le Cun yann.lecun@nyu.edu New York University Reviewed on OpenReview:https: // openreview. net/ forum? id= cHZn5Gdh8e Abstract A long-standing challenge in AI is to develop agents capable of solving a wide range`...

Extracted lessons:
1. JEPA-WM identity is latent predictive dynamics, not reconstruction/reward/value/policy heads.
2. The predictor/dynamics model is the main engineering object.
3. Action conditioning with AdaLN + RoPE is a strong predictor recipe.
4. Multistep rollout can improve robustness, but target contracts must be exact.
5. Train/eval context length mismatch is invalid.
6. L2 endpoint latent cost is a strong planning/calibration cost.
7. Deterministic predictors can average multimodal futures; residual deployment needs abstention/risk control.

Mapping to this repo:
- state = `z_bio`;
- action = perturbation/gene/drug descriptor;
- future state = perturbed teacher `z_bio`;
- protected transition floor = train-only full action-ridge delta;
- residual = optional floor-preserving correction, selected only through train-only calibration.
