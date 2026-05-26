# Identity Violations Considered

- Reusing the protected PLS raw-linear heads as the main BioAction-JEPA representation path: rejected. This would violate the real-JEPA requirement and the protected baseline rule.
- Training a direct expression mean decoder and adding a small cosine term: rejected. The latent JEPA objective must dominate with reconstruction/count weights at zero or auxiliary.
- Feeding `condition_key`, `biological_key`, or exact target-key one-hot features to the model: rejected. These labels are evaluation metadata only.
- Promoting exact-key `test` results alone: rejected. Tier 3 requires held-out perturbation/dose support.

