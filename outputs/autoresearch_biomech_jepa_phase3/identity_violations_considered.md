# Identity Violations Considered

Rejected or deferred ideas:

- Training BioTech-JEPA longer by default: rejected by prompt; latest run is diagnostic only.
- Using raw-linear PLS as the main representation path: rejected; PLS remains protected baseline/audit reference only.
- Using `condition_key`, `biological_key`, or exact target-key one-hot features: rejected as leakage.
- Using Norman `dose_val` as chemical dose: rejected; Norman guide notation is fixed guide presence.
- Claiming Norman batch disentanglement: rejected unless real batch/acquisition metadata is recovered.
- Replacing JEPA with a contrastive-only model or autoencoder: rejected; Phase 3 must retain online encoders, EMA targets, stop-gradient latent targets, query predictors, and latent prediction losses.
