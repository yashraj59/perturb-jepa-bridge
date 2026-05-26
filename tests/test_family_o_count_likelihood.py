from __future__ import annotations

import numpy as np

from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteConfig, generate_synthetic_biology_lite
from scripts.run_family_n_distillation import ConditionFeatureEncoder
from scripts.run_family_o_count_likelihood import (
    count_pair_records,
    nb_nll_for_records,
    poisson_nll_for_records,
    resolve_count_matrix,
)


def test_family_o_uses_raw_synthetic_counts_when_available():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=23))

    counts, diagnostics = resolve_count_matrix(dataset, pseudo_count_scale=40.0)

    np.testing.assert_allclose(counts, dataset.observed_counts)
    assert diagnostics["raw_count_available"] is True
    assert diagnostics["pseudo_count_used"] is False
    assert diagnostics["count_path"] == "raw_synthetic_observed_counts"
    assert diagnostics["integer_valued_fraction"] == 1.0
    assert 0.0 < diagnostics["zero_fraction"] < 1.0


def test_family_o_condition_features_exclude_batch_id():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=24))
    counts, _ = resolve_count_matrix(dataset, pseudo_count_scale=40.0)
    train_records = count_pair_records(dataset, counts, split="train")

    encoder = ConditionFeatureEncoder(dataset, train_records)

    payload = encoder.to_dict()
    assert payload["batch_id_feature_present"] is False
    assert all("batch" not in name for name in payload["feature_names"])


def test_family_o_count_nlls_are_finite_on_train_table_means():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=25))
    counts, _ = resolve_count_matrix(dataset, pseudo_count_scale=40.0)
    records = count_pair_records(dataset, counts, split="train")
    predictions = np.stack([record["target_count_mean"] for record in records])
    dispersion = np.full(dataset.config.genes, 0.25, dtype=float)

    poisson_nll = poisson_nll_for_records(records, predictions)
    nb_nll = nb_nll_for_records(records, predictions, dispersion)

    assert np.isfinite(poisson_nll)
    assert np.isfinite(nb_nll)
    assert poisson_nll > 0.0
    assert nb_nll > 0.0
