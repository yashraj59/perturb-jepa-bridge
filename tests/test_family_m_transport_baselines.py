from __future__ import annotations

import numpy as np

from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import (
    FeatureProjector,
    biological_key,
    pair_records,
    predict_knn_residual_transport,
    predict_residualized_matching,
    sinkhorn_plan,
)


def test_family_m_biological_key_excludes_batch_id() -> None:
    first = {"perturbation_id": 1, "cell_line_id": 0, "dose": 1.0, "time": 0.0, "batch_id": 0}
    second = dict(first, batch_id=1)

    assert biological_key(first) == biological_key(second)
    assert "batch_id" not in biological_key(first)


def test_residualized_matching_uses_exact_no_batch_condition_keys() -> None:
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=2))
    train = pair_records(dataset, split="train")
    test = pair_records(dataset, split="test")

    predictions, diagnostics = predict_residualized_matching(train, test)

    assert len(predictions) == len(test)
    assert diagnostics["exact_match_fraction"] == 1.0
    assert diagnostics["batch_id_excluded"] == 1.0


def test_knn_residual_transport_returns_one_prediction_per_test_pair() -> None:
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=2))
    train = pair_records(dataset, split="train")
    test = pair_records(dataset, split="test")
    fit_values = np.concatenate([record["control_cells"] for record in train], axis=0)
    projector = FeatureProjector(dataset=dataset, feature_space="program", fit_values=fit_values)

    predictions, diagnostics = predict_knn_residual_transport(train, test, projector=projector, k=3)

    assert np.asarray(predictions).shape == (len(test), dataset.config.genes)
    assert diagnostics["exact_match_fraction"] == 1.0
    assert diagnostics["mean_neighbor_count"] == 3.0


def test_sinkhorn_plan_matches_uniform_marginals() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.0], [2.0]])

    plan = sinkhorn_plan(x, y, epsilon=0.5, iterations=100)

    np.testing.assert_allclose(plan.sum(axis=1), np.full(3, 1.0 / 3.0), atol=1e-5)
    np.testing.assert_allclose(plan.sum(axis=0), np.full(2, 1.0 / 2.0), atol=1e-5)
