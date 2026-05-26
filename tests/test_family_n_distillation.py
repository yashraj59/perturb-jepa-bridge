from __future__ import annotations

import numpy as np

from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import biological_key, pair_records
from scripts.run_family_n_distillation import ConditionFeatureEncoder, TrainOnlyConditionalMeanTable


def test_family_n_feature_encoder_excludes_batch_id() -> None:
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=2))
    train = pair_records(dataset, split="train")
    encoder = ConditionFeatureEncoder(dataset, train)
    record = dict(train[0])
    other_batch = dict(record, batch_id=int(record["batch_id"]) + 100)

    assert biological_key(record) == biological_key(other_batch)
    assert not encoder.to_dict()["batch_id_feature_present"]
    np.testing.assert_allclose(encoder.transform([record]), encoder.transform([other_batch]))


def test_train_only_condition_table_ignores_test_targets() -> None:
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=2))
    train = pair_records(dataset, split="train")
    test = pair_records(dataset, split="test")
    table = TrainOnlyConditionalMeanTable(train)

    original, original_diagnostics = table.predict(test)
    corrupted_test = []
    for record in test:
        corrupted = dict(record)
        corrupted["target_mean"] = np.asarray(record["target_mean"], dtype=float) + 1_000_000.0
        corrupted_test.append(corrupted)
    corrupted, corrupted_diagnostics = table.predict(corrupted_test)

    np.testing.assert_allclose(np.stack(original), np.stack(corrupted))
    assert original_diagnostics["teacher_target_test_rows_used"] == 0.0
    assert corrupted_diagnostics["teacher_target_test_rows_used"] == 0.0


def test_train_only_condition_table_fallbacks_use_train_statistics() -> None:
    train = [
        {
            "perturbation_id": 1,
            "cell_line_id": 0,
            "dose": 0.0,
            "time": 0.0,
            "batch_id": 0,
            "target_mean": np.array([1.0, 2.0]),
        },
        {
            "perturbation_id": 1,
            "cell_line_id": 1,
            "dose": 1.0,
            "time": 0.0,
            "batch_id": 1,
            "target_mean": np.array([3.0, 4.0]),
        },
        {
            "perturbation_id": 2,
            "cell_line_id": 0,
            "dose": 0.0,
            "time": 0.0,
            "batch_id": 0,
            "target_mean": np.array([5.0, 6.0]),
        },
    ]
    table = TrainOnlyConditionalMeanTable(train)
    records = [
        dict(train[0], batch_id=99),
        dict(train[0], dose=0.25, batch_id=99),
        dict(train[0], cell_line_id=99, batch_id=99),
        dict(train[0], perturbation_id=99, cell_line_id=99, batch_id=99),
    ]

    predictions, diagnostics = table.predict(records)

    np.testing.assert_allclose(predictions[0], [1.0, 2.0])
    np.testing.assert_allclose(predictions[1], [1.0, 2.0])
    np.testing.assert_allclose(predictions[2], [2.0, 3.0])
    np.testing.assert_allclose(predictions[3], [3.0, 4.0])
    assert diagnostics["fallback_exact_count"] == 1.0
    assert diagnostics["fallback_nearest_same_perturbation_cell_count"] == 1.0
    assert diagnostics["fallback_global_perturbation_mean_count"] == 1.0
    assert diagnostics["fallback_global_train_mean_count"] == 1.0
    assert diagnostics["batch_id_excluded"] == 1.0
