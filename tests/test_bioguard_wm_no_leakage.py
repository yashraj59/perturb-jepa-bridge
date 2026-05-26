from scripts.run_bioguard_wm_phase8_v2 import sanitize_model_input_record


def test_forbidden_keys_are_dropped_from_model_inputs():
    record = {
        "source_z": 1,
        "action_features": 2,
        "condition_key": "x",
        "biological_key": "y",
        "exact_target_key": "z",
        "eval_target_mean": 3,
        "batch_id": 4,
    }

    sanitized = sanitize_model_input_record(record)

    assert "source_z" in sanitized
    assert "action_features" in sanitized
    assert "condition_key" not in sanitized
    assert "biological_key" not in sanitized
    assert "exact_target_key" not in sanitized
    assert "eval_target_mean" not in sanitized
    assert "batch_id" not in sanitized
