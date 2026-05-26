from scripts.run_bioguard_wm_phase8_v2 import jepa_identity_flags, sanitize_model_input_record


def _assert_real_jepa_contract(flags):
    assert flags["raw_linear_pls_main_path_used"] == 0.0
    assert flags["teacher_stop_gradient_verified"] == 1.0
    assert flags["action_conditioned_transition_jepa_present"] == 1.0
    assert flags["latent_prediction_loss_present"] == 1.0
    assert flags["cross_modal_jepa_present"] == 1.0


def test_full_jepa_identity_contract_rejects_raw_linear_and_missing_cross_modal():
    flags = jepa_identity_flags(full_jepa=True)

    _assert_real_jepa_contract(flags)


def test_forbidden_metadata_keys_are_not_model_inputs():
    record = sanitize_model_input_record(
        {
            "rna": 1,
            "condition_key": "forbidden",
            "biological_key": "forbidden",
            "exact_target_key": "forbidden",
            "target_key": "forbidden",
        }
    )

    assert "rna" in record
    assert "condition_key" not in record
    assert "biological_key" not in record
    assert "exact_target_key" not in record
    assert "target_key" not in record
