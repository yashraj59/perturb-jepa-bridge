from scripts.run_bioguard_wm_phase8_v2 import jepa_identity_flags


def test_operator_identity_flags_keep_pls_out_of_main_path():
    flags = jepa_identity_flags(full_jepa=False)

    assert flags["raw_linear_pls_main_path_used"] == 0.0
    assert flags["ridge_floor_fallback_present"] == 1.0
    assert flags["residual_floor_preservation_verified"] == 1.0


def test_full_identity_flags_include_encoder_and_teacher_paths():
    flags = jepa_identity_flags(full_jepa=True)

    assert flags["online_context_encoders_present"] == 1.0
    assert flags["ema_target_encoders_present"] == 1.0
    assert flags["teacher_stop_gradient_verified"] == 1.0
    assert flags["action_conditioned_transition_jepa_present"] == 1.0
