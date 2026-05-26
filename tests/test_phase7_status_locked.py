from perturb_jepa.training.bioguard_wm_status import parse_phase7_status


def test_phase7_status_locked_matches_expected_closure():
    status = parse_phase7_status()

    assert status.decision == "PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA"
    assert status.no_residual_passed
    assert status.later_experiments_not_run
    assert status.floor_values_match
