import pytest

from perturb_jepa.models.jepawm_predictor import context_contract_dict, validate_context_length


def test_context_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="context length mismatch"):
        validate_context_length(context_length_train=3, context_length_eval=4)


def test_context_contract_records_layout():
    contract = context_contract_dict(context_length_train=3, context_length_eval=3, rollout_steps=1)

    assert contract["context_length_train"] == 3
    assert contract["context_length_eval"] == 3
    assert contract["token_layout"] == ["source_control_z_bio", "ridge_floor_predicted_z_bio", "learned_residual_query_token"]
    assert contract["rollout_steps"] == 1
