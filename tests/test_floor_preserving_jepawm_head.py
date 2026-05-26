import torch

from perturb_jepa.models.jepawm_predictor import ActionAdaLNPredictor, BioJEPAWMContextConfig, FloorPreservingJEPAWMTransitionHead
from perturb_jepa.models.bioguard_wm_jepa import BioGuardWMJEPAConfig, FloorPreservingJEPAWMTransitionHead as Phase8FloorHead


def _head():
    config = BioJEPAWMContextConfig(z_dim=5, action_dim=3, predictor_dim=24, depth=1, heads=4, context_length=3)
    return FloorPreservingJEPAWMTransitionHead(ActionAdaLNPredictor(config))


def test_zero_gate_equals_floor_exactly():
    head = _head()
    source = torch.randn(4, 5)
    floor_delta = torch.randn(4, 5) * 0.1
    action = torch.randn(4, 3)

    out = head(source, action, ridge_floor_delta=floor_delta, residual_gate=0.0, residual_scale=1.0)

    assert torch.allclose(out["pred_z"], out["floor_z"], atol=0.0)


def test_zero_scale_equals_floor_exactly():
    head = _head()
    source = torch.randn(4, 5)
    floor_delta = torch.randn(4, 5) * 0.1
    action = torch.randn(4, 3)

    out = head(source, action, ridge_floor_delta=floor_delta, residual_gate=1.0, residual_scale=0.0)

    assert torch.allclose(out["pred_z"], out["floor_z"], atol=0.0)


def test_nonzero_gate_scale_can_change_prediction_after_output_bias():
    head = _head()
    with torch.no_grad():
        head.predictor.out_head.bias.fill_(0.1)
    source = torch.randn(4, 5)
    floor_delta = torch.randn(4, 5) * 0.1
    action = torch.randn(4, 3)

    out = head(source, action, ridge_floor_delta=floor_delta, residual_gate=1.0, residual_scale=1.0)

    assert not torch.allclose(out["pred_z"], out["floor_z"])


def test_phase8_floor_preserving_head_zero_scale_exact_floor():
    config = BioGuardWMJEPAConfig(bio_dim=5, action_dim=3, predictor_dim=24, predictor_depth=1, predictor_heads=4)
    head = Phase8FloorHead(config)
    source = torch.randn(4, 5)
    floor_delta = torch.randn(4, 5) * 0.1
    action = torch.randn(4, 3)

    out = head(source, action, floor_delta=floor_delta, residual_scale_override=0.0)

    assert torch.max(torch.abs(out["predicted_delta"] - floor_delta)).item() < 1.0e-7
    assert torch.max(torch.abs(out["predicted_endpoint"] - out["floor_endpoint"])).item() < 1.0e-7
