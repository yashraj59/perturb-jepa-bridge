import torch
import pytest

from perturb_jepa.models.jepawm_predictor import ActionAdaLNBlock, ActionAdaLNPredictor, BioJEPAWMContextConfig
from perturb_jepa.models.action_adaln_predictor import ActionAdaLNRoPEPredictor


def test_action_adaln_block_zero_modulation_does_not_explode():
    torch.manual_seed(0)
    block = ActionAdaLNBlock(16, 5, 4, adaln_zero=True)
    tokens = torch.randn(3, 3, 16)
    action = torch.randn(3, 5)
    out = block(tokens, action)

    assert out.shape == tokens.shape
    assert torch.isfinite(out).all()


def test_action_adaln_predictor_zero_output_init_returns_zero_residual():
    torch.manual_seed(0)
    config = BioJEPAWMContextConfig(z_dim=6, action_dim=4, predictor_dim=16, depth=2, heads=4, context_length=3)
    predictor = ActionAdaLNPredictor(config)
    source = torch.randn(5, 6)
    floor = source + 0.1 * torch.randn(5, 6)
    action = torch.randn(5, 4)

    residual, aux = predictor(source, floor, action)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-7)
    assert int(aux["context_length"].item()) == 3


def test_phase8_predictor_enforces_context_contract():
    predictor = ActionAdaLNRoPEPredictor(dim=16, action_dim=5, depth=1, num_heads=4, max_context_tokens=3, output_dim=6)
    tokens = torch.randn(2, 3, 16)
    action = torch.randn(2, 5)

    out = predictor(tokens, action)

    assert out.shape == (2, 6)
    assert torch.allclose(out, torch.zeros_like(out), atol=1.0e-7)
    with pytest.raises(ValueError, match="context token count"):
        predictor(torch.randn(2, 4, 16), action)
