import torch
import pytest

from perturb_jepa.models.jepawm_predictor import RotaryEmbedding, RotarySelfAttention
from perturb_jepa.models.jepawm_rope import RotaryEmbedding as Phase8RotaryEmbedding, apply_rope


def test_rotary_embedding_shapes_are_broadcastable():
    rope = RotaryEmbedding(8)
    cos, sin = rope(3, device=torch.device("cpu"), dtype=torch.float32)

    assert cos.shape == (1, 1, 3, 8)
    assert sin.shape == (1, 1, 3, 8)


def test_rotary_self_attention_shape_deterministic_and_finite():
    torch.manual_seed(0)
    attn = RotarySelfAttention(16, 4, use_rope=True).eval()
    x = torch.randn(2, 3, 16)

    with torch.no_grad():
        left = attn(x)
        right = attn(x)

    assert left.shape == (2, 3, 16)
    assert torch.allclose(left, right)
    assert torch.isfinite(left).all()


def test_phase8_rope_preserves_norm_and_rejects_odd_head_dim():
    rope = Phase8RotaryEmbedding(8)
    x = torch.randn(2, 4, 3, 8)
    cos, sin = rope(3, device=x.device, dtype=x.dtype)

    out = apply_rope(x, cos, sin)

    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1.0e-5)
    with pytest.raises(ValueError, match="even"):
        Phase8RotaryEmbedding(7)
