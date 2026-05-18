import inspect

import pytest
import torch

from perturb_jepa.models.adversary import BatchAdversary
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead


def test_projection_heads_accept_embeddings_only():
    rna_head = RNAProjectionHead(5, 7, 3)
    image_head = ImageProjectionHead(5, 7, 3)
    x = torch.randn(4, 5)

    assert rna_head(x).shape == (4, 3)
    assert image_head(x).shape == (4, 3)
    assert torch.allclose(rna_head(x).norm(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(image_head(x).norm(dim=-1), torch.ones(4), atol=1e-6)
    with pytest.raises(TypeError):
        rna_head(x, batch_id=torch.zeros(4, dtype=torch.long))


def test_batch_adversary_forward_has_no_label_argument():
    signature = inspect.signature(BatchAdversary.forward)
    assert "labels" not in signature.parameters
    assert "batch_id" not in signature.parameters
