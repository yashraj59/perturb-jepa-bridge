from __future__ import annotations

import torch

from perturb_jepa.models.bag_aggregator import MeanBagAggregator


def test_mean_bag_aggregator_preserves_output_contract():
    aggregator = MeanBagAggregator(4, num_prototypes=3)
    embeddings = torch.randn(2, 5, 4)
    output = aggregator(embeddings)
    assert output.prototypes.shape == (2, 3, 4)
    assert output.bag_embedding.shape == (2, 4)
    assert output.attention.shape == (2, 3, 5)
    assert torch.allclose(output.attention.sum(dim=-1), torch.ones(2, 3))


def test_mean_bag_aggregator_respects_mask():
    aggregator = MeanBagAggregator(2, num_prototypes=1)
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    output = aggregator(embeddings, mask=mask)
    expected = aggregator.bag_norm(torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(output.bag_embedding, expected)

