import torch

from perturb_jepa.models.bag_aggregator import MultiPrototypeBagAggregator


def test_multi_prototype_bag_aggregator_shapes_and_attention_mask():
    torch.manual_seed(0)
    model = MultiPrototypeBagAggregator(5, output_dim=7, num_prototypes=3)
    embeddings = torch.randn(2, 4, 5)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    output = model(embeddings, mask=mask)

    assert output.prototypes.shape == (2, 3, 7)
    assert output.bag_embedding.shape == (2, 7)
    assert output.attention.shape == (2, 3, 4)
    assert torch.allclose(output.attention.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(output.attention[0, :, 2:] == 0)
    assert torch.all(output.attention[1, :, [1, 3]] == 0)


def test_multi_prototype_bag_aggregator_default_k_and_gradients():
    torch.manual_seed(0)
    model = MultiPrototypeBagAggregator(4)
    embeddings = torch.randn(2, 5, 4, requires_grad=True)

    output = model(embeddings)
    loss = output.prototypes.pow(2).mean() + output.bag_embedding.pow(2).mean()
    loss.backward()

    assert output.prototypes.shape == (2, 8, 4)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
