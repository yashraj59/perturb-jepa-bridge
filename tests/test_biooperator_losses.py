import torch
import torch.nn.functional as F


def source_improvement_hinge(source, pred, target, margin=0.02, magnitude_ratio_limit=2.0):
    source_cos = F.cosine_similarity(source, target, dim=-1).detach()
    pred_cos = F.cosine_similarity(pred, target, dim=-1)
    pred_delta = pred - source
    true_delta = target - source
    ratio = pred_delta.norm(dim=-1) / true_delta.norm(dim=-1).clamp_min(1.0e-8)
    return F.relu(source_cos + margin - pred_cos).mean() + F.relu(ratio - magnitude_ratio_limit).mean()


def magnitude_contract_flags_explosion(source, pred, target, limit=2.0):
    ratio = (pred - source).norm(dim=-1) / (target - source).norm(dim=-1).clamp_min(1.0e-8)
    return bool((ratio > limit).any().item())


def test_hinge_penalizes_worse_than_source_and_exploding_magnitude():
    source = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    target = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    good = target.clone()
    worse = F.normalize(torch.tensor([[1.0, -0.1]]), dim=-1)
    exploding = source + 10.0 * (target - source)

    assert source_improvement_hinge(source, good, target).item() == 0.0
    assert source_improvement_hinge(source, worse, target).item() > 0.0
    assert source_improvement_hinge(source, exploding, target).item() > 0.0


def test_magnitude_contract_flags_endpoint_cosine_style_explosion():
    source = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[0.0, 1.0]])
    pred = source + 16.4 * (target - source)

    assert magnitude_contract_flags_explosion(source, pred, target, limit=2.0)
