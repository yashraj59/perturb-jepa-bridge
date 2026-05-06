import torch
import torch.nn.functional as F

from perturb_jepa.models.adversary import BatchAdversary, GradientReversalLayer, gradient_reversal_ramp


def test_gradient_reversal_layer_reverses_gradient():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = GradientReversalLayer(scale=0.5)(x).sum()
    y.backward()

    assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5]))


def test_batch_adversary_outputs_logits_and_reverses_feature_gradient():
    torch.manual_seed(0)
    adversary = BatchAdversary(3, 2, hidden_dim=4, depth=1, scale=1.0)
    x_plain = torch.randn(5, 3, requires_grad=True)
    x_reversed = x_plain.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 1, 0, 1, 0])

    plain_logits = adversary.classifier(x_plain)
    reversed_logits = adversary(x_reversed)
    F.cross_entropy(plain_logits, labels).backward()
    F.cross_entropy(reversed_logits, labels).backward()

    assert reversed_logits.shape == (5, 2)
    assert torch.allclose(x_reversed.grad, -x_plain.grad, atol=1e-6)


def test_gradient_reversal_ramp_bounds():
    assert gradient_reversal_ramp(0, 100) == 0.0
    mid = gradient_reversal_ramp(50, 100)
    end = gradient_reversal_ramp(100, 100)
    assert 0.0 < mid < end <= 1.0
