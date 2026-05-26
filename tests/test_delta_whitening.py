import torch

from perturb_jepa.models.bioflow_jepa import DeltaWhitening


def test_delta_whitening_fit_uses_supplied_train_delta_only():
    train = torch.tensor([[1.0, 2.0, 4.0], [3.0, 4.0, 8.0], [5.0, 6.0, 12.0]])
    heldout = torch.tensor([[20.0, 30.0, 60.0], [40.0, 50.0, 100.0]])
    whitener = DeltaWhitening(dim=3).fit(train)

    assert torch.allclose(whitener.mean.cpu(), train.mean(dim=0))
    assert not torch.allclose(whitener.mean.cpu(), torch.cat((train, heldout), dim=0).mean(dim=0))


def test_delta_whitening_inverse_reconstructs_without_heldout_fit():
    train = torch.randn(8, 5)
    heldout = torch.randn(4, 5)
    whitener = DeltaWhitening(dim=5).fit(train)

    reconstructed = whitener.unwhiten(whitener.whiten(heldout))

    assert torch.allclose(reconstructed, heldout, atol=1.0e-5)
