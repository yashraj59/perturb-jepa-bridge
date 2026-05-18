import torch

from perturb_jepa.distribution_losses import mmd_rbf_loss, sliced_wasserstein_loss


def test_mmd_rbf_loss_supports_masks_and_prototype_flattening():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5)
    y = x + 0.1 * torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    loss = mmd_rbf_loss(x, y, mask_x=mask, mask_y=mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert mmd_rbf_loss(x, x, mask_x=mask, mask_y=mask).abs() < 1e-6


def test_sliced_wasserstein_loss_supports_unequal_flattened_samples():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6)
    y = torch.randn(3, 2, 6)

    loss = sliced_wasserstein_loss(x, y, num_projections=8)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_distribution_losses_are_smaller_for_matched_distributions():
    torch.manual_seed(0)
    x = torch.randn(10, 4)
    y_close = x + 0.01 * torch.randn(10, 4)
    y_far = x + 2.0

    assert mmd_rbf_loss(x, y_close) < mmd_rbf_loss(x, y_far)
    assert sliced_wasserstein_loss(x, y_close, num_projections=16) < sliced_wasserstein_loss(
        x,
        y_far,
        num_projections=16,
    )
