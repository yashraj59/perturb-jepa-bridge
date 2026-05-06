import torch

from perturb_jepa.losses import MultiPositiveInfoNCELoss, build_multi_positive_weights


def test_multi_positive_infonce_exact_and_weak_weights():
    weights = build_multi_positive_weights(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        positive_mask=torch.tensor(
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, True],
            ]
        ),
        bio_keys={
            "perturbation": ["p0", "p1", "p1", "p2"],
            "moa": ["m0", "m0", "m1", "m1"],
        },
        weak_positive_weight=0.2,
    )

    assert weights[1, 2] == 1.0
    assert weights[0, 1] == 0.2
    assert weights[2, 3] == 0.2
    assert torch.diag(weights).eq(1.0).all()


def test_multi_positive_infonce_batch_metadata_does_not_affect_matching():
    weights = build_multi_positive_weights(
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        bio_keys={
            "condition": ["a", "b", "c"],
            "batch": ["same", "same", "same"],
        },
    )

    assert torch.equal(weights, torch.eye(3))


def test_multi_positive_infonce_loss_decreases_for_closer_matches():
    torch.manual_seed(0)
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.2)
    x = torch.randn(6, 8)
    y_far = torch.randn(6, 8)
    y_close = x + 0.05 * torch.randn(6, 8)
    labels = torch.arange(6)

    assert loss_fn(x, y_close, labels=labels) < loss_fn(x, y_far, labels=labels)


def test_multi_positive_infonce_is_symmetric_for_transposed_inputs():
    torch.manual_seed(0)
    loss_fn = MultiPositiveInfoNCELoss(temperature=0.3)
    x = torch.randn(4, 6)
    y = torch.randn(4, 6)
    positive_mask = torch.eye(4, dtype=torch.bool)

    xy = loss_fn(x, y, positive_mask=positive_mask)
    yx = loss_fn(y, x, positive_mask=positive_mask)

    assert torch.allclose(xy, yx, atol=1e-6)
