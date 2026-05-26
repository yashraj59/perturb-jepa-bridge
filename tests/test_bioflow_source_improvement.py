import torch

from perturb_jepa.training.bioflow_losses import bioflow_jepa_loss, source_improvement_hinge_loss


def test_source_improvement_hinge_zero_only_after_margin_improvement():
    target = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
    source = torch.nn.functional.normalize(torch.tensor([[0.8, 0.6, 0.0]]), dim=-1)
    better = target.clone()
    not_enough = source.clone()

    assert source_improvement_hinge_loss(source, better, target, margin=0.02).item() == 0.0
    assert source_improvement_hinge_loss(source, not_enough, target, margin=0.02).item() > 0.0


def test_action_negative_loss_increases_when_wrong_action_is_closer():
    source = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
    target = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
    pred = source
    wrong = target
    outputs = {
        "z_pred": pred,
        "target_z_bio_teacher": target.detach(),
        "source_z_bio_online": source,
        "true_delta": (target - source).detach(),
        "pred_delta": pred - source,
        "velocity_pred": target - source,
        "velocity_target": (target - source).detach(),
        "wrong_action_z_pred": wrong,
    }

    _, diagnostics = bioflow_jepa_loss(outputs)

    assert diagnostics["loss/action_negative"].item() > 0.0
