import torch
import pytest

from perturb_jepa.training.bioguard_wm_losses import multistep_rollout_loss
from perturb_jepa.training.bioguard_wm_rollouts import BiologicalRolloutExample, validate_rollout_examples_no_leakage


def test_step_two_rollout_compares_to_second_target():
    z1 = torch.tensor([[1.0, 0.0]])
    z2 = torch.tensor([[1.0, 1.0]])
    pred1 = z1.clone()
    pred2 = z2.clone()

    good = multistep_rollout_loss([pred1, pred2], [z1, z2])
    bad = multistep_rollout_loss([pred1, pred2], [z1, z1])

    assert good.item() == 0.0
    assert bad.item() > 0.0


def test_rollout_leakage_metadata_is_rejected():
    example = BiologicalRolloutExample(
        source_key="x",
        action_1="a",
        action_2=None,
        source_z=torch.zeros(2),
        target_1_z=torch.ones(2),
        target_2_z=None,
        action_1_features=torch.ones(2),
        action_2_features=None,
        metadata={"condition_key": "forbidden"},
    )

    with pytest.raises(ValueError, match="leakage"):
        validate_rollout_examples_no_leakage([example])
