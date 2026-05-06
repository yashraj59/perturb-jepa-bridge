from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from perturb_jepa.config import ExperimentConfig, KendallUncertaintyConfig, ObjectiveScheduleConfig
from perturb_jepa.losses import BridgeLossWeights
from perturb_jepa.training.objectives import scheduled_loss_weights
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import BridgeTrainer, loss_for_batch


def test_scheduled_loss_weights_warm_up_then_anneal_reconstruction() -> None:
    weights = BridgeLossWeights(
        rna_mask=2.0,
        image_mask=4.0,
        jepa=6.0,
        align=8.0,
        perturbation_cls=10.0,
        batch_adv=12.0,
        counterfactual=14.0,
    )
    schedule = ObjectiveScheduleConfig(
        enabled=True,
        reconstruction_warmup_steps=2,
        reconstruction_anneal_steps=2,
        reconstruction_final_scale=0.25,
    )

    warmup = scheduled_loss_weights(weights, schedule, step=1)
    assert warmup.rna_mask == pytest.approx(2.0)
    assert warmup.image_mask == pytest.approx(4.0)
    assert warmup.jepa == pytest.approx(0.0)
    assert warmup.align == pytest.approx(0.0)

    halfway = scheduled_loss_weights(weights, schedule, step=2)
    assert halfway.rna_mask == pytest.approx(2.0 * 0.625)
    assert halfway.image_mask == pytest.approx(4.0 * 0.625)
    assert halfway.jepa == pytest.approx(6.0 * 0.5)
    assert halfway.perturbation_cls == pytest.approx(10.0 * 0.5)

    annealed = scheduled_loss_weights(weights, schedule, step=3)
    assert annealed.rna_mask == pytest.approx(2.0 * 0.25)
    assert annealed.image_mask == pytest.approx(4.0 * 0.25)
    assert annealed.align == pytest.approx(8.0)
    assert annealed.counterfactual == pytest.approx(14.0)


def test_scheduled_loss_total_uses_reconstruction_only_during_warmup() -> None:
    seed_everything(0)
    config = ExperimentConfig.smoke()
    model = config.build_model()
    batch = make_synthetic_bridge_batch(batch_size=2)
    schedule = ObjectiveScheduleConfig(enabled=True, reconstruction_warmup_steps=4)

    total, terms = loss_for_batch(model, batch, weights=config.loss, schedule=schedule, step=0)

    expected = config.loss.rna_mask * terms["rna_mask"] + config.loss.image_mask * terms["image_mask"]
    assert torch.allclose(total, expected)
    assert terms["schedule/reconstruction_scale"].item() == pytest.approx(1.0)
    assert terms["schedule/non_reconstruction_scale"].item() == pytest.approx(0.0)
    assert "align" in terms


def test_training_config_round_trips_objective_controls() -> None:
    config = ExperimentConfig.from_dict(
        {
            "training": {
                "objective_schedule": {
                    "enabled": True,
                    "reconstruction_warmup_steps": 3,
                    "reconstruction_anneal_steps": 5,
                    "reconstruction_final_scale": 0.2,
                },
                "uncertainty_weighting": {
                    "enabled": True,
                    "term_names": ["rna_mask", "image_mask"],
                    "initial_log_variance": -0.25,
                },
            }
        }
    )

    assert config.training.objective_schedule.reconstruction_warmup_steps == 3
    assert config.training.uncertainty_weighting.term_names == ("rna_mask", "image_mask")
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_uncertainty_weighting_parameters_train_on_synthetic_batch() -> None:
    seed_everything(0)
    config = ExperimentConfig.smoke()
    config = replace(
        config,
        training=replace(
            config.training,
            uncertainty_weighting=KendallUncertaintyConfig(
                enabled=True,
                term_names=("rna_mask", "image_mask"),
            ),
        ),
    )
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        uncertainty_weighting=config.training.uncertainty_weighting,
    )
    assert trainer.uncertainty_weighting is not None
    assert len(optimizer.param_groups) == 2

    before = trainer.uncertainty_weighting.log_variances.detach().clone()
    terms = trainer.step(make_synthetic_bridge_batch(batch_size=2))
    after = trainer.uncertainty_weighting.log_variances.detach()

    assert not torch.allclose(before, after)
    assert "uncertainty/rna_mask_log_variance" in terms
    assert "uncertainty/image_mask_precision" in terms
    assert trainer.state_dict()["uncertainty_weighting"]["log_variances"].shape == (2,)


def test_uncertainty_weighting_can_wrap_grouped_loss_names() -> None:
    seed_everything(0)
    config = ExperimentConfig.smoke()
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        uncertainty_weighting=KendallUncertaintyConfig(enabled=True, term_names=("jepa", "align")),
    )

    terms = trainer.step(make_synthetic_bridge_batch(batch_size=2))

    assert "weighted/rna_jepa" in terms
    assert "weighted/image_jepa" in terms
    assert "weighted/jepa" in terms
    assert "uncertainty/jepa_precision" in terms
    assert "uncertainty/align_log_variance" in terms
