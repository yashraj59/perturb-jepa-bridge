from __future__ import annotations

from dataclasses import replace

import torch
import pandas as pd

from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.config import ExperimentConfig
from perturb_jepa.training.checkpoint import load_checkpoint, save_checkpoint
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import BridgeTrainer


def synthetic_batch_from_config(config: ExperimentConfig):
    synthetic = config.synthetic
    return make_synthetic_bridge_batch(
        batch_size=synthetic.batch_size,
        genes=synthetic.genes,
        vocab_size=synthetic.vocab_size,
        image_channels=synthetic.image_channels,
        image_size=synthetic.image_size,
        patch_size=synthetic.patch_size,
        num_perturbations=synthetic.num_perturbations,
        num_types=synthetic.num_types,
        num_cell_lines=synthetic.num_cell_lines,
        num_batches=synthetic.num_batches,
    )


def test_experiment_config_round_trips_json(tmp_path):
    smoke = ExperimentConfig.smoke()
    config = replace(smoke, model=replace(smoke.model, adversary_scale=0.5))
    path = tmp_path / "experiment.json"

    config.save_json(path)
    loaded = ExperimentConfig.load_json(path)

    assert loaded == config
    assert loaded.to_dict() == config.to_dict()

    partial = ExperimentConfig.from_dict({"model": {"rna": {"vocab_size": 256}}})
    assert partial.model.rna.vocab_size == 256
    assert partial.model.rna.dim == smoke.model.rna.dim


def test_seed_everything_makes_synthetic_batches_repeatable():
    seed_everything(123)
    first = synthetic_batch_from_config(ExperimentConfig.smoke())
    seed_everything(123)
    second = synthetic_batch_from_config(ExperimentConfig.smoke())

    assert torch.equal(first.gene_ids, second.gene_ids)
    assert torch.equal(first.rna_token_mask, second.rna_token_mask)
    assert torch.equal(first.images, second.images)
    assert torch.equal(first.perturbation_id, second.perturbation_id)


def test_bridge_trainer_runs_synthetic_steps():
    seed_everything(0)
    config = ExperimentConfig.smoke()
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=config.loss,
        ema_decay=config.training.ema_decay,
        grad_clip_norm=1.0,
    )

    history = trainer.fit([synthetic_batch_from_config(config), synthetic_batch_from_config(config)])

    assert trainer.global_step == 2
    assert len(history) == 2
    assert history[-1]["total"] > 0.0


def test_checkpoint_save_load_restores_training_state(tmp_path):
    seed_everything(0)
    config = ExperimentConfig.smoke()
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(model, optimizer, weights=config.loss)
    trainer.step(synthetic_batch_from_config(config))

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        trainer_state=trainer.state_dict(),
        experiment_config=config,
        metadata={"split": "synthetic"},
    )

    restored_model = config.build_model()
    restored_optimizer = config.build_optimizer(restored_model.parameters())
    checkpoint = load_checkpoint(checkpoint_path, model=restored_model, optimizer=restored_optimizer)
    restored_trainer = BridgeTrainer(restored_model, restored_optimizer, weights=config.loss)
    restored_trainer.load_state_dict(checkpoint["trainer_state"])

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), restored_model.state_dict()[name].cpu())
    assert restored_trainer.global_step == 1
    assert ExperimentConfig.from_dict(checkpoint["experiment_config"]) == config
    assert checkpoint["metadata"] == {"split": "synthetic"}


def test_metadata_vocab_checkpoint_round_trip_keeps_ids_stable():
    train = pd.DataFrame(
        {
            "perturbation": ["drugB"],
            "perturbation_type": ["compound"],
            "cell_line": ["U2OS"],
            "batch": ["batchB"],
        }
    )
    eval_frame = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugB"],
            "perturbation_type": ["compound", "compound"],
            "cell_line": ["A549", "U2OS"],
            "batch": ["batchA", "batchB"],
        }
    )

    saved_vocab = MetadataVocab.from_frame(train)
    restored = MetadataVocab.from_dict(saved_vocab.to_dict())
    rebuilt_eval_vocab = MetadataVocab.from_frame(eval_frame)

    assert restored.encode_row({"perturbation": "drugB"})["perturbation_id"] == 1
    assert rebuilt_eval_vocab.encode_row({"perturbation": "drugB"})["perturbation_id"] == 2
    assert restored.encode_row({"perturbation": "drugA"})["perturbation_id"] == 0
