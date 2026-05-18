from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from perturb_jepa.baselines.batch_only_baseline import batch_only_retrieval_metrics
from perturb_jepa.config import ExperimentConfig, ObjectiveScheduleConfig
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.losses import BridgeLossWeights, bridge_loss
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.objectives import weighted_bridge_total
from scripts.evaluate_retrieval import _EvalBagCollator
from scripts.train_bridge import _BridgeBagCollator, _load_pretrained_encoder


def _items() -> list[dict[str, object]]:
    condition = {
        "perturbation": "drugA",
        "perturbation_type": "compound",
        "dose": "10uM",
        "time": "48h",
        "cell_line": "U2OS",
        "condition_id": "drugA|10uM|48h|U2OS",
    }
    return [
        {
            "bio_key": ("drugA", "10uM", "48h", "U2OS"),
            "condition": condition,
            "condition_id": "drugA|10uM|48h|U2OS",
            "rna": {
                "x": np.ones((2, 4), dtype=np.float32),
                "sample_ids": ["r0", "r1"],
                "tech": [
                    {"batch": "rna_b1", "plate": "NA"},
                    {"batch": "rna_b2", "plate": "NA"},
                ],
                "cell_meta": [],
            },
            "image": {
                "x": np.ones((2, 1, 4, 4), dtype=np.float32),
                "sample_ids": ["i0", "i1"],
                "tech": [
                    {"batch": "image_b1", "plate": "plate1"},
                    {"batch": "image_b1", "plate": "plate2"},
                ],
                "cell_meta": [],
            },
        }
    ]


def _vocab() -> MetadataVocab:
    return MetadataVocab.from_frames(
        [
            pd.DataFrame(
                {
                    "perturbation": ["drugA", "drugA", "drugA"],
                    "perturbation_type": ["compound", "compound", "compound"],
                    "cell_line": ["U2OS", "U2OS", "U2OS"],
                    "batch": ["rna_b1", "rna_b2", "image_b1"],
                }
            )
        ]
    )


def _small_config(vocab: MetadataVocab) -> ExperimentConfig:
    config = ExperimentConfig.smoke()
    return replace(
        config,
        model=replace(
            config.model,
            rna=replace(config.model.rna, vocab_size=4, max_genes=4),
            image=replace(config.model.image, in_channels=1, image_size=4, patch_size=2, max_patches=4),
            perturbation=replace(
                config.model.perturbation,
                num_perturbations=vocab.num_perturbations,
                num_types=vocab.num_types,
                num_cell_lines=vocab.num_cell_lines,
                num_batches=vocab.num_batches,
            ),
        ),
    )


def test_real_bridge_collator_generates_masks_and_modality_batch_ids():
    vocab = _vocab()
    masked = _BridgeBagCollator(
        np.arange(4),
        vocab,
        device=torch.device("cpu"),
        rna_mask_prob=1.0,
        image_patch_mask_prob=1.0,
        image_patch_size=2,
    )(_items())
    unmasked = _BridgeBagCollator(
        np.arange(4),
        vocab,
        device=torch.device("cpu"),
        rna_mask_prob=0.0,
        image_patch_mask_prob=0.0,
        image_patch_size=2,
    )(_items())

    assert masked["rna_token_mask"].shape == (1, 2, 4)
    assert masked["image_patch_mask"].shape == (1, 2, 4)
    assert masked["rna_token_mask"].all()
    assert masked["image_patch_mask"].all()
    assert not unmasked["rna_token_mask"].any()
    assert not unmasked["image_patch_mask"].any()
    assert int(masked["rna_batch_id"][0]) != 0
    assert int(masked["image_batch_id"][0]) != 0
    assert int(masked["rna_batch_id"][0]) != int(masked["image_batch_id"][0])
    assert int(masked["metadata"]["batch_id"][0]) == 0


def test_batch_only_eval_metadata_is_modality_specific():
    batch = _EvalBagCollator(np.arange(4), _vocab(), device=torch.device("cpu"))(_items())
    rna_metadata = pd.DataFrame(batch["rna_rows"])
    image_metadata = pd.DataFrame(batch["image_rows"])
    rna_metadata["condition_key"] = rna_metadata["condition_id"]
    image_metadata["condition_key"] = image_metadata["condition_id"]

    assert rna_metadata.loc[0, "batch"] == "rna_b1"
    assert image_metadata.loc[0, "batch"] == "image_b1"
    metrics = batch_only_retrieval_metrics(rna_metadata, image_metadata, ks=(1,))
    assert "batch_only_recall@1" in metrics


def test_real_batch_objective_schedule_changes_loss_weights():
    vocab = _vocab()
    config = _small_config(vocab)
    model = config.build_model()
    batch = _BridgeBagCollator(
        np.arange(4),
        vocab,
        device=torch.device("cpu"),
        rna_mask_prob=1.0,
        image_patch_mask_prob=1.0,
        image_patch_size=2,
    )(_items())
    outputs = model(
        gene_ids=batch["gene_ids"],
        expression_values=batch["expression_values"],
        rna_token_mask=batch["rna_token_mask"],
        rna_bag_mask=batch["rna_bag_mask"],
        images=batch["images"],
        image_patch_mask=batch["image_patch_mask"],
        image_bag_mask=batch["image_bag_mask"],
        **batch["metadata"],
    )
    image_target = patchify(batch["images"].reshape(-1, 1, 4, 4), 2).reshape(1, 2, 4, 4)
    _, terms = bridge_loss(
        outputs,
        rna_values=batch["expression_values"],
        rna_mask=batch["rna_token_mask"],
        image_patches=image_target,
        image_patch_mask=batch["image_patch_mask"],
        perturbation_id=batch["metadata"]["perturbation_id"],
        rna_batch_id=batch["rna_batch_id"],
        image_batch_id=batch["image_batch_id"],
        weights=BridgeLossWeights(),
    )
    raw_terms = {name: value for name, value in terms.items() if name != "total"}
    schedule = ObjectiveScheduleConfig(enabled=True, reconstruction_warmup_steps=2)
    total, scheduled = weighted_bridge_total(raw_terms, weights=config.loss, schedule=schedule, step=0)

    expected = config.loss.rna_mask * raw_terms["rna_mask"] + config.loss.image_mask * raw_terms["image_mask"]
    assert torch.allclose(total, expected)
    assert scheduled["weighted/align"].item() == pytest.approx(0.0)


def test_pretrained_encoder_loader_syncs_student_and_teacher(tmp_path):
    source = _small_config(_vocab()).build_model()
    target = _small_config(_vocab()).build_model()
    with torch.no_grad():
        for index, parameter in enumerate(source.rna_encoder.parameters()):
            parameter.fill_(0.01 * (index + 1))
        for parameter in target.rna_encoder.parameters():
            parameter.zero_()
        for parameter in target.rna_teacher.parameters():
            parameter.fill_(3.0)

    checkpoint_path = tmp_path / "pretrain_rna.pt"
    save_checkpoint(
        checkpoint_path,
        model=source,
        metadata={"stage": "pretrain_rna"},
    )

    info = _load_pretrained_encoder(target, checkpoint_path, modality="rna", device=torch.device("cpu"))

    assert info["checkpoint_stage"] == "pretrain_rna"
    assert info["loaded_tensors"] > 0
    for name, tensor in target.rna_encoder.state_dict().items():
        assert torch.equal(tensor, source.rna_encoder.state_dict()[name])
    for name, tensor in target.rna_teacher.state_dict().items():
        assert torch.equal(tensor, target.rna_encoder.state_dict()[name])
    assert all(not parameter.requires_grad for parameter in target.rna_teacher.parameters())
