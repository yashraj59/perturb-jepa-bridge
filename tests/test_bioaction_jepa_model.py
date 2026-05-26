from __future__ import annotations

import torch

from perturb_jepa.evaluation.bioaction_metrics import evaluate_bioaction_batches
from perturb_jepa.models.bioaction_jepa import BioActionJEPA, BioActionJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.bioaction_losses import BioActionJEPALossWeights, bioaction_jepa_loss
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def _model_and_batch(rna_only: bool = False, image_only: bool = False):
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    config = BioActionJEPAConfig(
        rna=RNAEncoderConfig(vocab_size=dataset.config.genes, dim=16, depth=1, heads=4, max_genes=dataset.config.genes),
        image=ImageEncoderConfig(
            in_channels=dataset.config.image_channels,
            image_size=dataset.config.image_size,
            patch_size=dataset.config.patch_size,
            dim=16,
            depth=1,
            heads=4,
        ),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=dataset.config.num_perturbations,
            num_types=2,
            num_cell_lines=dataset.config.num_cell_lines,
            num_batches=dataset.config.num_batches,
            dim=16,
        ),
        shared_dim=16,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=4,
        target_query_dim=16,
        num_condition_prototypes=2,
        num_rna_program_targets=4,
        num_image_region_targets=4,
        gene_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment),
    )
    model = BioActionJEPA(config)
    batch = next(
        iter_bioaction_condition_batches(
            dataset,
            split="train",
            batch_size=2,
            steps=1,
            seed=1,
            device="cpu",
            rna_only=rna_only,
            image_only=image_only,
            paired=not (rna_only or image_only),
        )
    )
    return model, batch


def test_bioaction_jepa_builds_and_forward_paired():
    model, batch = _model_and_batch()
    outputs = model.forward_jepa(batch)
    required = (
        "rna_program_jepa_pred",
        "rna_program_jepa_target",
        "image_region_jepa_pred",
        "image_region_jepa_target",
        "rna_to_image_jepa_pred",
        "rna_to_image_jepa_target",
        "image_to_rna_jepa_pred",
        "image_to_rna_jepa_target",
        "joint_to_rna_jepa_pred",
        "joint_to_rna_jepa_target",
        "joint_to_image_jepa_pred",
        "joint_to_image_jepa_target",
        "transition_rna_jepa_pred",
        "transition_rna_jepa_target",
        "transition_image_jepa_pred",
        "transition_image_jepa_target",
        "transition_joint_jepa_pred",
        "transition_joint_jepa_target",
        "shared_state",
        "joint_condition_state",
        "rna_condition_state",
        "image_condition_state",
    )
    for key in required:
        assert key in outputs
    assert all(not value.requires_grad for key, value in outputs.items() if key.endswith("_target"))
    assert not any(parameter.requires_grad for parameter in model.rna_target_encoder.parameters())
    assert float(outputs["pls_raw_linear_used_as_main_path"]) == 0.0
    assert float(outputs["condition_key_exact_feature_present"]) == 0.0


def test_bioaction_jepa_forward_rna_only_and_image_only():
    for kwargs in ({"rna_only": True}, {"image_only": True}):
        model, batch = _model_and_batch(**kwargs)
        outputs = model.forward_jepa(batch)
        assert "transition_joint_jepa_pred" in outputs
        loss, _ = bioaction_jepa_loss(outputs, BioActionJEPALossWeights(raw_rna_reconstruction=0.0, raw_image_reconstruction=0.0))
        assert torch.isfinite(loss)
        assert float(loss.detach()) > 0.0


def test_bioaction_jepa_ema_update_changes_teacher():
    model, batch = _model_and_batch()
    before = next(model.rna_target_encoder.parameters()).detach().clone()
    outputs = model.forward_jepa(batch)
    loss, _ = bioaction_jepa_loss(outputs)
    loss.backward()
    with torch.no_grad():
        for parameter in model.rna_context_encoder.parameters():
            parameter.add_(0.01)
            break
    model.update_teachers(decay=0.5)
    after = next(model.rna_target_encoder.parameters()).detach()
    assert not torch.allclose(before, after)


def test_bioaction_evaluation_reports_batch_probe_metrics():
    model, _ = _model_and_batch()
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    batches = iter_bioaction_condition_batches(dataset, split="test", batch_size=4, steps=2, seed=2, device="cpu")
    metrics = evaluate_bioaction_batches(model, batches, device="cpu")
    assert "rna_embedding_batch_probe_n_classes" in metrics
    assert "image_embedding_batch_probe_n_classes" in metrics
    assert "joint_embedding_batch_probe_n_classes" in metrics
