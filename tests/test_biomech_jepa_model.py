import torch

from perturb_jepa.models.biomech_jepa import BioMechanisticJEPA, BioMechanisticJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.biomech_losses import biomech_jepa_loss
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def _model_and_batch(enable_population=True):
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    spec = synthetic_action_descriptor_spec(dataset)
    config = BioMechanisticJEPAConfig(
        rna=RNAEncoderConfig(vocab_size=dataset.config.genes, dim=16, depth=1, heads=4, max_genes=dataset.config.genes, pooling="mean_tokens"),
        image=ImageEncoderConfig(
            in_channels=dataset.config.image_channels,
            image_size=dataset.config.image_size,
            patch_size=dataset.config.patch_size,
            dim=16,
            depth=1,
            heads=4,
            pooling="mean_patches",
        ),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=dataset.config.num_perturbations,
            num_types=3,
            num_cell_lines=dataset.config.num_cell_lines,
            num_batches=dataset.config.num_batches,
            dim=16,
            descriptor_dim=spec.descriptor_dim,
            perturbation_feature_mode="feature_only",
        ),
        shared_dim=16,
        bio_dim=12,
        tech_dim=4,
        predictor_dim=16,
        target_query_dim=12,
        predictor_heads=4,
        num_condition_prototypes=2,
        num_rna_program_targets=4,
        num_image_region_targets=2,
        enable_delta_jepa=True,
        enable_program_action_encoder=True,
        enable_population_transition=enable_population,
        descriptor_gene_dim=spec.gene_dim,
        descriptor_program_dim=spec.program_dim,
        action_dim=12,
    )
    model = BioMechanisticJEPA(config)
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, bag_size=2, steps=1, seed=0))
    matrix = synthetic_action_descriptor_matrix(dataset, spec)
    batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, matrix)
    return model, batch


def test_biomech_jepa_outputs_delta_and_population_shapes():
    model, batch = _model_and_batch(enable_population=True)
    outputs = model.forward_jepa(batch)

    assert outputs["joint_z_bio"].shape == (2, 12)
    assert outputs["joint_z_tech"].shape == (2, 4)
    assert outputs["delta_teacher"].shape == (2, 12)
    assert outputs["delta_pred"].shape == (2, 12)
    assert not outputs["delta_jepa_target"].requires_grad
    assert outputs["predicted_target_bio_prototypes"].shape == (2, 2, 12)
    assert outputs["condition_key_exact_feature_present"].item() == 0
    assert outputs["pls_raw_linear_used_as_main_path"].item() == 0


def test_biomech_loss_is_finite():
    model, batch = _model_and_batch(enable_population=True)
    outputs = model.forward_jepa(batch)
    loss, diagnostics = biomech_jepa_loss(outputs)

    assert torch.isfinite(loss)
    assert diagnostics["raw_reconstruction_weighted_to_jepa_ratio"].item() == 0


def test_control_action_delta_prediction_is_zero_masked():
    model, batch = _model_and_batch(enable_population=False)
    batch.target_gene_ids = batch.control_gene_ids
    batch.target_expression_values = batch.control_expression_values
    batch.target_counts = batch.control_counts
    batch.target_images = batch.control_images
    batch.target_rna_bag_mask = batch.control_rna_bag_mask
    batch.target_image_bag_mask = batch.control_image_bag_mask
    batch.perturbation_id = torch.zeros_like(batch.perturbation_id)
    batch.perturbation_type_id = torch.zeros_like(batch.perturbation_type_id)
    batch.descriptor = torch.zeros_like(batch.descriptor)

    outputs = model.forward_jepa(batch)

    assert torch.allclose(outputs["delta_pred"], torch.zeros_like(outputs["delta_pred"]))
