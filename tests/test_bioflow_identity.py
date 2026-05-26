import torch

from perturb_jepa.models.bioflow_jepa import BioFlowJEPA, BioFlowJEPAConfig
from perturb_jepa.models.biotech_jepa import BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def _tiny_bioflow():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    spec = synthetic_action_descriptor_spec(dataset)
    base = BioTechJEPAConfig(
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
    )
    model = BioFlowJEPA(BioFlowJEPAConfig(base_biotech_config=base, action_dim=spec.descriptor_dim, flow_steps=2))
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, bag_size=2, steps=1, seed=0))
    matrix = synthetic_action_descriptor_matrix(dataset, spec)
    batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, matrix)
    return model, batch


def test_bioflow_outputs_identity_and_stop_gradient_contract():
    model, batch = _tiny_bioflow()
    outputs = model.forward_bioflow(batch)

    assert outputs["z_pred"].shape == (2, 12)
    assert outputs["velocity_pred"].shape == (2, 12)
    assert not outputs["target_z_bio_teacher"].requires_grad
    assert outputs["condition_key_exact_feature_present"].item() == 0
    assert outputs["biological_key_onehot_present"].item() == 0
    assert outputs["pls_raw_linear_used_as_main_path"].item() == 0
    assert outputs["z_tech_used_as_transition_shortcut"].item() == 0


def test_condition_key_tensors_are_not_model_inputs():
    _, batch = _tiny_bioflow()

    assert not hasattr(batch, "condition_key_tensor")
    assert not hasattr(batch, "biological_key_tensor")
