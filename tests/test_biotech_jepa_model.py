import torch

from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.biotech_losses import BioTechJEPALossWeights, biotech_jepa_loss
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def _tiny_model():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    config = BioTechJEPAConfig(
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
    return dataset, BioTechJEPA(config)


def test_biotech_jepa_has_factorized_real_jepa_outputs():
    dataset, model = _tiny_model()
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, bag_size=2, steps=1, seed=0))
    outputs = model.forward_jepa(batch)

    assert outputs["joint_z_bio"].shape == (2, 12)
    assert outputs["joint_z_tech"].shape == (2, 4)
    assert outputs["transition_bio_jepa_available"].item()
    assert outputs["rna_to_image_jepa_available"].item()
    assert outputs["image_to_rna_jepa_available"].item()
    assert not outputs["transition_bio_jepa_target"].requires_grad
    assert outputs["condition_key_exact_feature_present"].item() == 0
    assert outputs["pls_raw_linear_used_as_main_path"].item() == 0


def test_biotech_loss_is_finite_with_reconstruction_zero():
    dataset, model = _tiny_model()
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, bag_size=2, steps=1, seed=1))
    outputs = model.forward_jepa(batch)
    loss, diagnostics = biotech_jepa_loss(outputs, BioTechJEPALossWeights(count_aux=0.0))

    assert torch.isfinite(loss)
    assert diagnostics["raw_reconstruction_weighted_to_jepa_ratio"].item() == 0
