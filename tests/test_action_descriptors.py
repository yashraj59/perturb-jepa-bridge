import numpy as np
import torch

from perturb_jepa.training.action_descriptors import (
    descriptors_for_perturbation_ids,
    heldout_descriptor_coverage,
    synthetic_action_descriptor_matrix,
    synthetic_action_descriptor_spec,
)
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def test_synthetic_heldout_perturbations_have_nonleaky_descriptors():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_genetic_anchor_lite", seed=0))
    spec = synthetic_action_descriptor_spec(dataset)
    matrix = synthetic_action_descriptor_matrix(dataset, spec)
    coverage = heldout_descriptor_coverage(dataset, matrix)

    assert matrix.shape == (dataset.config.num_perturbations, spec.descriptor_dim)
    assert coverage["heldout_descriptor_coverage"] == 1.0
    assert coverage["heldout_nonzero_descriptor_fraction"] == 1.0
    assert np.all(matrix[dataset.config.control_perturbation_id] == 0.0)


def test_descriptor_lookup_returns_batch_rows():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_genetic_anchor_lite", seed=0))
    matrix = synthetic_action_descriptor_matrix(dataset)
    perturbation_id = torch.tensor([0, 9, 10, 11], dtype=torch.long)
    descriptors = descriptors_for_perturbation_ids(perturbation_id, matrix)

    assert descriptors.shape == (4, matrix.shape[1])
    assert descriptors[0].sum().item() == 0.0
    assert torch.all(descriptors[1:].sum(dim=1) > 0.0)
