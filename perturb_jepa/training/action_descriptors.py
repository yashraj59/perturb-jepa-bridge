from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset


@dataclass(frozen=True)
class ActionDescriptorSpec:
    gene_names: tuple[str, ...]
    program_names: tuple[str, ...]
    perturbation_to_gene: dict[int, int]
    perturbation_to_program: dict[int, int]
    control_perturbation_id: int

    @property
    def gene_dim(self) -> int:
        return len(self.gene_names)

    @property
    def program_dim(self) -> int:
        return len(self.program_names)

    @property
    def descriptor_dim(self) -> int:
        return self.gene_dim + self.program_dim


def synthetic_action_descriptor_spec(dataset: SyntheticBiologyLiteDataset) -> ActionDescriptorSpec:
    """Create non-leaky perturbation descriptors from synthetic config only.

    The synthetic generator historically exposed perturbation ids. Phase 3
    treats each non-control perturbation as a synthetic gene action and maps it
    to a deterministic program bucket, so held-out perturbations have
    descriptors without reading target expression or exact biological keys.
    """

    gene_names = tuple(f"synthetic_gene_{idx}" for idx in range(dataset.config.num_perturbations))
    program_count = max(1, int(dataset.config.num_programs))
    program_names = tuple(f"synthetic_program_{idx}" for idx in range(program_count))
    perturbation_to_gene = {idx: idx for idx in range(dataset.config.num_perturbations)}
    perturbation_to_program = {idx: idx % program_count for idx in range(dataset.config.num_perturbations)}
    perturbation_to_program[int(dataset.config.control_perturbation_id)] = 0
    return ActionDescriptorSpec(
        gene_names=gene_names,
        program_names=program_names,
        perturbation_to_gene=perturbation_to_gene,
        perturbation_to_program=perturbation_to_program,
        control_perturbation_id=int(dataset.config.control_perturbation_id),
    )


def synthetic_action_descriptor_matrix(
    dataset: SyntheticBiologyLiteDataset,
    spec: ActionDescriptorSpec | None = None,
) -> np.ndarray:
    spec = spec or synthetic_action_descriptor_spec(dataset)
    values = np.zeros((dataset.config.num_perturbations, spec.descriptor_dim), dtype=np.float32)
    for perturbation_id in range(dataset.config.num_perturbations):
        if perturbation_id == spec.control_perturbation_id:
            continue
        gene_index = spec.perturbation_to_gene[perturbation_id]
        program_index = spec.perturbation_to_program[perturbation_id]
        values[perturbation_id, gene_index] = 1.0
        values[perturbation_id, spec.gene_dim + program_index] = 1.0
    return values


def descriptors_for_perturbation_ids(
    perturbation_id: torch.Tensor,
    descriptor_matrix: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    matrix = torch.as_tensor(descriptor_matrix, dtype=torch.float32, device=perturbation_id.device)
    return matrix[perturbation_id.to(dtype=torch.long)]


def heldout_descriptor_coverage(dataset: SyntheticBiologyLiteDataset, descriptor_matrix: np.ndarray) -> dict[str, float]:
    heldout = tuple(int(value) for value in dataset.config.heldout_perturbations)
    if not heldout:
        return {
            "heldout_count": 0.0,
            "heldout_descriptor_coverage": 1.0,
            "heldout_nonzero_descriptor_fraction": 1.0,
        }
    rows = descriptor_matrix[np.asarray(heldout, dtype=int)]
    nonzero = rows.sum(axis=1) > 0.0
    finite = np.isfinite(rows).all(axis=1)
    return {
        "heldout_count": float(len(heldout)),
        "heldout_descriptor_coverage": float(finite.mean()),
        "heldout_nonzero_descriptor_fraction": float(nonzero.mean()),
    }
