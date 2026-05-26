from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from perturb_jepa.data.norman2019 import (
    NormanDataset,
    assert_no_combo_order_leakage,
    canonical_condition_key,
    is_combo,
    is_single,
    perturbation_genes,
)
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class NormanBioTechSpec:
    gene_indices: tuple[int, ...]
    action_genes: tuple[str, ...]
    condition_to_action_id: dict[str, int]

    @property
    def num_genes(self) -> int:
        return len(self.gene_indices)

    @property
    def descriptor_dim(self) -> int:
        return len(self.action_genes)

    @property
    def num_actions(self) -> int:
        return max(1, len(self.condition_to_action_id))


def build_norman_biotech_spec(dataset: NormanDataset, *, gene_count: int = 512) -> NormanBioTechSpec:
    if dataset.split is None:
        raise ValueError("Norman dataset must have a GEARS split before building BioTech batches")
    assert_no_combo_order_leakage(dataset.split)
    train_indices = [dataset.condition_to_index[condition] for condition in dataset.split.train_conditions if condition in dataset.condition_to_index]
    train_values = dataset.condition_means[np.asarray(train_indices, dtype=int)]
    variances = np.nan_to_num(train_values, nan=0.0).var(axis=0)
    count = min(int(gene_count), int(variances.shape[0]))
    gene_indices = tuple(int(value) for value in np.argsort(variances)[::-1][:count])
    action_genes = tuple(
        sorted(
            {
                gene
                for condition in dataset.conditions
                for gene in perturbation_genes(condition)
                if gene != "ctrl"
            }
        )
    )
    canonical_keys = sorted({canonical_condition_key(condition) for condition in dataset.conditions if condition != dataset.ctrl_condition})
    condition_to_action_id = {
        condition: int(canonical_keys.index(canonical_condition_key(condition)))
        for condition in dataset.conditions
        if condition != dataset.ctrl_condition
    }
    return NormanBioTechSpec(gene_indices=gene_indices, action_genes=action_genes, condition_to_action_id=condition_to_action_id)


def iter_norman_biotech_condition_batches(
    dataset: NormanDataset,
    spec: NormanBioTechSpec,
    *,
    split: str = "train",
    batch_size: int = 4,
    steps: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Iterator[BioActionConditionBatch]:
    if dataset.split is None:
        raise ValueError("Norman dataset must have a split")
    rng = np.random.default_rng(seed)
    conditions = [condition for condition in _conditions_for_split(dataset, split) if condition != dataset.ctrl_condition]
    if not conditions:
        raise ValueError(f"Norman split {split!r} has no non-control target conditions")
    order = np.arange(len(conditions))
    gene_ids = np.arange(spec.num_genes, dtype=np.int64)
    emitted = 0
    while steps is None or emitted < steps:
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            if steps is not None and emitted >= steps:
                return
            selected = [conditions[int(index)] for index in order[start : start + int(batch_size)]]
            yield make_norman_biotech_condition_batch(dataset, spec, selected, split=split, gene_ids=gene_ids, device=device)
            emitted += 1
        if steps is None:
            return


def make_norman_biotech_condition_batch(
    dataset: NormanDataset,
    spec: NormanBioTechSpec,
    conditions: list[str],
    *,
    split: str,
    gene_ids: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> BioActionConditionBatch:
    if not conditions:
        raise ValueError("conditions must not be empty")
    genes = np.asarray(spec.gene_indices, dtype=np.int64)
    batch = len(conditions)
    ids = np.asarray(gene_ids if gene_ids is not None else np.arange(spec.num_genes, dtype=np.int64), dtype=np.int64)
    ids = np.broadcast_to(ids, (batch, 1, ids.size)).copy()
    control_values = np.broadcast_to(dataset.ctrl_mean[genes], (batch, 1, genes.size)).copy()
    target_values = np.stack([dataset.mean_for(condition)[genes] for condition in conditions], axis=0)[:, None, :]
    control_counts = np.broadcast_to(dataset.ctrl_count_mean[genes], (batch, 1, genes.size)).copy()
    target_counts = np.stack([dataset.count_means[dataset.condition_to_index[condition], genes] for condition in conditions], axis=0)[:, None, :]
    mask = torch.ones((batch, 1), dtype=torch.bool, device=device)
    descriptors = np.stack([_descriptor_for_condition(condition, spec) for condition in conditions], axis=0)
    perturbation_type_id = [2 if is_combo(condition) else 1 if is_single(condition) else 0 for condition in conditions]
    biological_key = [tuple(spec.action_genes[index] for index, value in enumerate(descriptor) if value > 0.0) for descriptor in descriptors]
    return BioActionConditionBatch(
        control_gene_ids=torch.as_tensor(ids, dtype=torch.long, device=device),
        control_expression_values=torch.as_tensor(control_values, dtype=torch.float32, device=device),
        control_counts=torch.as_tensor(control_counts, dtype=torch.float32, device=device),
        control_images=None,
        control_rna_bag_mask=mask,
        control_image_bag_mask=None,
        target_gene_ids=torch.as_tensor(ids.copy(), dtype=torch.long, device=device),
        target_expression_values=torch.as_tensor(target_values, dtype=torch.float32, device=device),
        target_counts=torch.as_tensor(target_counts, dtype=torch.float32, device=device),
        target_images=None,
        target_rna_bag_mask=mask.clone(),
        target_image_bag_mask=None,
        perturbation_id=torch.as_tensor([spec.condition_to_action_id[condition] for condition in conditions], dtype=torch.long, device=device),
        perturbation_type_id=torch.as_tensor(perturbation_type_id, dtype=torch.long, device=device),
        cell_line_id=torch.zeros(batch, dtype=torch.long, device=device),
        batch_id=torch.zeros(batch, dtype=torch.long, device=device),
        dose=torch.ones(batch, dtype=torch.float32, device=device),
        time=torch.zeros(batch, dtype=torch.float32, device=device),
        descriptor=torch.as_tensor(descriptors, dtype=torch.float32, device=device),
        condition_key=list(conditions),
        biological_key=biological_key,  # type: ignore[arg-type]
        split=[split for _ in conditions],
    )


def _conditions_for_split(dataset: NormanDataset, split: str) -> tuple[str, ...]:
    if dataset.split is None:
        raise ValueError("Norman dataset must have a split")
    if split == "train":
        return dataset.split.train_conditions
    if split == "val":
        return dataset.split.val_conditions
    if split == "test":
        return dataset.split.test_conditions
    if split.startswith("test_"):
        subgroup = split.removeprefix("test_")
        if subgroup in dataset.split.test_subgroups:
            return dataset.split.test_subgroups[subgroup]
    if split.startswith("val_"):
        subgroup = split.removeprefix("val_")
        if subgroup in dataset.split.val_subgroups:
            return dataset.split.val_subgroups[subgroup]
    raise ValueError(f"unsupported Norman split: {split}")


def _descriptor_for_condition(condition: str, spec: NormanBioTechSpec) -> np.ndarray:
    values = np.zeros(spec.descriptor_dim, dtype=np.float32)
    gene_to_index = {gene: index for index, gene in enumerate(spec.action_genes)}
    for gene in perturbation_genes(condition):
        index = gene_to_index.get(gene)
        if index is not None:
            values[index] = 1.0
    return values
