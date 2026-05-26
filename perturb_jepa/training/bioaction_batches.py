from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset


@dataclass
class BioActionConditionBatch:
    # source/control observation
    control_gene_ids: torch.Tensor | None
    control_expression_values: torch.Tensor | None
    control_counts: torch.Tensor | None
    control_images: torch.Tensor | None
    control_rna_bag_mask: torch.Tensor | None
    control_image_bag_mask: torch.Tensor | None

    # same-condition or target/perturbed observation
    target_gene_ids: torch.Tensor | None
    target_expression_values: torch.Tensor | None
    target_counts: torch.Tensor | None
    target_images: torch.Tensor | None
    target_rna_bag_mask: torch.Tensor | None
    target_image_bag_mask: torch.Tensor | None

    # metadata/action
    perturbation_id: torch.Tensor
    perturbation_type_id: torch.Tensor
    cell_line_id: torch.Tensor
    batch_id: torch.Tensor
    dose: torch.Tensor
    time: torch.Tensor
    descriptor: torch.Tensor | None = None

    # evaluation labels only; never numeric model features
    condition_key: list[str] | None = None
    biological_key: list[tuple[int, int, float, float]] | None = None
    split: list[str] | None = None

    def to_device(self, device: torch.device | str) -> "BioActionConditionBatch":
        return BioActionConditionBatch(
            control_gene_ids=_to_device(self.control_gene_ids, device),
            control_expression_values=_to_device(self.control_expression_values, device),
            control_counts=_to_device(self.control_counts, device),
            control_images=_to_device(self.control_images, device),
            control_rna_bag_mask=_to_device(self.control_rna_bag_mask, device),
            control_image_bag_mask=_to_device(self.control_image_bag_mask, device),
            target_gene_ids=_to_device(self.target_gene_ids, device),
            target_expression_values=_to_device(self.target_expression_values, device),
            target_counts=_to_device(self.target_counts, device),
            target_images=_to_device(self.target_images, device),
            target_rna_bag_mask=_to_device(self.target_rna_bag_mask, device),
            target_image_bag_mask=_to_device(self.target_image_bag_mask, device),
            perturbation_id=self.perturbation_id.to(device),
            perturbation_type_id=self.perturbation_type_id.to(device),
            cell_line_id=self.cell_line_id.to(device),
            batch_id=self.batch_id.to(device),
            dose=self.dose.to(device),
            time=self.time.to(device),
            descriptor=_to_device(self.descriptor, device),
            condition_key=self.condition_key,
            biological_key=self.biological_key,
            split=self.split,
        )


def iter_bioaction_condition_batches(
    dataset: SyntheticBiologyLiteDataset,
    *,
    split: str = "train",
    batch_size: int = 4,
    bag_size: int | None = None,
    steps: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    modality_dropout: float = 0.0,
    rna_only: bool = False,
    image_only: bool = False,
    paired: bool = True,
    transition_only: bool = False,
) -> Iterator[BioActionConditionBatch]:
    """Yield leakage-safe control->perturbed condition-pair batches.

    The loader samples target groups from `split` and pairs each target with a
    control condition matched on cell line, dose, time, and preferably batch.
    For held-out perturbation evaluation there may be no control rows in the
    held-out split, so controls fall back to train split while target rows stay
    in the requested eval split. For train batches, both target and fallback
    controls are train-only.
    """

    if rna_only and image_only:
        raise ValueError("rna_only and image_only cannot both be true")
    if modality_dropout < 0.0 or modality_dropout > 1.0:
        raise ValueError("modality_dropout must be in [0, 1]")

    rng = np.random.default_rng(seed)
    bag_size = int(bag_size or dataset.config.cells_per_condition)
    pairs = condition_pair_records(dataset, split=split)
    if not pairs:
        raise ValueError(f"split {split!r} has no non-control target pairs")

    order = np.arange(len(pairs))
    emitted = 0
    while steps is None or emitted < steps:
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            if steps is not None and emitted >= steps:
                return
            selected = [pairs[int(idx)] for idx in order[start : start + batch_size]]
            batch = make_bioaction_condition_batch(
                dataset,
                selected,
                bag_size=bag_size,
                rng=rng,
                device=device,
            )
            if transition_only:
                yield batch
            else:
                yield apply_modality_options(
                    batch,
                    rng=rng,
                    modality_dropout=modality_dropout,
                    rna_only=rna_only,
                    image_only=image_only,
                    paired=paired,
                )
            emitted += 1
        if steps is None:
            return


def condition_pair_records(dataset: SyntheticBiologyLiteDataset, *, split: str) -> list[dict[str, object]]:
    groups = _condition_groups_with_metadata(dataset)
    target_groups = [row for row in groups if row["split"] == split and int(row["perturbation_id"]) != dataset.config.control_perturbation_id]
    records: list[dict[str, object]] = []
    for target in target_groups:
        control = _find_control_group(dataset, groups, target=target, split=split)
        if control is None:
            continue
        records.append(
            {
                "control_indices": control["indices"],
                "target_indices": target["indices"],
                "perturbation_id": int(target["perturbation_id"]),
                "perturbation_type_id": int(target["perturbation_type_id"]),
                "cell_line_id": int(target["cell_line_id"]),
                "batch_id": int(target["batch_id"]),
                "dose": float(target["dose"]),
                "time": float(target["time"]),
                "condition_key": str(target["condition_key"]),
                "biological_key": (
                    int(target["perturbation_id"]),
                    int(target["cell_line_id"]),
                    float(target["dose"]),
                    float(target["time"]),
                ),
                "split": str(target["split"]),
                "control_split": str(control["split"]),
            }
        )
    return records


def make_bioaction_condition_batch(
    dataset: SyntheticBiologyLiteDataset,
    records: list[dict[str, object]],
    *,
    bag_size: int,
    rng: np.random.Generator,
    device: torch.device | str,
) -> BioActionConditionBatch:
    if not records:
        raise ValueError("records must not be empty")
    control_index = _sample_indices([np.asarray(record["control_indices"], dtype=int) for record in records], bag_size, rng)
    target_index = _sample_indices([np.asarray(record["target_indices"], dtype=int) for record in records], bag_size, rng)
    batch = len(records)
    gene_ids = np.broadcast_to(dataset.gene_ids, (batch, bag_size, dataset.gene_ids.size)).copy()
    ones_mask = torch.ones((batch, bag_size), dtype=torch.bool, device=device)
    return BioActionConditionBatch(
        control_gene_ids=torch.as_tensor(gene_ids, dtype=torch.long, device=device),
        control_expression_values=torch.as_tensor(dataset.expression_values[control_index], dtype=torch.float32, device=device),
        control_counts=torch.as_tensor(dataset.observed_counts[control_index], dtype=torch.float32, device=device),
        control_images=torch.as_tensor(dataset.images[control_index], dtype=torch.float32, device=device),
        control_rna_bag_mask=ones_mask,
        control_image_bag_mask=ones_mask,
        target_gene_ids=torch.as_tensor(gene_ids, dtype=torch.long, device=device),
        target_expression_values=torch.as_tensor(dataset.expression_values[target_index], dtype=torch.float32, device=device),
        target_counts=torch.as_tensor(dataset.observed_counts[target_index], dtype=torch.float32, device=device),
        target_images=torch.as_tensor(dataset.images[target_index], dtype=torch.float32, device=device),
        target_rna_bag_mask=ones_mask.clone(),
        target_image_bag_mask=ones_mask.clone(),
        perturbation_id=torch.as_tensor([record["perturbation_id"] for record in records], dtype=torch.long, device=device),
        perturbation_type_id=torch.as_tensor([record["perturbation_type_id"] for record in records], dtype=torch.long, device=device),
        cell_line_id=torch.as_tensor([record["cell_line_id"] for record in records], dtype=torch.long, device=device),
        batch_id=torch.as_tensor([record["batch_id"] for record in records], dtype=torch.long, device=device),
        dose=torch.as_tensor([record["dose"] for record in records], dtype=torch.float32, device=device),
        time=torch.as_tensor([record["time"] for record in records], dtype=torch.float32, device=device),
        descriptor=None,
        condition_key=[str(record["condition_key"]) for record in records],
        biological_key=[record["biological_key"] for record in records],  # type: ignore[list-item]
        split=[str(record["split"]) for record in records],
    )


def apply_modality_options(
    batch: BioActionConditionBatch,
    *,
    rng: np.random.Generator,
    modality_dropout: float = 0.0,
    rna_only: bool = False,
    image_only: bool = False,
    paired: bool = True,
) -> BioActionConditionBatch:
    if rna_only:
        return _drop_image(batch)
    if image_only:
        return _drop_rna(batch)
    if not paired:
        return _drop_image(batch) if rng.random() < 0.5 else _drop_rna(batch)
    if modality_dropout <= 0.0 or rng.random() >= modality_dropout:
        return batch
    return _drop_image(batch) if rng.random() < 0.5 else _drop_rna(batch)


def exact_train_key_fraction(dataset: SyntheticBiologyLiteDataset, *, eval_split: str) -> float:
    train = {
        _biological_key(row)
        for row in _condition_groups_with_metadata(dataset)
        if row["split"] == "train" and int(row["perturbation_id"]) != dataset.config.control_perturbation_id
    }
    eval_keys = [
        _biological_key(row)
        for row in _condition_groups_with_metadata(dataset)
        if row["split"] == eval_split and int(row["perturbation_id"]) != dataset.config.control_perturbation_id
    ]
    if not eval_keys:
        return 0.0
    return float(sum(key in train for key in eval_keys) / len(eval_keys))


def _condition_groups_with_metadata(dataset: SyntheticBiologyLiteDataset) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, group in dataset.metadata.groupby(["split", "bag_key"], sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "indices": group.index.to_numpy(dtype=int),
                "split": str(first["split"]),
                "condition_key": str(first["condition_key"]),
                "perturbation_id": int(first["perturbation_id"]),
                "perturbation_type_id": int(first["perturbation_type_id"]),
                "cell_line_id": int(first["cell_line_id"]),
                "batch_id": int(first["batch_id"]),
                "dose": float(first["dose"]),
                "time": float(first["time"]),
            }
        )
    return rows


def _find_control_group(
    dataset: SyntheticBiologyLiteDataset,
    groups: list[dict[str, object]],
    *,
    target: dict[str, object],
    split: str,
) -> dict[str, object] | None:
    def matches(row: dict[str, object], *, require_batch: bool, allowed_split: str | None) -> bool:
        if int(row["perturbation_id"]) != dataset.config.control_perturbation_id:
            return False
        if allowed_split is not None and row["split"] != allowed_split:
            return False
        if int(row["cell_line_id"]) != int(target["cell_line_id"]):
            return False
        if abs(float(row["dose"]) - float(target["dose"])) > 1e-8:
            return False
        if abs(float(row["time"]) - float(target["time"])) > 1e-8:
            return False
        if require_batch and int(row["batch_id"]) != int(target["batch_id"]):
            return False
        return True

    # Use train controls for train. For held-out perturbation evaluation, a
    # same-split control usually does not exist, so train controls are allowed
    # as source observations while target rows remain held out.
    split_candidates = (split, "train") if split != "train" else ("train",)
    for allowed_split in split_candidates:
        for require_batch in (True, False):
            for row in groups:
                if matches(row, require_batch=require_batch, allowed_split=allowed_split):
                    return row
    return None


def _sample_indices(groups: list[np.ndarray], bag_size: int, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for group in groups:
        replace = group.size < bag_size
        sampled.append(rng.choice(group, size=bag_size, replace=replace))
    return np.stack(sampled, axis=0)


def _drop_rna(batch: BioActionConditionBatch) -> BioActionConditionBatch:
    batch.control_gene_ids = None
    batch.control_expression_values = None
    batch.control_counts = None
    batch.control_rna_bag_mask = None
    batch.target_gene_ids = None
    batch.target_expression_values = None
    batch.target_counts = None
    batch.target_rna_bag_mask = None
    return batch


def _drop_image(batch: BioActionConditionBatch) -> BioActionConditionBatch:
    batch.control_images = None
    batch.control_image_bag_mask = None
    batch.target_images = None
    batch.target_image_bag_mask = None
    return batch


def _to_device(value: torch.Tensor | None, device: torch.device | str) -> torch.Tensor | None:
    return None if value is None else value.to(device)


def _biological_key(row: dict[str, object] | pd.Series) -> tuple[int, int, float, float]:
    return (
        int(row["perturbation_id"]),
        int(row["cell_line_id"]),
        float(row["dose"]),
        float(row["time"]),
    )

