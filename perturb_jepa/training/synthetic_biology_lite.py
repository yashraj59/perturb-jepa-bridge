from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from perturb_jepa.training.synthetic import SyntheticBridgeBatch


@dataclass(frozen=True)
class SyntheticBiologyLiteConfig:
    name: str = "synth_micro"
    seed: int = 0
    decoder_seed: int = 1729
    num_perturbations: int = 4
    num_cell_lines: int = 2
    doses: tuple[float, ...] = (0.0, 1.0)
    num_batches: int = 2
    cells_per_condition: int = 8
    genes: int = 128
    latent_dim: int = 16
    tech_dim: int = 4
    num_programs: int = 8
    image_channels: int = 1
    image_size: int = 16
    patch_size: int = 4
    biological_noise_std: float = 0.08
    technical_noise_std: float = 0.05
    count_depth: float = 40.0
    overdispersion: float = 0.25
    dropout_base: float = -1.0
    heldout_perturbations: tuple[int, ...] = ()
    heldout_doses: tuple[float, ...] = ()
    heldout_batches: tuple[int, ...] = ()
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    control_perturbation_id: int = 0
    batch_confounding: bool = False
    perturbation_effect_mode: str = "random"
    program_effect_strength: float = 0.8
    perturbation_idiosyncratic_std: float = 0.12

    def __post_init__(self) -> None:
        if self.control_perturbation_id < 0 or self.control_perturbation_id >= self.num_perturbations:
            raise ValueError("control_perturbation_id must be within perturbation range")
        if self.num_programs <= 0:
            raise ValueError("num_programs must be positive")
        if self.genes > 512:
            raise ValueError("synthetic lite generator is capped at 512 genes")
        if self.cells_per_condition > 16:
            raise ValueError("synthetic lite generator is capped at 16 cells per condition")
        if self.image_size > 24:
            raise ValueError("synthetic lite generator is capped at 24x24 images")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not self.doses:
            raise ValueError("at least one dose is required")
        if self.train_fraction <= 0.0 or self.val_fraction < 0.0 or self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train/val fractions must leave a positive test split")
        if self.perturbation_effect_mode not in {"random", "program_aligned"}:
            raise ValueError("perturbation_effect_mode must be 'random' or 'program_aligned'")
        if self.program_effect_strength <= 0.0:
            raise ValueError("program_effect_strength must be positive")
        if self.perturbation_idiosyncratic_std < 0.0:
            raise ValueError("perturbation_idiosyncratic_std must be non-negative")
        object.__setattr__(self, "doses", tuple(float(dose) for dose in self.doses))
        object.__setattr__(self, "heldout_perturbations", tuple(int(value) for value in self.heldout_perturbations))
        object.__setattr__(self, "heldout_doses", tuple(float(value) for value in self.heldout_doses))
        object.__setattr__(self, "heldout_batches", tuple(int(value) for value in self.heldout_batches))


@dataclass
class SyntheticBiologyLiteDataset:
    config: SyntheticBiologyLiteConfig
    gene_ids: np.ndarray
    expression_values: np.ndarray
    observed_counts: np.ndarray
    clean_rna: np.ndarray
    images: np.ndarray
    z_bio: np.ndarray
    z_tech: np.ndarray
    metadata: pd.DataFrame
    perturbation_directions: np.ndarray
    cell_line_baselines: np.ndarray
    interactions: np.ndarray
    batch_offsets: np.ndarray
    gene_program_assignment: np.ndarray
    dose_curve_by_dose: dict[str, float]

    def condition_frame(self, split: str | None = None) -> pd.DataFrame:
        frame = self.metadata if split is None else self.metadata[self.metadata["split"] == split]
        columns = [
            "condition_id",
            "condition_key",
            "perturbation",
            "perturbation_id",
            "dose",
            "cell_line",
            "cell_line_id",
            "batch",
            "batch_id",
            "time",
        ]
        return frame.loc[:, columns].drop_duplicates().reset_index(drop=True)

    def condition_bag_indices(self, *, split: str) -> list[np.ndarray]:
        return self._condition_groups(split=split)

    def metadata_for_condition_bags(self, *, split: str) -> pd.DataFrame:
        rows = []
        for group in self._condition_groups(split=split):
            rows.append(self.metadata.iloc[int(group[0])])
        return pd.DataFrame(rows).reset_index(drop=True)

    def iter_condition_batches(
        self,
        *,
        split: str = "train",
        batch_size: int = 8,
        bag_size: int | None = None,
        steps: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        rna_mask_prob: float = 0.2,
        image_mask_prob: float = 0.2,
        device: torch.device | str = "cpu",
    ) -> Iterator[SyntheticBridgeBatch]:
        bag_size = int(bag_size or self.config.cells_per_condition)
        groups = self._condition_groups(split=split)
        if not groups:
            raise ValueError(f"split {split!r} has no condition groups")
        rng = np.random.default_rng(seed)
        order = np.arange(len(groups))
        emitted = 0
        while steps is None or emitted < steps:
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                if steps is not None and emitted >= steps:
                    return
                selected_groups = [groups[int(idx)] for idx in order[start : start + batch_size]]
                yield self._make_bridge_batch(
                    selected_groups,
                    bag_size=bag_size,
                    rng=rng,
                    rna_mask_prob=rna_mask_prob,
                    image_mask_prob=image_mask_prob,
                    device=device,
                )
                emitted += 1
            if not shuffle and steps is None:
                return

    def export(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "generation_config.json").write_text(
            json.dumps(_jsonable(asdict(self.config)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ground_truth = {
            "latent_biological_state": self.z_bio.tolist(),
            "latent_technical_state": self.z_tech.tolist(),
            "perturbation_direction_matrix": self.perturbation_directions.tolist(),
            "cell_line_baseline_matrix": self.cell_line_baselines.tolist(),
            "interaction_tensor": self.interactions.tolist(),
            "dose_curve": self.dose_curve_by_dose,
            "gene_program_assignment": self.gene_program_assignment.tolist(),
            "batch_offsets": self.batch_offsets.tolist(),
            "clean_rna_before_technical_corruption": self.clean_rna.tolist(),
            "observed_rna_after_technical_corruption": self.expression_values.tolist(),
            "observed_counts_after_technical_corruption": self.observed_counts.tolist(),
            "split_labels": self.metadata["split"].tolist(),
            "cell_ids": self.metadata["cell_id"].tolist(),
            "condition_keys": self.metadata["condition_key"].tolist(),
        }
        (output_path / "ground_truth.json").write_text(
            json.dumps(_jsonable(ground_truth), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.metadata.to_csv(output_path / "metadata.tsv", sep="\t", index=False)
        return output_path

    def _condition_groups(self, *, split: str) -> list[np.ndarray]:
        frame = self.metadata[self.metadata["split"] == split]
        groups: list[np.ndarray] = []
        for _, group in frame.groupby("bag_key", sort=True):
            indices = group.index.to_numpy(dtype=int)
            if indices.size:
                groups.append(indices)
        return groups

    def _make_bridge_batch(
        self,
        selected_groups: Sequence[np.ndarray],
        *,
        bag_size: int,
        rng: np.random.Generator,
        rna_mask_prob: float,
        image_mask_prob: float,
        device: torch.device | str,
    ) -> SyntheticBridgeBatch:
        sampled_indices = []
        for group in selected_groups:
            replace = group.size < bag_size
            sampled_indices.append(rng.choice(group, size=bag_size, replace=replace))
        index = np.stack(sampled_indices, axis=0)
        first = index[:, 0]
        frame = self.metadata.iloc[first]
        gene_ids = np.broadcast_to(self.gene_ids, (len(selected_groups), bag_size, self.gene_ids.size))
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        return SyntheticBridgeBatch(
            gene_ids=torch.as_tensor(gene_ids.copy(), dtype=torch.long, device=device),
            expression_values=torch.as_tensor(self.expression_values[index], dtype=torch.float32, device=device),
            rna_token_mask=torch.rand((len(selected_groups), bag_size, self.config.genes), device=device) < rna_mask_prob,
            images=torch.as_tensor(self.images[index], dtype=torch.float32, device=device),
            image_patch_mask=torch.rand((len(selected_groups), bag_size, num_patches), device=device) < image_mask_prob,
            perturbation_id=torch.as_tensor(frame["perturbation_id"].to_numpy(), dtype=torch.long, device=device),
            perturbation_type_id=torch.as_tensor(frame["perturbation_type_id"].to_numpy(), dtype=torch.long, device=device),
            cell_line_id=torch.as_tensor(frame["cell_line_id"].to_numpy(), dtype=torch.long, device=device),
            batch_id=torch.as_tensor(frame["batch_id"].to_numpy(), dtype=torch.long, device=device),
            dose=torch.as_tensor(frame["dose"].to_numpy(dtype=float), dtype=torch.float32, device=device),
            time=torch.as_tensor(frame["time"].to_numpy(dtype=float), dtype=torch.float32, device=device),
        )


def generate_synthetic_biology_lite(config: SyntheticBiologyLiteConfig) -> SyntheticBiologyLiteDataset:
    rng = np.random.default_rng(config.seed)
    decoder_rng = np.random.default_rng(config.decoder_seed)
    gene_ids = np.arange(config.genes, dtype=np.int64)
    gene_program_assignment = np.arange(config.genes, dtype=np.int64) % config.num_programs
    decoder_rng.shuffle(gene_program_assignment)

    program_loadings = decoder_rng.normal(0.0, 0.08, size=(config.num_programs, config.genes)).astype(np.float32)
    program_loadings[gene_program_assignment, np.arange(config.genes)] += decoder_rng.uniform(0.8, 1.2, size=config.genes)
    bio_to_program = decoder_rng.normal(0.0, 0.35, size=(config.latent_dim, config.num_programs)).astype(np.float32)
    rna_decoder = bio_to_program @ program_loadings
    tech_decoder = decoder_rng.normal(0.0, 0.12, size=(config.tech_dim, config.genes)).astype(np.float32)
    gene_bias = decoder_rng.normal(0.0, 0.25, size=config.genes).astype(np.float32)

    image_pixels = config.image_channels * config.image_size * config.image_size
    image_decoder = decoder_rng.normal(0.0, 0.25, size=(config.latent_dim, image_pixels)).astype(np.float32)
    image_tech_decoder = decoder_rng.normal(0.0, 0.08, size=(config.tech_dim, image_pixels)).astype(np.float32)
    image_bias = decoder_rng.normal(0.0, 0.1, size=image_pixels).astype(np.float32)

    cell_line_baselines = rng.normal(0.0, 0.65, size=(config.num_cell_lines, config.latent_dim)).astype(np.float32)
    perturbation_directions = _perturbation_directions(config, rng)
    perturbation_directions[config.control_perturbation_id] = 0.0
    interactions = rng.normal(
        0.0,
        0.08,
        size=(config.num_cell_lines, config.num_perturbations, config.latent_dim),
    ).astype(np.float32)
    interactions[:, config.control_perturbation_id] = 0.0
    batch_offsets = rng.normal(0.0, 0.45, size=(config.num_batches, config.tech_dim)).astype(np.float32)
    library_vector = rng.normal(0.0, 0.35, size=config.tech_dim).astype(np.float32)
    dropout_vector = rng.normal(0.0, 0.25, size=config.tech_dim).astype(np.float32)
    dropout_gene_bias = decoder_rng.normal(0.0, 0.2, size=config.genes).astype(np.float32)

    rows: list[dict[str, object]] = []
    z_bio_values: list[np.ndarray] = []
    z_tech_values: list[np.ndarray] = []
    clean_values: list[np.ndarray] = []
    observed_counts: list[np.ndarray] = []
    expression_values: list[np.ndarray] = []
    image_values: list[np.ndarray] = []
    dose_curve_by_dose = {str(float(dose)): _dose_curve(float(dose)) for dose in config.doses}
    condition_id = 0

    for perturbation_id in range(config.num_perturbations):
        allowed_batches = _allowed_batches_for_perturbation(config, perturbation_id)
        for cell_line_id in range(config.num_cell_lines):
            for dose in config.doses:
                for batch_id in allowed_batches:
                    condition_key = f"pert={perturbation_id}|dose={float(dose):g}|cell={cell_line_id}"
                    bag_key = f"{condition_key}|batch={batch_id}"
                    curve = 0.0 if perturbation_id == config.control_perturbation_id else dose_curve_by_dose[str(float(dose))]
                    for cell_in_condition in range(config.cells_per_condition):
                        split = _split_for_cell(
                            config,
                            perturbation_id=perturbation_id,
                            dose=float(dose),
                            batch_id=batch_id,
                            cell_in_condition=cell_in_condition,
                        )
                        library_size_effect = rng.normal(0.0, 0.35)
                        dropout_effect = rng.normal(0.0, 0.25)
                        z_bio = (
                            cell_line_baselines[cell_line_id]
                            + curve * perturbation_directions[perturbation_id]
                            + interactions[cell_line_id, perturbation_id]
                            + rng.normal(0.0, config.biological_noise_std, size=config.latent_dim)
                        ).astype(np.float32)
                        z_tech = (
                            batch_offsets[batch_id]
                            + library_size_effect * library_vector
                            + dropout_effect * dropout_vector
                            + rng.normal(0.0, config.technical_noise_std, size=config.tech_dim)
                        ).astype(np.float32)
                        clean_log_mu = np.clip(gene_bias + z_bio @ rna_decoder, -3.5, 3.5)
                        clean_mu = np.exp(clean_log_mu).astype(np.float32)
                        technical_log_shift = np.clip(z_tech @ tech_decoder + library_size_effect, -2.5, 2.5)
                        observed_mu = np.exp(np.log(clean_mu + 1e-6) + technical_log_shift)
                        counts = _gamma_poisson(rng, observed_mu * config.count_depth, config.overdispersion)
                        dropout_logits = config.dropout_base + dropout_gene_bias - 0.35 * np.log1p(observed_mu) + dropout_effect
                        dropout_probability = _sigmoid(dropout_logits)
                        counts = counts * (rng.random(config.genes) > dropout_probability)
                        expression = np.log1p(counts).astype(np.float32)
                        image = _sigmoid(z_bio @ image_decoder + z_tech @ image_tech_decoder + image_bias)
                        image = image + rng.normal(0.0, 0.02, size=image.shape)
                        image = np.clip(image, 0.0, 1.0).reshape(
                            config.image_channels,
                            config.image_size,
                            config.image_size,
                        )

                        cell_id = f"{config.name}_cell_{len(rows):06d}"
                        rows.append(
                            {
                                "cell_id": cell_id,
                                "condition_id": condition_id,
                                "condition_key": condition_key,
                                "bag_key": bag_key,
                                "perturbation": f"perturbation_{perturbation_id}",
                                "perturbation_id": perturbation_id,
                                "perturbation_type_id": 0 if perturbation_id == config.control_perturbation_id else 1,
                                "dose": float(dose),
                                "cell_line": f"cell_line_{cell_line_id}",
                                "cell_line_id": cell_line_id,
                                "batch": f"batch_{batch_id}",
                                "batch_id": batch_id,
                                "time": 0.0,
                                "split": split,
                            }
                        )
                        z_bio_values.append(z_bio)
                        z_tech_values.append(z_tech)
                        clean_values.append(np.log1p(clean_mu * config.count_depth).astype(np.float32))
                        observed_counts.append(counts.astype(np.float32))
                        expression_values.append(expression)
                        image_values.append(image.astype(np.float32))
                    condition_id += 1

    metadata = pd.DataFrame(rows)
    return SyntheticBiologyLiteDataset(
        config=config,
        gene_ids=gene_ids,
        expression_values=np.stack(expression_values).astype(np.float32),
        observed_counts=np.stack(observed_counts).astype(np.float32),
        clean_rna=np.stack(clean_values).astype(np.float32),
        images=np.stack(image_values).astype(np.float32),
        z_bio=np.stack(z_bio_values).astype(np.float32),
        z_tech=np.stack(z_tech_values).astype(np.float32),
        metadata=metadata,
        perturbation_directions=perturbation_directions,
        cell_line_baselines=cell_line_baselines,
        interactions=interactions,
        batch_offsets=batch_offsets,
        gene_program_assignment=gene_program_assignment,
        dose_curve_by_dose=dose_curve_by_dose,
    )


def synthetic_lite_config(name: str, *, seed: int = 0) -> SyntheticBiologyLiteConfig:
    if name == "synth_micro":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=4,
            num_cell_lines=2,
            doses=(0.0, 1.0),
            num_batches=2,
            cells_per_condition=8,
            genes=128,
        )
    if name == "synth_easy_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=8,
            num_cell_lines=3,
            doses=(0.0, 1.0),
            num_batches=2,
            cells_per_condition=12,
            genes=256,
        )
    if name == "synth_medium_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
        )
    if name == "synth_heldout_perturbation_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_perturbations=(9, 10, 11),
        )
    if name == "synth_genetic_anchor_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(1.0,),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_perturbations=(9, 10, 11),
        )
    if name == "synth_program_aligned_genetic_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(1.0,),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            num_programs=4,
            heldout_perturbations=(9, 10, 11),
            perturbation_effect_mode="program_aligned",
            program_effect_strength=0.8,
            perturbation_idiosyncratic_std=0.08,
        )
    if name == "synth_program_aligned_extrapolative_holdout_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(1.0,),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            num_programs=4,
            heldout_perturbations=(9, 10, 11),
            perturbation_effect_mode="program_aligned",
            program_effect_strength=0.8,
            perturbation_idiosyncratic_std=0.08,
        )
    if name == "synth_program_aligned_random_holdout_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(1.0,),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            num_programs=4,
            heldout_perturbations=(1, 4, 11),
            perturbation_effect_mode="program_aligned",
            program_effect_strength=0.8,
            perturbation_idiosyncratic_std=0.08,
        )
    if name == "synth_program_aligned_stratified_holdout_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(1.0,),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            num_programs=4,
            heldout_perturbations=(5, 6, 7),
            perturbation_effect_mode="program_aligned",
            program_effect_strength=0.8,
            perturbation_idiosyncratic_std=0.08,
        )
    if name == "synth_batch_confound_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            batch_confounding=True,
        )
    if name == "synth_dose_extrapolation_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 2.0, 3.0, 5.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_doses=(2.0, 5.0),
        )
    if name == "synth_chemical_dose_anchor_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 0.3, 1.0, 3.0, 10.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_doses=(3.0, 10.0),
        )
    raise ValueError(f"unknown synthetic lite dataset: {name}")


def _perturbation_directions(config: SyntheticBiologyLiteConfig, rng: np.random.Generator) -> np.ndarray:
    if config.perturbation_effect_mode == "random":
        return rng.normal(0.0, 0.8, size=(config.num_perturbations, config.latent_dim)).astype(np.float32)
    program_directions = rng.normal(
        0.0,
        float(config.program_effect_strength),
        size=(config.num_programs, config.latent_dim),
    ).astype(np.float32)
    perturbation_noise = rng.normal(
        0.0,
        float(config.perturbation_idiosyncratic_std),
        size=(config.num_perturbations, config.latent_dim),
    ).astype(np.float32)
    program_ids = np.arange(config.num_perturbations, dtype=np.int64) % config.num_programs
    directions = program_directions[program_ids] + perturbation_noise
    directions[config.control_perturbation_id] = 0.0
    return directions.astype(np.float32)


def _allowed_batches_for_perturbation(config: SyntheticBiologyLiteConfig, perturbation_id: int) -> tuple[int, ...]:
    if not config.batch_confounding or perturbation_id == config.control_perturbation_id:
        return tuple(range(config.num_batches))
    return (perturbation_id % config.num_batches,)


def _split_for_cell(
    config: SyntheticBiologyLiteConfig,
    *,
    perturbation_id: int,
    dose: float,
    batch_id: int,
    cell_in_condition: int,
) -> str:
    if perturbation_id in config.heldout_perturbations:
        return "test_heldout_perturbation"
    if any(abs(dose - heldout) < 1e-8 for heldout in config.heldout_doses):
        return "test_heldout_dose"
    if batch_id in config.heldout_batches:
        return "test_heldout_batch"
    fraction = (cell_in_condition + 0.5) / config.cells_per_condition
    if fraction <= config.train_fraction:
        return "train"
    if fraction <= config.train_fraction + config.val_fraction:
        return "val"
    return "test"


def _dose_curve(dose: float) -> float:
    dose = max(float(dose), 0.0)
    return float(dose / (1.0 + dose))


def _gamma_poisson(rng: np.random.Generator, mean: np.ndarray, overdispersion: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    if overdispersion <= 0.0:
        return rng.poisson(mean).astype(np.float32)
    shape = 1.0 / overdispersion
    scale = np.maximum(mean * overdispersion, 1e-6)
    gamma_mean = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(gamma_mean).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
