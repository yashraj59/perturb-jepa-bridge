from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.conditions import MetadataVocab
from perturb_jepa.data.schema import DEFAULT_METADATA_SCHEMA, make_condition_id, normalize_value
from perturb_jepa.data.splits import (
    grouped_hash_split,
    heldout_batch_split,
    heldout_cell_line_split,
    heldout_dose_time_split,
    heldout_moa_split,
    heldout_perturbation_split,
)
from perturb_jepa.data.scrna import as_dense_array, library_size_log1p, normalize_scrna_obs, normalize_scrna_var, select_high_variance_genes
from perturb_jepa.models.bridge import PerturbJEPABridgeConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig


def load_yaml_or_json_config(path: Path | None) -> tuple[ExperimentConfig, dict[str, Any]]:
    if path is None:
        config = ExperimentConfig.smoke()
        return config, config.to_dict()
    if path.suffix.lower() == ".json":
        config = ExperimentConfig.load_json(path)
        return config, config.to_dict()
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig.from_dict(raw), dict(raw)


def raw_get(raw: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    value: Any = raw
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def read_h5ad_subset(
    path: str | Path,
    *,
    max_cells: int | None = None,
    seed: int = 0,
):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install the data extra to read h5ad files: pip install -e '.[data]'") from exc
    adata = ad.read_h5ad(path)
    if max_cells is not None and max_cells > 0 and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        adata = adata[indices].copy()
    adata.obs = normalize_scrna_obs(pd.DataFrame(adata.obs).copy())
    adata.var = normalize_scrna_var(pd.DataFrame(adata.var).copy())
    return adata


def prepare_expression_matrix(
    matrix,
    *,
    n_top_genes: int | None,
    max_genes: int | None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    values = library_size_log1p(matrix) if normalize else as_dense_array(matrix).astype(np.float32, copy=False)
    if n_top_genes is None:
        n_top_genes = max_genes
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= values.shape[1]:
        gene_indices = np.arange(values.shape[1], dtype=np.int64)
    else:
        gene_indices = select_high_variance_genes(values, n_top=n_top_genes).astype(np.int64, copy=False)
    if max_genes is not None and max_genes > 0:
        gene_indices = gene_indices[:max_genes]
    selected = values[:, gene_indices].astype(np.float32, copy=False)
    token_ids = np.arange(selected.shape[1], dtype=np.int64)
    return selected, token_ids


def override_bridge_config_for_real_data(
    config: ExperimentConfig,
    *,
    num_genes: int | None = None,
    max_genes: int | None = None,
    image_channels: int | None = None,
    image_size: int | None = None,
    metadata_vocab: MetadataVocab | None = None,
) -> ExperimentConfig:
    model = config.model
    rna = model.rna
    image = model.image
    perturbation = model.perturbation
    if num_genes is not None:
        rna = replace(
            rna,
            vocab_size=max(int(num_genes), 1),
            max_genes=int(max_genes or num_genes),
        )
    if image_channels is not None or image_size is not None:
        image = replace(
            image,
            in_channels=int(image_channels if image_channels is not None else image.in_channels),
            image_size=int(image_size if image_size is not None else image.image_size),
        )
    if metadata_vocab is not None:
        perturbation = replace(
            perturbation,
            num_perturbations=metadata_vocab.num_perturbations,
            num_types=metadata_vocab.num_types,
            num_cell_lines=metadata_vocab.num_cell_lines,
            num_batches=metadata_vocab.num_batches,
        )
    model = PerturbJEPABridgeConfig(
        rna=rna,
        image=image,
        perturbation=perturbation,
        shared_dim=model.shared_dim,
        num_bag_prototypes=model.num_bag_prototypes,
        dropout=model.dropout,
        adversary_scale=model.adversary_scale,
    )
    return replace(config, model=model)


def metadata_tensors(
    rows: Sequence[Mapping[str, object]],
    vocab: MetadataVocab,
    *,
    device: torch.device | str,
    strict: bool = False,
) -> dict[str, torch.Tensor]:
    encoded = [vocab.encode_row(row, strict=strict) for row in rows]
    device = torch.device(device)
    return {
        "perturbation_id": torch.tensor([int(row["perturbation_id"]) for row in encoded], dtype=torch.long, device=device),
        "perturbation_type_id": torch.tensor([int(row["perturbation_type_id"]) for row in encoded], dtype=torch.long, device=device),
        "cell_line_id": torch.tensor([int(row["cell_line_id"]) for row in encoded], dtype=torch.long, device=device),
        "batch_id": torch.tensor([int(row["batch_id"]) for row in encoded], dtype=torch.long, device=device),
        "dose": torch.tensor([float(row["dose"]) for row in encoded], dtype=torch.float32, device=device),
        "time": torch.tensor([float(row["time"]) for row in encoded], dtype=torch.float32, device=device),
    }


def make_token_mask(shape: tuple[int, ...], mask_prob: float, *, device: torch.device | str) -> torch.Tensor:
    if mask_prob <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    return torch.rand(shape, device=device) < mask_prob


def parse_heldout_values(values: str | Sequence[object] | None, *, strategy: str) -> list[object] | None:
    if values is None or values == "":
        return None
    if isinstance(values, str):
        pieces: Sequence[object] = [piece.strip() for piece in values.split(",") if piece.strip()]
    else:
        pieces = list(values)
    if not pieces:
        return None
    if strategy == "heldout_dose_time":
        parsed: list[object] = []
        for piece in pieces:
            if isinstance(piece, (tuple, list)) and len(piece) == 2:
                parsed.append((piece[0], piece[1]))
                continue
            text = str(piece)
            delimiter = "|" if "|" in text else ":"
            if delimiter not in text:
                raise ValueError("heldout_dose_time values must be formatted as dose|time")
            dose, time = text.split(delimiter, 1)
            parsed.append((dose, time))
        return parsed
    return list(pieces)


def assign_real_data_splits(
    rna_metadata: pd.DataFrame,
    image_metadata: pd.DataFrame,
    *,
    split_strategy: str = "none",
    split_col: str = "split",
    train_split_value: str = "train",
    eval_split_value: str = "test",
    heldout_values: Sequence[object] | None = None,
    heldout_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Assign leakage-safe splits over the union of RNA and image metadata."""

    rna = rna_metadata.reset_index(drop=True).copy()
    image = image_metadata.reset_index(drop=True).copy()
    if split_strategy == "none":
        return rna, image, {
            "split_strategy": split_strategy,
            "split_col": split_col,
            "train_split_value": train_split_value,
            "eval_split_value": eval_split_value,
            "heldout_groups": [],
        }

    combined = pd.concat(
        [
            rna.assign(_modality="rna", _row=np.arange(len(rna))),
            image.assign(_modality="image", _row=np.arange(len(image))),
        ],
        ignore_index=True,
        sort=False,
    )
    for column in DEFAULT_METADATA_SCHEMA.biological_keys:
        if column not in combined.columns:
            raise ValueError(f"metadata is missing required split column {column!r}")
        combined[column] = combined[column].map(normalize_value)
    if "batch" in combined.columns:
        combined["batch"] = combined["batch"].map(normalize_value)
    heldout = parse_heldout_values(heldout_values, strategy=split_strategy)

    if split_strategy == "random_grouped":
        combined["condition_id"] = [make_condition_id(row) for row in combined.to_dict(orient="records")]
        split = grouped_hash_split(
            combined,
            ["condition_id"],
            fractions={train_split_value: 1.0 - heldout_fraction, eval_split_value: heldout_fraction},
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["condition_id"]
    elif split_strategy == "heldout_batch":
        split = heldout_batch_split(
            combined,
            heldout_batches=heldout,
            heldout_fraction=heldout_fraction,
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["batch"]
    elif split_strategy == "heldout_perturbation":
        split = heldout_perturbation_split(
            combined,
            heldout_perturbations=heldout,
            heldout_fraction=heldout_fraction,
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["perturbation"]
    elif split_strategy == "heldout_dose_time":
        split = heldout_dose_time_split(
            combined,
            heldout_dose_times=heldout,
            heldout_fraction=heldout_fraction,
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["dose", "time"]
    elif split_strategy == "heldout_cell_line":
        split = heldout_cell_line_split(
            combined,
            heldout_cell_lines=heldout,
            heldout_fraction=heldout_fraction,
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["cell_line"]
    elif split_strategy == "heldout_moa":
        split = heldout_moa_split(
            combined,
            heldout_moas=heldout,
            heldout_fraction=heldout_fraction,
            seed=seed,
            output_col=split_col,
        )
        heldout_group_cols = ["moa"]
    else:
        raise ValueError(f"unsupported split_strategy: {split_strategy}")

    if train_split_value != "train" or eval_split_value != "test":
        split[split_col] = split[split_col].map({"train": train_split_value, "test": eval_split_value}).fillna(split[split_col])
    heldout_groups = _split_group_values(split, heldout_group_cols, split_col=split_col, split_value=eval_split_value)
    rna_split = split.loc[split["_modality"] == "rna"].sort_values("_row").drop(columns=["_modality", "_row"])
    image_split = split.loc[split["_modality"] == "image"].sort_values("_row").drop(columns=["_modality", "_row"])
    metadata = {
        "split_strategy": split_strategy,
        "split_col": split_col,
        "train_split_value": train_split_value,
        "eval_split_value": eval_split_value,
        "heldout_fraction": heldout_fraction,
        "heldout_values": [_format_heldout_value(value) for value in heldout] if heldout is not None else [],
        "heldout_group_columns": heldout_group_cols,
        "heldout_groups": heldout_groups,
    }
    return rna_split.reset_index(drop=True), image_split.reset_index(drop=True), metadata


def filter_metadata_by_split(frame: pd.DataFrame, *, split_col: str, split_value: str, name: str) -> pd.DataFrame:
    if split_col not in frame.columns:
        raise ValueError(f"{name} metadata does not contain split column {split_col!r}")
    filtered = frame.loc[frame[split_col].astype(str) == str(split_value)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"{name} metadata has no rows for {split_col}={split_value!r}")
    return filtered


def _split_group_values(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    split_col: str,
    split_value: str,
) -> list[str]:
    if not group_cols:
        return []
    selected = frame.loc[frame[split_col].astype(str) == str(split_value)]
    if selected.empty:
        return []
    values = selected.loc[:, list(group_cols)].fillna("NA").astype(str).agg("|".join, axis=1)
    return sorted(set(values.tolist()))


def _format_heldout_value(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return "|".join(normalize_value(item) for item in value)
    return normalize_value(value)


def infer_image_shape_from_array(image: np.ndarray) -> tuple[int, int]:
    if image.ndim != 3:
        raise ValueError("expected image array with shape [channels, height, width]")
    if image.shape[1] != image.shape[2]:
        raise ValueError("only square images are supported by the current image encoder config")
    return int(image.shape[0]), int(image.shape[1])


def resize_for_manifest(manifest: pd.DataFrame, image_size: int, *, image_path_col: str = "image_path") -> tuple[int, int] | None:
    if image_path_col not in manifest.columns:
        return (int(image_size), int(image_size))
    suffixes = {Path(str(path)).suffix.lower() for path in manifest[image_path_col].head(32)}
    if suffixes and suffixes.issubset({".npy"}):
        return None
    return (int(image_size), int(image_size))
