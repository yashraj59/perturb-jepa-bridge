from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.data.conditions import MetadataVocab
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
) -> dict[str, torch.Tensor]:
    encoded = [vocab.encode_row(row) for row in rows]
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
