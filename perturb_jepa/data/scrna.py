from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from perturb_jepa.data.conditions import MetadataVocab, build_condition_bags
from perturb_jepa.data.schema import normalize_scrna_obs, normalize_scrna_var


@dataclass(frozen=True)
class GeneVocab:
    gene_ids: list[str]
    gene_symbols: list[str]
    token_to_idx: dict[str, int]

    @classmethod
    def from_var(cls, var) -> "GeneVocab":
        normalized = normalize_scrna_var(var)
        gene_ids = normalized["gene_id"].astype(str).tolist()
        gene_symbols = normalized["gene_symbol"].astype(str).tolist()
        token_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
        for idx, symbol in enumerate(gene_symbols):
            token_to_idx.setdefault(symbol, idx)
        return cls(gene_ids=gene_ids, gene_symbols=gene_symbols, token_to_idx=token_to_idx)


def read_h5ad(path: str | Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install the data extra to read h5ad files: pip install -e '.[data]'") from exc
    adata = ad.read_h5ad(path)
    adata.obs = normalize_scrna_obs(adata.obs)
    adata.var = normalize_scrna_var(adata.var)
    return adata


def as_dense_array(matrix) -> np.ndarray:
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def library_size_log1p(matrix, *, target_sum: float = 1e4) -> np.ndarray:
    dense = as_dense_array(matrix).astype(np.float32, copy=False)
    totals = dense.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1.0)
    normalized = dense / totals * target_sum
    return np.log1p(normalized).astype(np.float32, copy=False)


def select_high_variance_genes(matrix, *, n_top: int) -> np.ndarray:
    if n_top <= 0:
        raise ValueError("n_top must be positive")
    dense = as_dense_array(matrix).astype(np.float32, copy=False)
    variance = dense.var(axis=0)
    n_top = min(n_top, dense.shape[1])
    return np.argsort(variance)[-n_top:][::-1]


def make_token_batch(
    matrix,
    gene_indices: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dense = as_dense_array(matrix).astype(np.float32, copy=False)
    if row_indices is None:
        row_indices = np.arange(dense.shape[0])
    genes = np.asarray(gene_indices, dtype=np.int64)
    rows = np.asarray(row_indices, dtype=np.int64)
    values = dense[np.ix_(rows, genes)]
    gene_tokens = np.broadcast_to(genes[None, :], values.shape).copy()
    return gene_tokens, values.astype(np.float32, copy=False)


@dataclass(frozen=True)
class SCRNATokenExample:
    gene_ids: np.ndarray
    expression_values: np.ndarray
    condition_key: str
    obs_index: str
    perturbation_id: int
    perturbation_type_id: int
    cell_line_id: int
    batch_id: int
    dose: float
    time: float


@dataclass
class SCRNATokenBatch:
    gene_ids: torch.Tensor
    expression_values: torch.Tensor
    rna_token_mask: torch.Tensor
    perturbation_id: torch.Tensor
    perturbation_type_id: torch.Tensor
    cell_line_id: torch.Tensor
    batch_id: torch.Tensor
    dose: torch.Tensor
    time: torch.Tensor
    condition_key: list[str]
    obs_index: list[str]


class SCRNATokenDataset(Dataset):
    """Normalized scRNA-seq rows as fixed-length gene token sequences."""

    def __init__(
        self,
        matrix,
        obs: pd.DataFrame,
        var: pd.DataFrame | None = None,
        *,
        gene_indices: Sequence[int] | None = None,
        n_top_genes: int | None = None,
        normalize: bool = True,
        target_sum: float = 1e4,
        metadata_vocab: MetadataVocab | None = None,
    ) -> None:
        dense = as_dense_array(matrix).astype(np.float32, copy=False)
        if dense.ndim != 2:
            raise ValueError("matrix must have shape [cells, genes]")
        if len(obs) != dense.shape[0]:
            raise ValueError("obs rows must match matrix cells")

        if var is None:
            var = pd.DataFrame(index=[str(index) for index in range(dense.shape[1])])
        if len(var) != dense.shape[1]:
            raise ValueError("var rows must match matrix genes")
        if gene_indices is not None and n_top_genes is not None:
            raise ValueError("set either gene_indices or n_top_genes, not both")

        self.obs = normalize_scrna_obs(obs)
        self.var = normalize_scrna_var(var)
        self.gene_vocab = GeneVocab.from_var(self.var)
        self.metadata_vocab = metadata_vocab or MetadataVocab.from_frame(self.obs)
        self.encoded_obs = self.metadata_vocab.encode_frame(self.obs)
        self.condition_bags = build_condition_bags(self.obs)

        expression = library_size_log1p(dense, target_sum=target_sum) if normalize else dense.copy()
        if gene_indices is None:
            if n_top_genes is None:
                selected = np.arange(expression.shape[1], dtype=np.int64)
            else:
                selected = select_high_variance_genes(expression, n_top=n_top_genes).astype(np.int64, copy=False)
        else:
            selected = np.asarray(gene_indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("gene_indices must be a one-dimensional sequence")
        if selected.size == 0:
            raise ValueError("at least one gene index is required")
        if selected.min() < 0 or selected.max() >= dense.shape[1]:
            raise ValueError("gene_indices contain out-of-bounds positions")

        self.gene_indices = selected
        self.expression_values = expression[:, self.gene_indices].astype(np.float32, copy=False)
        self.gene_ids = self.gene_indices.astype(np.int64, copy=True)

    def __len__(self) -> int:
        return self.expression_values.shape[0]

    def __getitem__(self, index: int) -> SCRNATokenExample:
        row = self.encoded_obs.iloc[index]
        return SCRNATokenExample(
            gene_ids=self.gene_ids.copy(),
            expression_values=self.expression_values[index].copy(),
            condition_key=str(row["condition_key"]),
            obs_index=str(self.obs.index[index]),
            perturbation_id=int(row["perturbation_id"]),
            perturbation_type_id=int(row["perturbation_type_id"]),
            cell_line_id=int(row["cell_line_id"]),
            batch_id=int(row["batch_id"]),
            dose=float(row["dose"]),
            time=float(row["time"]),
        )


class SCRNATokenCollator:
    def __init__(self, *, mask_prob: float = 0.15, seed: int | None = None) -> None:
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be between 0 and 1")
        self.mask_prob = mask_prob
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(self, examples: Sequence[SCRNATokenExample]) -> SCRNATokenBatch:
        if not examples:
            raise ValueError("cannot collate an empty scRNA batch")
        gene_ids = torch.as_tensor(np.stack([example.gene_ids for example in examples]), dtype=torch.long)
        expression_values = torch.as_tensor(
            np.stack([example.expression_values for example in examples]),
            dtype=torch.float32,
        )
        if self.mask_prob == 0.0:
            token_mask = torch.zeros_like(expression_values, dtype=torch.bool)
        else:
            token_mask = torch.rand(expression_values.shape, generator=self.generator) < self.mask_prob
        return SCRNATokenBatch(
            gene_ids=gene_ids,
            expression_values=expression_values,
            rna_token_mask=token_mask,
            perturbation_id=torch.tensor([example.perturbation_id for example in examples], dtype=torch.long),
            perturbation_type_id=torch.tensor(
                [example.perturbation_type_id for example in examples],
                dtype=torch.long,
            ),
            cell_line_id=torch.tensor([example.cell_line_id for example in examples], dtype=torch.long),
            batch_id=torch.tensor([example.batch_id for example in examples], dtype=torch.long),
            dose=torch.tensor([example.dose for example in examples], dtype=torch.float32),
            time=torch.tensor([example.time for example in examples], dtype=torch.float32),
            condition_key=[example.condition_key for example in examples],
            obs_index=[example.obs_index for example in examples],
        )
