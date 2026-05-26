from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perturb_jepa.models.cell_jepa_rna import (
    CellJEPARNAConfig,
    CellJEPARNAWarmstart,
    make_cell_jepa_views,
    per_cell_quantile_binning,
)
from perturb_jepa.training.seed import seed_everything


@dataclass(frozen=True)
class CellJEPAWarmstartTrainConfig:
    seed: int = 0
    steps: int = 30
    batch_size: int = 32
    max_train_genes: int = 128
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    device: str = "cpu"


def preprocess_cell_jepa_expression(values: np.ndarray | torch.Tensor, config: CellJEPARNAConfig) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if config.use_quantile_binning:
        return per_cell_quantile_binning(tensor, num_bins=config.num_expression_bins)
    return tensor


def train_cell_jepa_rna_warmstart(
    expression_values: np.ndarray,
    model_config: CellJEPARNAConfig,
    train_config: CellJEPAWarmstartTrainConfig,
    *,
    output_dir: Path | None = None,
) -> tuple[CellJEPARNAWarmstart, dict[str, Any]]:
    seed_everything(train_config.seed)
    device = torch.device(train_config.device if not train_config.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = CellJEPARNAWarmstart(model_config).to(device)
    values = preprocess_cell_jepa_expression(expression_values, model_config).to(device)
    genes = values.shape[1]
    gene_ids_all = torch.arange(genes, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    predictor_optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    rng = np.random.default_rng(train_config.seed)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(train_config.seed)
    trace: list[dict[str, float]] = []
    for step in range(max(1, int(train_config.steps))):
        row_idx = rng.choice(values.shape[0], size=min(train_config.batch_size, values.shape[0]), replace=values.shape[0] < train_config.batch_size)
        cols = _sample_gene_columns(values[row_idx].detach().cpu().numpy(), train_config.max_train_genes, rng)
        batch_values = values[torch.as_tensor(row_idx, dtype=torch.long, device=device)][:, torch.as_tensor(cols, dtype=torch.long, device=device)]
        gene_ids = gene_ids_all[torch.as_tensor(cols, dtype=torch.long, device=device)].unsqueeze(0).expand(batch_values.shape[0], -1)
        view = make_cell_jepa_views(
            batch_values,
            mask_prob=model_config.mask_prob,
            mask_value=model_config.mask_value,
            expressed_gene_subsample=model_config.expressed_gene_subsample,
            generator=torch_generator,
        )
        outputs = model.forward_from_view(gene_ids, view)
        optimizer.zero_grad(set_to_none=True)
        predictor_optimizer.zero_grad(set_to_none=True)
        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(model.student.parameters()) + list(model.predictor.parameters()), 1.0)
        optimizer.step()
        predictor_optimizer.step()
        model.update_teacher()
        trace.append(
            {
                "step": float(step),
                "loss": float(outputs["loss"].detach().cpu()),
                "jepa_loss": float(outputs["jepa_loss"].detach().cpu()),
                "reconstruction_loss": float(outputs["reconstruction_loss"].detach().cpu()),
                "weighted_jepa_to_rec_ratio": float(outputs["weighted_jepa_to_rec_ratio"].detach().cpu()),
            }
        )
    metrics = {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "trace": trace,
        "final_loss": trace[-1]["loss"],
        "final_jepa_loss": trace[-1]["jepa_loss"],
        "final_reconstruction_loss": trace[-1]["reconstruction_loss"],
        "final_weighted_jepa_to_rec_ratio": trace[-1]["weighted_jepa_to_rec_ratio"],
        "device_used": str(device),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    return model, metrics


@torch.no_grad()
def encode_cell_jepa_rna(
    model: CellJEPARNAWarmstart,
    expression_values: np.ndarray,
    *,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    torch_device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()
    values = preprocess_cell_jepa_expression(expression_values, model.config).to(torch_device)
    gene_ids = torch.arange(values.shape[1], dtype=torch.long, device=torch_device).unsqueeze(0)
    embeddings: list[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch_values = values[start : start + batch_size]
        batch_gene_ids = gene_ids.expand(batch_values.shape[0], -1)
        embeddings.append(model.encode(batch_gene_ids, batch_values).detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float64)


@torch.no_grad()
def cell_jepa_dropout_robustness(
    model: CellJEPARNAWarmstart,
    expression_values: np.ndarray,
    *,
    repeats: int = 4,
    batch_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    torch_device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()
    values = preprocess_cell_jepa_expression(expression_values, model.config).to(torch_device)
    gene_ids = torch.arange(values.shape[1], dtype=torch.long, device=torch_device).unsqueeze(0)
    repeated: list[np.ndarray] = []
    generator = torch.Generator(device=torch_device)
    for repeat in range(max(2, repeats)):
        generator.manual_seed(seed + repeat)
        pieces: list[np.ndarray] = []
        for start in range(0, values.shape[0], batch_size):
            batch_values = values[start : start + batch_size]
            view = make_cell_jepa_views(
                batch_values,
                mask_prob=model.config.mask_prob,
                mask_value=model.config.mask_value,
                expressed_gene_subsample=model.config.expressed_gene_subsample,
                generator=generator,
            )
            batch_gene_ids = gene_ids.expand(batch_values.shape[0], -1)
            pieces.append(model.encode(batch_gene_ids, view.student_values).detach().cpu().numpy())
        repeated.append(np.concatenate(pieces, axis=0))
    sims: list[float] = []
    base = _l2_normalize(repeated[0])
    for other in repeated[1:]:
        sims.append(float(np.mean(np.sum(base * _l2_normalize(other), axis=1))))
    return float(np.mean(sims)) if sims else 0.0


@torch.no_grad()
def encode_cell_jepa_augmented_view(
    model: CellJEPARNAWarmstart,
    expression_values: np.ndarray,
    *,
    batch_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    torch_device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = model.to(torch_device)
    model.eval()
    values = preprocess_cell_jepa_expression(expression_values, model.config).to(torch_device)
    gene_ids = torch.arange(values.shape[1], dtype=torch.long, device=torch_device).unsqueeze(0)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)
    pieces: list[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch_values = values[start : start + batch_size]
        view = make_cell_jepa_views(
            batch_values,
            mask_prob=model.config.mask_prob,
            mask_value=model.config.mask_value,
            expressed_gene_subsample=model.config.expressed_gene_subsample,
            generator=generator,
        )
        batch_gene_ids = gene_ids.expand(batch_values.shape[0], -1)
        pieces.append(model.encode(batch_gene_ids, view.student_values).detach().cpu().numpy())
    return np.concatenate(pieces, axis=0).astype(np.float64)


def _sample_gene_columns(batch_values: np.ndarray, max_genes: int, rng: np.random.Generator) -> np.ndarray:
    genes = batch_values.shape[1]
    if max_genes <= 0 or max_genes >= genes:
        return np.arange(genes, dtype=int)
    expressed = np.flatnonzero((batch_values > 0).any(axis=0))
    if expressed.size >= max_genes:
        return np.sort(rng.choice(expressed, size=max_genes, replace=False).astype(int))
    remaining = np.setdiff1d(np.arange(genes, dtype=int), expressed, assume_unique=False)
    needed = max_genes - expressed.size
    extra = rng.choice(remaining, size=needed, replace=False).astype(int) if needed > 0 and remaining.size >= needed else remaining
    return np.sort(np.concatenate([expressed.astype(int), extra.astype(int)]))


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)
