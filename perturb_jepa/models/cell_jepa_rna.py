from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.common import MLP
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig


@dataclass(frozen=True)
class CellJEPARNAConfig:
    vocab_size: int
    dim: int = 32
    depth: int = 1
    heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.1
    max_genes: int = 512
    pooling: str = "mean_tokens"
    predictor_hidden_dim: int = 64
    mask_prob: float = 0.15
    mask_value: float = -1.0
    num_expression_bins: int = 50
    use_quantile_binning: bool = True
    expressed_gene_subsample: int | None = 128
    ema_momentum: float = 0.99
    w_jepa: float = 50.0
    w_rec: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= self.mask_prob < 1.0:
            raise ValueError("mask_prob must be in [0, 1)")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in [0, 1)")
        if self.w_jepa <= self.w_rec:
            raise ValueError("Cell-JEPA warmstart requires w_jepa to dominate w_rec")
        if self.num_expression_bins <= 1:
            raise ValueError("num_expression_bins must be greater than 1")
        if self.expressed_gene_subsample is not None and self.expressed_gene_subsample <= 0:
            raise ValueError("expressed_gene_subsample must be positive when provided")


@dataclass(frozen=True)
class CellJEPAView:
    teacher_values: torch.Tensor
    student_values: torch.Tensor
    expression_value_mask: torch.Tensor
    subsample_keep_mask: torch.Tensor


def per_cell_quantile_binning(values: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    """Map non-zero expression values to per-cell rank bins, preserving zeros."""

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, genes]")
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1")
    binned = torch.zeros_like(values, dtype=torch.float32)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        expressed = torch.nonzero(row > 0, as_tuple=False).flatten()
        if expressed.numel() == 0:
            continue
        ordered = expressed[torch.argsort(row[expressed], stable=True)]
        ranks = torch.arange(ordered.numel(), device=values.device, dtype=torch.float32)
        bins = torch.floor(ranks * float(num_bins) / float(ordered.numel())).clamp(max=num_bins - 1) + 1.0
        binned[row_idx, ordered] = bins.to(dtype=torch.float32)
    return binned


def make_cell_jepa_views(
    teacher_values: torch.Tensor,
    *,
    mask_prob: float,
    mask_value: float,
    expressed_gene_subsample: int | None,
    generator: torch.Generator | None = None,
) -> CellJEPAView:
    """Create masked student expression values without hiding gene identities."""

    if teacher_values.ndim != 2:
        raise ValueError("teacher_values must have shape [batch, genes]")
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError("mask_prob must be in [0, 1)")
    batch, genes = teacher_values.shape
    keep = torch.ones((batch, genes), dtype=torch.bool, device=teacher_values.device)
    if expressed_gene_subsample is not None and expressed_gene_subsample < genes:
        for row_idx in range(batch):
            expressed = torch.nonzero(teacher_values[row_idx] > 0, as_tuple=False).flatten()
            if expressed.numel() <= expressed_gene_subsample:
                continue
            order = torch.randperm(expressed.numel(), generator=generator, device=teacher_values.device)
            keep_row = torch.zeros(genes, dtype=torch.bool, device=teacher_values.device)
            keep_row[expressed[order[:expressed_gene_subsample]]] = True
            keep_row[teacher_values[row_idx] <= 0] = True
            keep[row_idx] = keep_row

    student_values = teacher_values.clone()
    student_values = torch.where(keep, student_values, torch.zeros_like(student_values))
    maskable = (teacher_values > 0) & keep
    random_values = torch.rand((batch, genes), generator=generator, device=teacher_values.device)
    expression_mask = maskable & (random_values < mask_prob)
    for row_idx in range(batch):
        if maskable[row_idx].any() and not expression_mask[row_idx].any():
            candidates = torch.nonzero(maskable[row_idx], as_tuple=False).flatten()
            choice = candidates[torch.randint(candidates.numel(), (1,), generator=generator, device=teacher_values.device)]
            expression_mask[row_idx, choice] = True
    student_values = torch.where(expression_mask, torch.full_like(student_values, float(mask_value)), student_values)
    return CellJEPAView(
        teacher_values=teacher_values,
        student_values=student_values,
        expression_value_mask=expression_mask,
        subsample_keep_mask=keep,
    )


class CellJEPARNAWarmstart(nn.Module):
    """Cell-JEPA-style RNA representation warmstart for diagnostic use."""

    def __init__(self, config: CellJEPARNAConfig) -> None:
        super().__init__()
        self.config = config
        encoder_config = RNAEncoderConfig(
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            max_genes=config.max_genes,
            pooling=config.pooling,
        )
        self.student = RNAEncoder(encoder_config)
        self.teacher = copy.deepcopy(self.student)
        self.predictor = MLP(config.dim, config.predictor_hidden_dim, config.dim, depth=2, dropout=config.dropout)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def forward_from_view(self, gene_ids: torch.Tensor, view: CellJEPAView) -> dict[str, torch.Tensor]:
        student_out = self.student(gene_ids, view.student_values, token_mask=None)
        with torch.no_grad():
            teacher_out = self.teacher(gene_ids, view.teacher_values, token_mask=None)
        prediction = self.predictor(student_out.cell_embedding)
        target = teacher_out.cell_embedding.detach()
        jepa_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
        if view.expression_value_mask.any():
            rec_loss = F.mse_loss(student_out.reconstruction[view.expression_value_mask], view.teacher_values[view.expression_value_mask])
        else:
            rec_loss = student_out.reconstruction.sum() * 0.0
        total = self.config.w_jepa * jepa_loss + self.config.w_rec * rec_loss
        weighted_rec = self.config.w_rec * rec_loss.detach()
        weighted_jepa = self.config.w_jepa * jepa_loss.detach()
        loss_ratio = weighted_jepa / weighted_rec.clamp_min(1.0e-8)
        return {
            "loss": total,
            "jepa_loss": jepa_loss,
            "reconstruction_loss": rec_loss,
            "weighted_jepa_to_rec_ratio": loss_ratio,
            "student_embedding": student_out.cell_embedding,
            "prediction": prediction,
            "target_embedding": target,
            "student_reconstruction": student_out.reconstruction,
        }

    @torch.no_grad()
    def update_teacher(self) -> None:
        momentum = float(self.config.ema_momentum)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
            teacher_param.mul_(momentum).add_(student_param, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, gene_ids: torch.Tensor, expression_values: torch.Tensor) -> torch.Tensor:
        return self.student(gene_ids, expression_values, token_mask=None).cell_embedding
