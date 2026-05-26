from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.models.common import MLP


@dataclass(frozen=True)
class SparseResidualLossWeights:
    global_delta: float = 0.05
    top_de: float = 8.0
    program_gene: float = 4.0
    program_direction: float = 1.0
    program_sign: float = 0.25
    outside_sparsity: float = 0.02
    decorrelation: float = 0.01


class SparsePerturbationResidualHead(nn.Module):
    """Perturbation-only delta head with program and low-rank gene factors."""

    def __init__(
        self,
        *,
        num_genes: int,
        num_programs: int,
        program_assignment: Sequence[int],
        num_perturbations: int,
        num_cell_lines: int,
        hidden_dim: int,
        z_pert_dim: int,
        dictionary_rank: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_genes <= 0:
            raise ValueError("num_genes must be positive")
        if num_programs <= 0:
            raise ValueError("num_programs must be positive")
        if dictionary_rank <= 0:
            raise ValueError("dictionary_rank must be positive")
        assignment = torch.as_tensor(tuple(int(value) for value in program_assignment), dtype=torch.long)
        if assignment.numel() != num_genes:
            raise ValueError("program_assignment length must match num_genes")
        if int(assignment.min().item()) < 0 or int(assignment.max().item()) >= num_programs:
            raise ValueError("program_assignment contains an out-of-range program id")

        self.num_genes = int(num_genes)
        self.num_programs = int(num_programs)
        self.num_perturbations = int(num_perturbations)
        self.num_cell_lines = int(num_cell_lines)
        self.dictionary_rank = int(dictionary_rank)
        self.register_buffer("program_assignment", assignment, persistent=True)

        context_dim = num_programs + num_perturbations + num_cell_lines + 2
        self.context_encoder = MLP(context_dim, hidden_dim, z_pert_dim, depth=2, dropout=dropout)
        self.program_decoder = MLP(z_pert_dim, hidden_dim, num_programs, depth=2, dropout=dropout)
        self.low_rank_coeff = MLP(z_pert_dim, hidden_dim, dictionary_rank, depth=2, dropout=dropout)
        self.low_rank_dictionary = nn.Parameter(torch.zeros(dictionary_rank, num_genes))
        self._zero_program_decoder()

    def forward(
        self,
        *,
        source_rna: torch.Tensor,
        perturbation_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.context_features(
            source_rna=source_rna,
            perturbation_id=perturbation_id,
            cell_line_id=cell_line_id,
            dose=dose,
            time=time,
        )
        z_pert = self.context_encoder(context)
        program_delta = self.program_decoder(z_pert)
        program_gene_delta = program_delta.index_select(dim=1, index=self.program_assignment)
        low_rank_coeff = self.low_rank_coeff(z_pert)
        low_rank_delta = low_rank_coeff @ self.low_rank_dictionary
        delta_hat = program_gene_delta + low_rank_delta
        return {
            "sparse_delta_hat": delta_hat,
            "sparse_program_delta": program_delta,
            "sparse_program_gene_delta": program_gene_delta,
            "sparse_low_rank_coeff": low_rank_coeff,
            "sparse_low_rank_delta": low_rank_delta,
            "sparse_z_pert": z_pert,
            "sparse_context": context,
            "sparse_source_program_context": context[:, : self.num_programs],
            "sparse_metadata_context": context[:, self.num_programs :],
        }

    def context_features(
        self,
        *,
        source_rna: torch.Tensor,
        perturbation_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        source = match_last_dim(source_rna, self.num_genes)
        source_program = program_means(source, self.program_assignment, num_programs=self.num_programs)
        metadata = torch.cat(
            (
                F.one_hot(perturbation_id, num_classes=self.num_perturbations).to(dtype=source.dtype),
                F.one_hot(cell_line_id, num_classes=self.num_cell_lines).to(dtype=source.dtype),
                dose.to(dtype=source.dtype).unsqueeze(-1),
                time.to(dtype=source.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return torch.cat((source_program, metadata.to(device=source.device)), dim=-1)

    def _zero_program_decoder(self) -> None:
        final = self.program_decoder.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("program_decoder must end with a Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


def sparse_delta_losses(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    z_inv: torch.Tensor,
    z_pert: torch.Tensor,
    program_assignment: torch.Tensor,
    top_de_k: int,
    top_programs: int,
    weights: SparseResidualLossWeights = SparseResidualLossWeights(),
) -> dict[str, torch.Tensor]:
    if prediction_delta.shape != target_delta.shape:
        raise ValueError("prediction_delta and target_delta must have matching shapes")
    assignment = program_assignment.to(device=prediction_delta.device, dtype=torch.long)
    program_mask, top_de_mask, effect_mask = active_synthetic_effect_mask(
        target_delta,
        assignment,
        top_de_k=top_de_k,
        top_programs=top_programs,
    )
    sample_weights = torch.ones_like(target_delta)
    sample_weights = sample_weights + float(weights.program_gene) * program_mask.to(dtype=target_delta.dtype)
    sample_weights = sample_weights + float(weights.top_de) * top_de_mask.to(dtype=target_delta.dtype)

    residual = prediction_delta - target_delta
    global_delta_mse = F.mse_loss(prediction_delta, target_delta)
    focused_mse = (residual.square() * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    program_pred = program_means(prediction_delta, assignment)
    program_target = program_means(target_delta, assignment)
    program_direction = 1.0 - F.cosine_similarity(program_pred, program_target, dim=-1).mean()
    sign_mask = program_target.abs() > 1e-6
    if bool(sign_mask.any()):
        program_sign = F.relu(-program_pred * program_target.sign()).masked_select(sign_mask).mean()
    else:
        program_sign = prediction_delta.new_zeros(())
    outside_sparsity = _masked_mean(prediction_delta.abs(), ~effect_mask)
    decorrelation = decorrelation_penalty(z_inv, z_pert)
    total = (
        float(weights.global_delta) * global_delta_mse
        + focused_mse
        + float(weights.program_direction) * program_direction
        + float(weights.program_sign) * program_sign
        + float(weights.outside_sparsity) * outside_sparsity
        + float(weights.decorrelation) * decorrelation
    )
    return {
        "total": total,
        "global_delta_mse": global_delta_mse,
        "focused_program_top_de_mse": focused_mse,
        "program_direction_loss": program_direction,
        "program_sign_loss": program_sign,
        "outside_sparsity_loss": outside_sparsity,
        "decorrelation_loss": decorrelation,
        "effect_mask_fraction": effect_mask.to(dtype=prediction_delta.dtype).mean(),
        "top_de_mask_fraction": top_de_mask.to(dtype=prediction_delta.dtype).mean(),
        "program_mask_fraction": program_mask.to(dtype=prediction_delta.dtype).mean(),
    }


def active_synthetic_effect_mask(
    target_delta: torch.Tensor,
    program_assignment: torch.Tensor,
    *,
    top_de_k: int,
    top_programs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_delta.ndim != 2:
        raise ValueError("target_delta must be [batch, genes]")
    assignment = program_assignment.to(device=target_delta.device, dtype=torch.long)
    if assignment.shape != (target_delta.shape[-1],):
        raise ValueError("program_assignment length must match target_delta genes")
    batch, genes = target_delta.shape
    top_de_mask = torch.zeros((batch, genes), dtype=torch.bool, device=target_delta.device)
    if top_de_k > 0:
        k = min(int(top_de_k), genes)
        top_de_index = target_delta.abs().topk(k, dim=1).indices
        top_de_mask.scatter_(1, top_de_index, True)

    program_mask = torch.zeros((batch, genes), dtype=torch.bool, device=target_delta.device)
    if top_programs > 0:
        program_delta = program_means(target_delta, assignment)
        k = min(int(top_programs), program_delta.shape[1])
        top_program_index = program_delta.abs().topk(k, dim=1).indices
        gene_program = assignment.view(1, 1, genes)
        program_mask = gene_program.eq(top_program_index.unsqueeze(-1)).any(dim=1)
    return program_mask, top_de_mask, program_mask | top_de_mask


def decorrelation_penalty(z_inv: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
    if z_inv.shape[0] != z_pert.shape[0]:
        raise ValueError("z_inv and z_pert batch sizes must match")
    if z_inv.shape[0] < 2:
        return z_pert.new_zeros(())
    inv = _standardize(z_inv.detach())
    pert = _standardize(z_pert)
    cross = inv.transpose(0, 1) @ pert / float(max(1, z_inv.shape[0] - 1))
    return cross.square().mean()


def program_means(values: torch.Tensor, assignment: torch.Tensor, *, num_programs: int | None = None) -> torch.Tensor:
    if values.shape[-1] != assignment.numel():
        raise ValueError("program assignment length must match values last dimension")
    num_programs = int(num_programs or (int(assignment.max().item()) + 1))
    membership = F.one_hot(assignment, num_classes=num_programs).to(dtype=values.dtype, device=values.device)
    counts = membership.sum(dim=0).clamp_min(1.0)
    return values @ membership / counts


def match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    if values.shape[-1] == target_dim:
        return values
    if values.shape[-1] > target_dim:
        return values[..., :target_dim]
    return F.pad(values, (0, target_dim - values.shape[-1]))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have matching shapes")
    if not bool(mask.any()):
        return values.new_zeros(())
    return values.masked_select(mask).mean()


def _standardize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return centered / scale
