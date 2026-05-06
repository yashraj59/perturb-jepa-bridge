from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SyntheticBridgeBatch:
    gene_ids: torch.Tensor
    expression_values: torch.Tensor
    rna_token_mask: torch.Tensor
    images: torch.Tensor
    image_patch_mask: torch.Tensor
    perturbation_id: torch.Tensor
    perturbation_type_id: torch.Tensor
    cell_line_id: torch.Tensor
    batch_id: torch.Tensor
    dose: torch.Tensor
    time: torch.Tensor


def make_synthetic_bridge_batch(
    *,
    batch_size: int = 4,
    genes: int = 32,
    vocab_size: int = 128,
    image_channels: int = 3,
    image_size: int = 32,
    patch_size: int = 8,
    num_perturbations: int = 16,
    num_types: int = 3,
    num_cell_lines: int = 4,
    num_batches: int = 4,
    device: torch.device | str = "cpu",
) -> SyntheticBridgeBatch:
    device = torch.device(device)
    gene_ids = torch.randint(0, vocab_size, (batch_size, genes), device=device)
    expression_values = torch.randn(batch_size, genes, device=device)
    rna_token_mask = torch.rand(batch_size, genes, device=device) < 0.25
    images = torch.randn(batch_size, image_channels, image_size, image_size, device=device)
    num_patches = (image_size // patch_size) ** 2
    image_patch_mask = torch.rand(batch_size, num_patches, device=device) < 0.25
    return SyntheticBridgeBatch(
        gene_ids=gene_ids,
        expression_values=expression_values,
        rna_token_mask=rna_token_mask,
        images=images,
        image_patch_mask=image_patch_mask,
        perturbation_id=torch.randint(0, num_perturbations, (batch_size,), device=device),
        perturbation_type_id=torch.randint(0, num_types, (batch_size,), device=device),
        cell_line_id=torch.randint(0, num_cell_lines, (batch_size,), device=device),
        batch_id=torch.randint(0, num_batches, (batch_size,), device=device),
        dose=torch.rand(batch_size, device=device),
        time=torch.rand(batch_size, device=device),
    )
