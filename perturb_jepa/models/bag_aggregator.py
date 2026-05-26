from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MultiPrototypeBagAggregatorOutput:
    prototypes: torch.Tensor
    bag_embedding: torch.Tensor
    attention: torch.Tensor

    def __iter__(self):
        yield self.prototypes
        yield self.bag_embedding
        yield self.attention


class MultiPrototypeBagAggregator(nn.Module):
    """Aggregate instance embeddings into learned bag prototypes."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        num_prototypes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.num_prototypes = num_prototypes
        self.query = nn.Parameter(torch.empty(num_prototypes, self.output_dim))
        self.key = nn.Linear(input_dim, self.output_dim)
        self.value = nn.Linear(input_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.bag_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> MultiPrototypeBagAggregatorOutput:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, instances, dim]")
        batch, instances, dim = embeddings.shape
        if dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {dim}")
        if mask is not None and mask.shape != (batch, instances):
            raise ValueError(f"mask must have shape {(batch, instances)}")

        valid = None if mask is None else mask.to(dtype=torch.bool)
        keys = self.key(embeddings)
        values = self.value(embeddings)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        logits = torch.einsum("bkd,bnd->bkn", query, keys) / (self.output_dim**0.5)

        if valid is not None:
            logits = logits.masked_fill(~valid[:, None, :], torch.finfo(logits.dtype).min)
            empty_bags = ~valid.any(dim=1)
            if bool(empty_bags.any()):
                logits = logits.clone()
                logits[empty_bags] = 0.0

        attention = F.softmax(logits, dim=-1)
        if valid is not None:
            attention = attention * valid[:, None, :].to(dtype=attention.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)

        prototypes = torch.einsum("bkn,bnd->bkd", self.dropout(attention), values)
        prototypes = self.output_norm(prototypes)
        bag_embedding = self.bag_norm(prototypes.mean(dim=1))
        return MultiPrototypeBagAggregatorOutput(
            prototypes=prototypes,
            bag_embedding=bag_embedding,
            attention=attention,
        )


class MeanBagAggregator(nn.Module):
    """Parameter-free mean pooling with the multi-prototype output contract."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        num_prototypes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        if self.output_dim != self.input_dim:
            raise ValueError("MeanBagAggregator requires output_dim to match input_dim")
        self.num_prototypes = num_prototypes
        self.bag_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> MultiPrototypeBagAggregatorOutput:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, instances, dim]")
        batch, instances, dim = embeddings.shape
        if dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {dim}")
        if mask is not None and mask.shape != (batch, instances):
            raise ValueError(f"mask must have shape {(batch, instances)}")

        if mask is None:
            weights = torch.full((batch, instances), 1.0 / float(instances), device=embeddings.device, dtype=embeddings.dtype)
        else:
            valid = mask.to(device=embeddings.device, dtype=embeddings.dtype)
            weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(torch.finfo(valid.dtype).eps)
        pooled = torch.einsum("bn,bnd->bd", weights, self.dropout(embeddings))
        pooled = self.bag_norm(pooled)
        prototypes = pooled[:, None, :].expand(batch, self.num_prototypes, self.output_dim)
        attention = weights[:, None, :].expand(batch, self.num_prototypes, instances)
        return MultiPrototypeBagAggregatorOutput(
            prototypes=prototypes,
            bag_embedding=pooled,
            attention=attention,
        )
