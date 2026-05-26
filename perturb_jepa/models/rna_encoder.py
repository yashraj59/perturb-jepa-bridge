from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from perturb_jepa.models.common import MLP


@dataclass(frozen=True)
class RNAEncoderConfig:
    vocab_size: int
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    max_genes: int = 2048
    pooling: str = "cls"


@dataclass
class RNAEncoderOutput:
    token_embeddings: torch.Tensor
    cell_embedding: torch.Tensor
    reconstruction: torch.Tensor


class RNAEncoder(nn.Module):
    def __init__(self, config: RNAEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.value_embedding = MLP(1, config.dim, config.dim, depth=2, dropout=config.dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_genes + 1, config.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.dim * config.mlp_ratio,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.dim)
        self.reconstruction_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 1))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> RNAEncoderOutput:
        if gene_ids.ndim != 2 or expression_values.ndim != 2:
            raise ValueError("gene_ids and expression_values must have shape [batch, genes]")
        if gene_ids.shape != expression_values.shape:
            raise ValueError("gene_ids and expression_values must have the same shape")
        batch, genes = gene_ids.shape
        if genes > self.config.max_genes:
            raise ValueError(f"got {genes} genes, but max_genes={self.config.max_genes}")

        tokens = self.gene_embedding(gene_ids) + self.value_embedding(expression_values.unsqueeze(-1))
        if token_mask is not None:
            tokens = torch.where(token_mask.unsqueeze(-1), self.mask_token.expand(batch, genes, -1), tokens)

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : genes + 1]
        encoded = self.norm(self.encoder(tokens))
        gene_tokens = encoded[:, 1:]
        reconstruction = self.reconstruction_head(gene_tokens).squeeze(-1)
        if self.config.pooling == "cls":
            cell_embedding = encoded[:, 0]
        elif self.config.pooling == "mean_tokens":
            cell_embedding = gene_tokens.mean(dim=1)
        else:
            raise ValueError(f"unsupported RNA pooling mode: {self.config.pooling}")

        return RNAEncoderOutput(
            token_embeddings=gene_tokens,
            cell_embedding=cell_embedding,
            reconstruction=reconstruction,
        )
