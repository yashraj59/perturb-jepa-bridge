from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.models.jepawm_rope import RotaryEmbedding, apply_rope


class RoPESelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, rope_base: float = 10000.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch, token_count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(token_count, device=tokens.device, dtype=tokens.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, token_count, self.dim)
        return self.proj(out)


class ActionAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = RoPESelfAttention(dim, num_heads=num_heads, dropout=dropout, rope_base=rope_base)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.action_mlp = nn.Sequential(nn.LayerNorm(max(1, action_dim)), nn.Linear(max(1, action_dim), 6 * dim))
        if adaln_zero:
            final = self.action_mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 0:
            action = torch.zeros(action.shape[0], 1, device=tokens.device, dtype=tokens.dtype)
        action = action.to(device=tokens.device, dtype=tokens.dtype)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.action_mlp(action).chunk(6, dim=-1)
        attn_in = self.norm_attn(tokens)
        attn_in = attn_in * (1.0 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        tokens = tokens + gate_attn[:, None, :] * self.attn(attn_in)
        mlp_in = self.norm_mlp(tokens)
        mlp_in = mlp_in * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return tokens + gate_mlp[:, None, :] * self.mlp(mlp_in)


class ActionAdaLNRoPEPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_context_tokens: int = 4,
        output_dim: int | None = None,
        adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.max_context_tokens = int(max_context_tokens)
        self.output_dim = int(output_dim or dim)
        self.type_embeddings = nn.Parameter(torch.zeros(self.max_context_tokens, self.dim))
        self.blocks = nn.ModuleList(
            [
                ActionAdaLNBlock(
                    self.dim,
                    self.action_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    adaln_zero=adaln_zero,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.dim)
        self.output = nn.Linear(self.dim, self.output_dim)
        if adaln_zero:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        action: torch.Tensor,
        *,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, T, D]")
        if tokens.shape[1] != self.max_context_tokens:
            raise ValueError("context token count does not match the train-time contract")
        if tokens.shape[2] != self.dim:
            raise ValueError("token dim does not match predictor dim")
        if token_type_ids is None:
            token_type_ids = torch.arange(tokens.shape[1], device=tokens.device)
        if token_type_ids.ndim == 1:
            expected = torch.arange(tokens.shape[1], device=tokens.device)
            if not torch.equal(token_type_ids.to(tokens.device), expected):
                raise ValueError("token type order does not match the context contract")
            type_emb = self.type_embeddings[token_type_ids.to(tokens.device)][None, :, :]
        elif token_type_ids.ndim == 2:
            expected = torch.arange(tokens.shape[1], device=tokens.device)[None, :].expand_as(token_type_ids.to(tokens.device))
            if not torch.equal(token_type_ids.to(tokens.device), expected):
                raise ValueError("token type order does not match the context contract")
            type_emb = F.embedding(token_type_ids.to(tokens.device), self.type_embeddings)
        else:
            raise ValueError("token_type_ids must be rank 1 or rank 2")
        x = tokens + type_emb.to(dtype=tokens.dtype)
        for block in self.blocks:
            x = block(x, action)
        return self.output(self.norm(x[:, -1]))
