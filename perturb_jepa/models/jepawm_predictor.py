from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BioJEPAWMContextConfig:
    z_dim: int
    action_dim: int
    predictor_dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    context_length: int = 3
    use_uncertainty_token: bool = False
    use_rope: bool = True
    adaln_zero: bool = True
    residual_output_zero_init: bool = True
    max_context_length_seen_at_train: int = 3

    def __post_init__(self) -> None:
        if self.context_length < 3:
            raise ValueError("context_length must be at least 3 for source/floor/query tokens")
        if self.max_context_length_seen_at_train < self.context_length:
            raise ValueError("max_context_length_seen_at_train must be >= context_length")
        if self.predictor_dim % self.heads != 0:
            raise ValueError("predictor_dim must be divisible by heads")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotarySelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, dropout: float = 0.0, use_rope: bool = True) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.use_rope = bool(use_rope)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=tokens.device, dtype=tokens.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean")
            logits = logits.masked_fill(~attention_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.out(out)


class ActionAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        *,
        use_rope: bool = True,
        adaln_zero: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = RotarySelfAttention(dim, heads, dropout=dropout, use_rope=use_rope)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.action_to_mod = nn.Sequential(nn.LayerNorm(max(1, action_dim)), nn.Linear(max(1, action_dim), 6 * dim))
        if adaln_zero:
            final = self.action_to_mod[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 0:
            action = torch.zeros(action.shape[0], 1, device=action.device, dtype=tokens.dtype)
        mod = self.action_to_mod(action.to(dtype=tokens.dtype))
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        x = self.norm1(tokens)
        x = x * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        h = self.attn(x)
        tokens = tokens + gate1[:, None, :] * h
        x = self.norm2(tokens)
        x = x * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        h = self.mlp(x)
        return tokens + gate2[:, None, :] * h


class ActionAdaLNPredictor(nn.Module):
    def __init__(self, config: BioJEPAWMContextConfig) -> None:
        super().__init__()
        self.config = config
        d = config.predictor_dim
        self.action_encoder = nn.Sequential(nn.LayerNorm(max(1, config.action_dim)), nn.Linear(max(1, config.action_dim), d), nn.GELU(), nn.Linear(d, d))
        self.source_proj = nn.Linear(config.z_dim, d)
        self.floor_proj = nn.Linear(config.z_dim, d)
        self.context_proj = nn.Linear(config.z_dim, d)
        self.uncertainty_proj = nn.Linear(1, d) if config.use_uncertainty_token else None
        self.query_token = nn.Parameter(torch.zeros(1, 1, d))
        self.type_embeddings = nn.Parameter(torch.zeros(5, d))
        self.blocks = nn.ModuleList(
            [
                ActionAdaLNBlock(
                    d,
                    config.action_dim,
                    config.heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    use_rope=config.use_rope,
                    adaln_zero=config.adaln_zero,
                )
                for _ in range(config.depth)
            ]
        )
        self.out_norm = nn.LayerNorm(d)
        self.out_head = nn.Linear(d, config.z_dim)
        if config.residual_output_zero_init:
            nn.init.zeros_(self.out_head.weight)
            nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        source_z: torch.Tensor,
        floor_z: torch.Tensor,
        action_features: torch.Tensor,
        *,
        prev_context_z: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = [
            self.source_proj(source_z) + self.type_embeddings[0],
            self.floor_proj(floor_z) + self.type_embeddings[1],
            self.query_token.expand(source_z.shape[0], -1, -1).squeeze(1) + self.type_embeddings[2],
        ]
        if prev_context_z is not None:
            tokens.append(self.context_proj(prev_context_z) + self.type_embeddings[3])
        if self.config.use_uncertainty_token:
            if uncertainty is None:
                uncertainty = torch.zeros(source_z.shape[0], 1, device=source_z.device, dtype=source_z.dtype)
            if self.uncertainty_proj is None:
                raise RuntimeError("uncertainty token requested but projection is absent")
            tokens.append(self.uncertainty_proj(uncertainty) + self.type_embeddings[4])
        token_tensor = torch.stack(tokens, dim=1)
        validate_context_length(
            context_length_train=self.config.context_length,
            context_length_eval=token_tensor.shape[1],
            max_context_length_seen_at_train=self.config.max_context_length_seen_at_train,
        )
        action = action_features
        if action.shape[-1] == 0:
            action = torch.zeros(action.shape[0], 1, device=action.device, dtype=source_z.dtype)
        for block in self.blocks:
            token_tensor = block(token_tensor, action)
        query = self.out_norm(token_tensor[:, 2])
        residual = self.out_head(query)
        aux = {
            "context_length": torch.as_tensor(token_tensor.shape[1], device=source_z.device),
            "action_embedding_norm": self.action_encoder(action).norm(dim=-1).mean().detach(),
        }
        return residual, aux


class FloorPreservingJEPAWMTransitionHead(nn.Module):
    def __init__(self, predictor: ActionAdaLNPredictor) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        source_z: torch.Tensor,
        action_features: torch.Tensor,
        *,
        ridge_floor_delta: torch.Tensor | None = None,
        floor_z: torch.Tensor | None = None,
        residual_gate: torch.Tensor | float = 0.0,
        residual_scale: torch.Tensor | float = 0.0,
        prev_context_z: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if floor_z is None:
            if ridge_floor_delta is None:
                raise ValueError("either floor_z or ridge_floor_delta must be provided")
            floor_z = source_z + ridge_floor_delta
        if ridge_floor_delta is None:
            ridge_floor_delta = floor_z - source_z
        raw, aux = self.predictor(source_z, floor_z, action_features, prev_context_z=prev_context_z, uncertainty=uncertainty)
        gate = _as_batch_scalar(residual_gate, source_z)
        scale = _as_batch_scalar(residual_scale, source_z)
        scaled = gate * scale * raw
        pred_z = floor_z + scaled
        return {
            "pred_z": pred_z,
            "floor_z": floor_z,
            "ridge_floor_delta": ridge_floor_delta,
            "residual_delta_raw": raw,
            "residual_delta_scaled": scaled,
            "residual_gate": gate,
            "residual_scale": scale,
            **aux,
        }


def validate_context_length(
    *,
    context_length_train: int,
    context_length_eval: int,
    max_context_length_seen_at_train: int | None = None,
) -> None:
    if int(context_length_train) != int(context_length_eval):
        raise ValueError("context length mismatch")
    if max_context_length_seen_at_train is not None and int(max_context_length_seen_at_train) < int(context_length_eval):
        raise ValueError("context length mismatch")


def context_contract_dict(
    *,
    context_length_train: int,
    context_length_eval: int,
    rollout_steps: int,
    use_uncertainty_token: bool = False,
) -> dict[str, Any]:
    validate_context_length(context_length_train=context_length_train, context_length_eval=context_length_eval)
    return {
        "context_length_train": int(context_length_train),
        "context_length_eval": int(context_length_eval),
        "token_layout": ["source_control_z_bio", "ridge_floor_predicted_z_bio", "learned_residual_query_token"],
        "use_uncertainty_token": bool(use_uncertainty_token),
        "rollout_steps": int(rollout_steps),
        "context_contract_validated": True,
    }


def _as_batch_scalar(value: torch.Tensor | float, source_z: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device=source_z.device, dtype=source_z.dtype)
    else:
        tensor = torch.as_tensor(float(value), device=source_z.device, dtype=source_z.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(source_z.shape[0], 1)
    elif tensor.ndim == 1:
        tensor = tensor[:, None]
    return tensor
