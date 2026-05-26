from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if int(dim) % 2 != 0:
            raise ValueError("RotaryEmbedding dim must be even")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, int(dim), 2, dtype=torch.float32) / int(dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.inv_freq.device
        dtype = dtype or torch.float32
        positions = torch.arange(int(seq_len), device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        return emb.cos().to(dtype=dtype)[None, None, :, :], emb.sin().to(dtype=dtype)[None, None, :, :]
