from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ImageEncoderConfig:
    in_channels: int = 3
    image_size: int = 64
    patch_size: int = 8
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    max_patches: int = 1024

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size


@dataclass
class ImageEncoderOutput:
    patch_embeddings: torch.Tensor
    image_embedding: torch.Tensor
    patch_reconstruction: torch.Tensor


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)


class ImageEncoder(nn.Module):
    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_patches + 1, config.dim))
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
        self.reconstruction_head = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.patch_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor, *, patch_mask: torch.Tensor | None = None) -> ImageEncoderOutput:
        batch = images.shape[0]
        patch_tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patch_tokens.shape[1]
        if patches > self.config.max_patches:
            raise ValueError(f"got {patches} patches, but max_patches={self.config.max_patches}")
        if patch_mask is not None:
            if patch_mask.shape != (batch, patches):
                raise ValueError(f"patch_mask must have shape {(batch, patches)}")
            patch_tokens = torch.where(
                patch_mask.unsqueeze(-1),
                self.mask_token.expand(batch, patches, -1),
                patch_tokens,
            )
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, patch_tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : patches + 1]
        encoded = self.norm(self.encoder(tokens))
        image_patches = encoded[:, 1:]
        reconstruction = self.reconstruction_head(image_patches)
        return ImageEncoderOutput(
            patch_embeddings=image_patches,
            image_embedding=encoded[:, 0],
            patch_reconstruction=reconstruction,
        )
