from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_samples(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("expected at least [samples, features]")
    if x.ndim == 2:
        if mask is not None:
            if mask.shape != x.shape[:1]:
                raise ValueError(f"mask must have shape {x.shape[:1]}")
            return x[mask.to(dtype=torch.bool)]
        return x
    samples = x.reshape(-1, x.shape[-1])
    if mask is None:
        return samples
    if mask.shape != x.shape[:-1]:
        raise ValueError(f"mask must have shape {x.shape[:-1]}")
    return samples[mask.reshape(-1).to(dtype=torch.bool)]


def mmd_rbf_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_x: torch.Tensor | None = None,
    mask_y: torch.Tensor | None = None,
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    x = _flatten_samples(x, mask_x)
    y = _flatten_samples(y, mask_y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD expects flattened matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD feature dimensions must match")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if not sigmas:
        raise ValueError("sigmas must be non-empty")

    dist_xx = torch.cdist(x, x).pow(2)
    dist_yy = torch.cdist(y, y).pow(2)
    dist_xy = torch.cdist(x, y).pow(2)
    k_xx = torch.zeros_like(dist_xx)
    k_yy = torch.zeros_like(dist_yy)
    k_xy = torch.zeros_like(dist_xy)
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("sigmas must be positive")
        gamma = 1.0 / (2.0 * sigma * sigma)
        k_xx = k_xx + torch.exp(-gamma * dist_xx)
        k_yy = k_yy + torch.exp(-gamma * dist_yy)
        k_xy = k_xy + torch.exp(-gamma * dist_xy)
    scale = float(len(sigmas))
    return k_xx.mean() / scale + k_yy.mean() / scale - 2.0 * k_xy.mean() / scale


def sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 64,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x = _flatten_samples(x)
    y = _flatten_samples(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sliced Wasserstein expects flattened matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein feature dimensions must match")
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")

    projections = torch.randn(x.shape[1], num_projections, device=x.device, dtype=x.dtype, generator=generator)
    projections = F.normalize(projections, dim=0)
    x_proj = (x @ projections).sort(dim=0).values
    y_proj = (y @ projections).sort(dim=0).values
    if x_proj.shape[0] != y_proj.shape[0]:
        sample_count = min(x_proj.shape[0], y_proj.shape[0])
        x_idx = torch.linspace(0, x_proj.shape[0] - 1, sample_count, device=x.device).long()
        y_idx = torch.linspace(0, y_proj.shape[0] - 1, sample_count, device=y.device).long()
        x_proj = x_proj.index_select(0, x_idx)
        y_proj = y_proj.index_select(0, y_idx)
    return (x_proj - y_proj).pow(2).mean()
