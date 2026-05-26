from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LinearReadoutMap:
    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        bias = np.asarray(self.bias, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("readout weight must have shape [input_dim, output_dim]")
        if bias.shape != (weight.shape[1],):
            raise ValueError("readout bias must have shape [output_dim]")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.weight.shape[1])

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) @ self.weight + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearReadoutMap":
        return cls(
            weight=np.asarray(payload["weight"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )


@dataclass(frozen=True)
class PrefitPLSReadout:
    rna: LinearReadoutMap
    image: LinearReadoutMap
    rank: int
    output_standardize: bool = False
    singular_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rna.output_dim != self.image.output_dim:
            raise ValueError("RNA and image readout maps must have the same output dimension")
        if int(self.rank) != self.rna.output_dim:
            raise ValueError("rank must match readout output dimension")
        if self.singular_values is not None:
            object.__setattr__(self, "singular_values", np.asarray(self.singular_values, dtype=np.float32))

    @property
    def output_dim(self) -> int:
        return self.rna.output_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "prefit_pls_readout",
            "rank": int(self.rank),
            "output_standardize": bool(self.output_standardize),
            "singular_values": None if self.singular_values is None else self.singular_values.tolist(),
            "rna": self.rna.to_dict(),
            "image": self.image.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrefitPLSReadout":
        if payload.get("kind") not in (None, "prefit_pls_readout"):
            raise ValueError(f"unsupported readout kind: {payload.get('kind')}")
        singular = payload.get("singular_values")
        return cls(
            rna=LinearReadoutMap.from_dict(payload["rna"]),
            image=LinearReadoutMap.from_dict(payload["image"]),
            rank=int(payload["rank"]),
            output_standardize=bool(payload.get("output_standardize", False)),
            singular_values=None if singular is None else np.asarray(singular, dtype=np.float32),
        )


def fit_pls_readout(
    rna_values: np.ndarray,
    image_values: np.ndarray,
    *,
    rank: int,
    output_standardize: bool = False,
) -> PrefitPLSReadout:
    rna = np.asarray(rna_values, dtype=np.float32)
    image = np.asarray(image_values, dtype=np.float32)
    if rna.ndim != 2 or image.ndim != 2:
        raise ValueError("PLS readout fitting expects 2D RNA and image matrices")
    if rna.shape[0] != image.shape[0]:
        raise ValueError("RNA and image matrices must have the same number of rows")
    if rank <= 0:
        raise ValueError("rank must be positive")

    x_mean, x_std, xz = _standardize_fit_transform(rna)
    y_mean, y_std, yz = _standardize_fit_transform(image)
    cross = xz.T @ yz / max(1, xz.shape[0] - 1)
    u, singular, vt = np.linalg.svd(cross, full_matrices=False)
    keep = min(int(rank), u.shape[1], vt.shape[0])
    x_projection = u[:, :keep] * np.sqrt(singular[:keep])[None, :]
    y_projection = vt.T[:, :keep] * np.sqrt(singular[:keep])[None, :]
    x_shared = xz @ x_projection
    y_shared = yz @ y_projection
    if output_standardize:
        pooled = np.concatenate([x_shared, y_shared], axis=0)
        output_mean = pooled.mean(axis=0)
        output_std = np.where(pooled.std(axis=0) < 1e-6, 1.0, pooled.std(axis=0))
    else:
        output_mean = np.zeros(keep, dtype=np.float32)
        output_std = np.ones(keep, dtype=np.float32)

    return PrefitPLSReadout(
        rna=_compose_map(x_mean, x_std, x_projection, output_mean, output_std),
        image=_compose_map(y_mean, y_std, y_projection, output_mean, output_std),
        rank=keep,
        output_standardize=output_standardize,
        singular_values=singular[:keep],
    )


def install_prefit_pls_readout(
    model: torch.nn.Module,
    readout: PrefitPLSReadout,
    *,
    freeze: bool = True,
    device: torch.device | str | None = None,
) -> None:
    _install_linear_map(model.rna_raw_linear_projection, readout.rna, freeze=freeze, device=device)
    _install_linear_map(model.image_raw_linear_projection, readout.image, freeze=freeze, device=device)


def install_prefit_pls_distillation_head(
    model: torch.nn.Module,
    readout: PrefitPLSReadout,
    *,
    freeze: bool = False,
    device: torch.device | str | None = None,
) -> None:
    _install_linear_map(model.rna_distilled_linear_projection, readout.rna, freeze=freeze, device=device)
    _install_linear_map(model.image_distilled_linear_projection, readout.image, freeze=freeze, device=device)


def freeze_prefit_pls_readout(model: torch.nn.Module) -> None:
    for module in (model.rna_raw_linear_projection, model.image_raw_linear_projection):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def save_prefit_pls_readout(readout: PrefitPLSReadout, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(readout.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_prefit_pls_readout(path: str | Path) -> PrefitPLSReadout:
    return PrefitPLSReadout.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _standardize_fit_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) < 1e-6, 1.0, values.std(axis=0))
    return mean, std, (values - mean[None, :]) / std[None, :]


def _compose_map(
    mean: np.ndarray,
    std: np.ndarray,
    projection: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
) -> LinearReadoutMap:
    weight = (projection / std[:, None]) / output_std[None, :]
    bias = (-((mean / std) @ projection) - output_mean) / output_std
    return LinearReadoutMap(weight=weight.astype(np.float32), bias=bias.astype(np.float32))


def _install_linear_map(
    layer: torch.nn.Linear,
    fitted: LinearReadoutMap,
    *,
    freeze: bool,
    device: torch.device | str | None,
) -> None:
    target_device = torch.device(device) if device is not None else layer.weight.device
    if layer.in_features != fitted.input_dim or layer.out_features != fitted.output_dim:
        raise ValueError(
            f"linear map shape mismatch: layer=({layer.in_features}, {layer.out_features}), "
            f"map=({fitted.input_dim}, {fitted.output_dim})"
        )
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(fitted.weight.T, dtype=layer.weight.dtype, device=target_device))
        layer.bias.copy_(torch.as_tensor(fitted.bias, dtype=layer.bias.dtype, device=target_device))
    for parameter in layer.parameters():
        parameter.requires_grad_(not freeze)
