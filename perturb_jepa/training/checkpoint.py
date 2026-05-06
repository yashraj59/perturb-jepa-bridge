from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    trainer_state: dict[str, Any] | None = None,
    experiment_config: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "trainer_state": trainer_state or {},
        "experiment_config": _serialize_config(experiment_config),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = _torch_load(Path(path), map_location=map_location)
    if "model_state" not in checkpoint:
        raise KeyError("checkpoint is missing model_state")

    if model is not None:
        model.load_state_dict(checkpoint["model_state"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def _serialize_config(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    raise TypeError("experiment_config must be a dict, dataclass, or expose to_dict()")


def _torch_load(path: Path, *, map_location: str | torch.device | None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
