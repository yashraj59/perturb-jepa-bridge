from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from perturb_jepa.losses import BridgeLossWeights
from perturb_jepa.models.bridge import PerturbJEPABridge, PerturbJEPABridgeConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig


def default_bridge_config() -> PerturbJEPABridgeConfig:
    return PerturbJEPABridgeConfig(
        rna=RNAEncoderConfig(vocab_size=128, dim=32, depth=1, heads=4, max_genes=64),
        image=ImageEncoderConfig(in_channels=3, image_size=32, patch_size=8, dim=32, depth=1, heads=4),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=16,
            num_types=3,
            num_cell_lines=4,
            num_batches=4,
            dim=32,
        ),
        shared_dim=32,
    )


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def build(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        name = self.name.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps,
            )
        if name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps,
            )
        raise ValueError(f"unsupported optimizer: {self.name}")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2
    device: str = "cpu"
    seed: int = 0
    ema_decay: float = 0.996
    grad_clip_norm: float | None = None
    log_every: int = 1


@dataclass(frozen=True)
class SyntheticBatchConfig:
    batch_size: int = 4
    genes: int = 32
    vocab_size: int = 128
    image_channels: int = 3
    image_size: int = 32
    patch_size: int = 8
    num_perturbations: int = 16
    num_types: int = 3
    num_cell_lines: int = 4
    num_batches: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "synthetic-smoke"
    model: PerturbJEPABridgeConfig = field(default_factory=default_bridge_config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: BridgeLossWeights = field(default_factory=BridgeLossWeights)
    synthetic: SyntheticBatchConfig = field(default_factory=SyntheticBatchConfig)

    @classmethod
    def smoke(cls) -> ExperimentConfig:
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        return cls(
            name=str(data.get("name", cls.name)),
            model=_bridge_config_from_dict(data.get("model", {})),
            optimizer=_dataclass_from_mapping(OptimizerConfig, data.get("optimizer", {})),
            training=_dataclass_from_mapping(TrainingConfig, data.get("training", {})),
            loss=_dataclass_from_mapping(BridgeLossWeights, data.get("loss", {})),
            synthetic=_dataclass_from_mapping(SyntheticBatchConfig, data.get("synthetic", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(value))

    @classmethod
    def load_json(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def build_model(self) -> PerturbJEPABridge:
        return PerturbJEPABridge(self.model)

    def build_optimizer(self, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return self.optimizer.build(parameters)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _dataclass_from_mapping(cls: type[Any], data: Mapping[str, Any], *, default: Any | None = None) -> Any:
    if is_dataclass(data):
        return data
    if data is None:
        data = {}
    merged: dict[str, Any] = {}
    if default is not None:
        merged.update(_to_jsonable(default))
    merged.update(dict(data))
    field_names = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in merged.items() if key in field_names}
    if cls is OptimizerConfig and "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    return cls(**kwargs)


def _bridge_config_from_dict(data: Mapping[str, Any]) -> PerturbJEPABridgeConfig:
    default = default_bridge_config()
    if not data:
        return default

    return PerturbJEPABridgeConfig(
        rna=_dataclass_from_mapping(RNAEncoderConfig, data.get("rna", {}), default=default.rna),
        image=_dataclass_from_mapping(ImageEncoderConfig, data.get("image", {}), default=default.image),
        perturbation=_dataclass_from_mapping(
            PerturbationEncoderConfig,
            data.get("perturbation", {}),
            default=default.perturbation,
        ),
        shared_dim=int(data.get("shared_dim", default.shared_dim)),
        dropout=float(data.get("dropout", default.dropout)),
        adversary_scale=float(data.get("adversary_scale", default.adversary_scale)),
    )
