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
        num_bag_prototypes=4,
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
class ObjectiveScheduleConfig:
    enabled: bool = False
    reconstruction_warmup_steps: int = 0
    reconstruction_anneal_steps: int = 0
    reconstruction_final_scale: float = 1.0
    warmup_non_reconstruction_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.reconstruction_warmup_steps < 0:
            raise ValueError("reconstruction_warmup_steps must be non-negative")
        if self.reconstruction_anneal_steps < 0:
            raise ValueError("reconstruction_anneal_steps must be non-negative")
        if self.reconstruction_final_scale < 0.0:
            raise ValueError("reconstruction_final_scale must be non-negative")
        if self.warmup_non_reconstruction_scale < 0.0:
            raise ValueError("warmup_non_reconstruction_scale must be non-negative")


@dataclass(frozen=True)
class KendallUncertaintyConfig:
    enabled: bool = False
    term_names: tuple[str, ...] = ()
    initial_log_variance: float = 0.0
    loss_scale: float = 0.5
    regularizer_scale: float = 0.5

    def __post_init__(self) -> None:
        term_names = tuple(self.term_names)
        if len(set(term_names)) != len(term_names):
            raise ValueError("term_names must not contain duplicates")
        if self.loss_scale < 0.0:
            raise ValueError("loss_scale must be non-negative")
        if self.regularizer_scale < 0.0:
            raise ValueError("regularizer_scale must be non-negative")
        object.__setattr__(self, "term_names", term_names)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2
    batch_size: int = 4
    device: str = "cpu"
    seed: int = 0
    ema_decay: float = 0.996
    grad_clip_norm: float | None = None
    log_every: int = 1
    objective_schedule: ObjectiveScheduleConfig = field(default_factory=ObjectiveScheduleConfig)
    uncertainty_weighting: KendallUncertaintyConfig = field(default_factory=KendallUncertaintyConfig)


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
class DataConfig:
    rna_anndata: str | None = None
    image_manifest: str | None = None
    image_root: str | None = None
    condition_key: str = "condition_key"
    split_col: str = "split"
    train_split_value: str = "train"
    eval_split_value: str | None = None
    split_strategy: str = "none"
    heldout_values: tuple[str, ...] = ()
    heldout_fraction: float = 0.2
    rna_bag_size: int = 128
    image_bag_size: int = 128
    min_rna_bag_size: int = 1
    min_image_bag_size: int = 1
    rna_mask_prob: float = 0.0
    image_patch_mask_prob: float = 0.0
    technical_summary: str = "mode"

    def __post_init__(self) -> None:
        valid_strategies = {
            "none",
            "random_grouped",
            "heldout_batch",
            "heldout_perturbation",
            "heldout_dose_time",
            "heldout_cell_line",
            "heldout_moa",
        }
        if self.split_strategy not in valid_strategies:
            raise ValueError(f"split_strategy must be one of {sorted(valid_strategies)}")
        if not 0.0 <= self.rna_mask_prob <= 1.0:
            raise ValueError("rna_mask_prob must be between 0 and 1")
        if not 0.0 <= self.image_patch_mask_prob <= 1.0:
            raise ValueError("image_patch_mask_prob must be between 0 and 1")
        if self.heldout_fraction <= 0.0 or self.heldout_fraction >= 1.0:
            raise ValueError("heldout_fraction must be between 0 and 1")
        object.__setattr__(self, "heldout_values", tuple(self.heldout_values))


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "synthetic-smoke"
    model: PerturbJEPABridgeConfig = field(default_factory=default_bridge_config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: BridgeLossWeights = field(default_factory=BridgeLossWeights)
    data: DataConfig = field(default_factory=DataConfig)
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
            data=_dataclass_from_mapping(DataConfig, data.get("data", {})),
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
    if cls is TrainingConfig:
        if "schedule" in merged and "objective_schedule" not in merged:
            kwargs["objective_schedule"] = merged["schedule"]
        kwargs["objective_schedule"] = _dataclass_from_mapping(
            ObjectiveScheduleConfig,
            kwargs.get("objective_schedule", {}),
            default=TrainingConfig().objective_schedule,
        )
        kwargs["uncertainty_weighting"] = _dataclass_from_mapping(
            KendallUncertaintyConfig,
            kwargs.get("uncertainty_weighting", {}),
            default=TrainingConfig().uncertainty_weighting,
        )
    if cls is KendallUncertaintyConfig and "term_names" in kwargs:
        kwargs["term_names"] = tuple(kwargs["term_names"])
    if cls is DataConfig:
        if "split_key" in merged and "split_col" not in kwargs:
            kwargs["split_col"] = merged["split_key"]
        if "heldout_values" in kwargs and isinstance(kwargs["heldout_values"], str):
            kwargs["heldout_values"] = tuple(
                value.strip() for value in kwargs["heldout_values"].split(",") if value.strip()
            )
        elif "heldout_values" in kwargs:
            kwargs["heldout_values"] = tuple(kwargs["heldout_values"] or ())
    if cls is BridgeLossWeights:
        alias_map = {
            "contrastive_weight": "align",
            "mmd_weight": "mmd",
            "sliced_wasserstein_weight": "sliced_wasserstein",
            "batch_adv_weight": "batch_adv",
            "perturbation_cls_weight": "perturbation_cls",
        }
        for alias, target in alias_map.items():
            if alias in merged and target not in kwargs:
                kwargs[target] = merged[alias]
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
        num_bag_prototypes=int(data.get("num_bag_prototypes", default.num_bag_prototypes)),
        dropout=float(data.get("dropout", default.dropout)),
        adversary_scale=float(data.get("adversary_scale", default.adversary_scale)),
    )
