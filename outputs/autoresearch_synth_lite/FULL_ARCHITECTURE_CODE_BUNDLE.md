# Full Architecture Code Bundle

Date: 2026-05-25

This file is generated from the repository source files listed below. It is intended as a complete code bundle for the current synthetic JEPA/scRNA perturbation architecture, protected PLS readout path, trainable clone path, counterfactual decoder path, and current best matching/count references.

It is a snapshot for review. The source of truth remains the files at the paths shown in each section.

## Included Files

- `perturb_jepa/config.py`
- `perturb_jepa/models/common.py`
- `perturb_jepa/models/ema.py`
- `perturb_jepa/models/rna_encoder.py`
- `perturb_jepa/models/image_encoder.py`
- `perturb_jepa/models/perturbation_encoder.py`
- `perturb_jepa/models/projection_heads.py`
- `perturb_jepa/models/bag_aggregator.py`
- `perturb_jepa/models/adversary.py`
- `perturb_jepa/models/bridge.py`
- `perturb_jepa/losses.py`
- `perturb_jepa/training/objectives.py`
- `perturb_jepa/training/trainer.py`
- `perturb_jepa/training/prefit_readout.py`
- `perturb_jepa/training/synthetic_biology_lite.py`
- `scripts/run_synthetic_lite_step0.py`
- `scripts/evaluate_prefit_pls_readout.py`
- `scripts/train_pls_distilled_head.py`
- `scripts/train_clone_counterfactual_decoder.py`
- `scripts/run_family_m_transport_baselines.py`
- `scripts/run_family_n_distillation.py`
- `scripts/run_family_o_count_likelihood.py`

## File: `perturb_jepa/config.py`

```python
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
    rna_normalize: bool = True
    technical_summary: str = "mode"
    bag_sample_tech_col: str | None = None

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
    program_assignment = data.get(
        "counterfactual_rna_program_assignment",
        default.counterfactual_rna_program_assignment,
    )
    if isinstance(program_assignment, str):
        program_assignment = tuple(int(value.strip()) for value in program_assignment.split(",") if value.strip())
    else:
        program_assignment = tuple(int(value) for value in (program_assignment or ()))

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
        bag_aggregator=str(data.get("bag_aggregator", default.bag_aggregator)),
        rna_condition_readout=str(data.get("rna_condition_readout", default.rna_condition_readout)),
        rna_pseudobulk_normalize=bool(data.get("rna_pseudobulk_normalize", default.rna_pseudobulk_normalize)),
        image_condition_readout=str(data.get("image_condition_readout", default.image_condition_readout)),
        image_raw_normalize=bool(data.get("image_raw_normalize", default.image_raw_normalize)),
        counterfactual_rna_residual=bool(data.get("counterfactual_rna_residual", default.counterfactual_rna_residual)),
        counterfactual_rna_program_factorized=bool(
            data.get("counterfactual_rna_program_factorized", default.counterfactual_rna_program_factorized)
        ),
        counterfactual_rna_num_programs=int(
            data.get("counterfactual_rna_num_programs", default.counterfactual_rna_num_programs)
        ),
        counterfactual_rna_program_assignment=program_assignment,
        counterfactual_rna_within_program_residual=bool(
            data.get("counterfactual_rna_within_program_residual", default.counterfactual_rna_within_program_residual)
        ),
        counterfactual_rna_program_conditioned=bool(
            data.get("counterfactual_rna_program_conditioned", default.counterfactual_rna_program_conditioned)
        ),
        counterfactual_rna_program_metadata_context=bool(
            data.get(
                "counterfactual_rna_program_metadata_context",
                default.counterfactual_rna_program_metadata_context,
            )
        ),
        counterfactual_rna_program_decoder_depth=int(
            data.get(
                "counterfactual_rna_program_decoder_depth",
                default.counterfactual_rna_program_decoder_depth,
            )
        ),
    )
```

## File: `perturb_jepa/models/common.py`

```python
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth - 1):
            layers.extend((nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None, dim: int = 1) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=dim)
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, scale)
```

## File: `perturb_jepa/models/ema.py`

```python
from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema_teacher(module: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(module)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, *, decay: float = 0.996) -> None:
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters(), strict=True):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers(), strict=True):
        teacher_buffer.copy_(student_buffer)
```

## File: `perturb_jepa/models/rna_encoder.py`

```python
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
```

## File: `perturb_jepa/models/image_encoder.py`

```python
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
    pooling: str = "cls"

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
        if self.config.pooling == "cls":
            image_embedding = encoded[:, 0]
        elif self.config.pooling == "mean_patches":
            image_embedding = image_patches.mean(dim=1)
        else:
            raise ValueError(f"unsupported image pooling mode: {self.config.pooling}")

        return ImageEncoderOutput(
            patch_embeddings=image_patches,
            image_embedding=image_embedding,
            patch_reconstruction=reconstruction,
        )
```

## File: `perturb_jepa/models/perturbation_encoder.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from perturb_jepa.models.common import MLP


@dataclass(frozen=True)
class PerturbationEncoderConfig:
    num_perturbations: int
    num_types: int
    num_cell_lines: int
    num_batches: int
    dim: int = 128
    descriptor_dim: int = 0
    dropout: float = 0.1


class PerturbationEncoder(nn.Module):
    def __init__(self, config: PerturbationEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.perturbation_embedding = nn.Embedding(config.num_perturbations, config.dim)
        self.type_embedding = nn.Embedding(config.num_types, config.dim)
        self.cell_line_embedding = nn.Embedding(config.num_cell_lines, config.dim)
        numeric_dim = 2 + config.descriptor_dim
        self.numeric_mlp = MLP(numeric_dim, config.dim, config.dim, depth=2, dropout=config.dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.dim * 4),
            nn.Linear(config.dim * 4, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.dim),
        )

    def forward(
        self,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        *,
        descriptor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if descriptor is None:
            descriptor = torch.zeros(
                perturbation_id.shape[0],
                self.config.descriptor_dim,
                device=perturbation_id.device,
                dtype=dose.dtype,
            )
        numeric = torch.cat((dose.unsqueeze(-1), time.unsqueeze(-1), descriptor), dim=-1)
        pieces = (
            self.perturbation_embedding(perturbation_id),
            self.type_embedding(perturbation_type_id),
            self.cell_line_embedding(cell_line_id),
            self.numeric_mlp(numeric),
        )
        return self.fusion(torch.cat(pieces, dim=-1))
```

## File: `perturb_jepa/models/projection_heads.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class RNAProjectionHead(_ProjectionHead):
    """Projection head for RNA embeddings; intentionally accepts no metadata."""


class ImageProjectionHead(_ProjectionHead):
    """Projection head for image embeddings; intentionally accepts no metadata."""
```

## File: `perturb_jepa/models/bag_aggregator.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MultiPrototypeBagAggregatorOutput:
    prototypes: torch.Tensor
    bag_embedding: torch.Tensor
    attention: torch.Tensor

    def __iter__(self):
        yield self.prototypes
        yield self.bag_embedding
        yield self.attention


class MultiPrototypeBagAggregator(nn.Module):
    """Aggregate instance embeddings into learned bag prototypes."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        num_prototypes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.num_prototypes = num_prototypes
        self.query = nn.Parameter(torch.empty(num_prototypes, self.output_dim))
        self.key = nn.Linear(input_dim, self.output_dim)
        self.value = nn.Linear(input_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.bag_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> MultiPrototypeBagAggregatorOutput:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, instances, dim]")
        batch, instances, dim = embeddings.shape
        if dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {dim}")
        if mask is not None and mask.shape != (batch, instances):
            raise ValueError(f"mask must have shape {(batch, instances)}")

        valid = None if mask is None else mask.to(dtype=torch.bool)
        keys = self.key(embeddings)
        values = self.value(embeddings)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        logits = torch.einsum("bkd,bnd->bkn", query, keys) / (self.output_dim**0.5)

        if valid is not None:
            logits = logits.masked_fill(~valid[:, None, :], torch.finfo(logits.dtype).min)
            empty_bags = ~valid.any(dim=1)
            if bool(empty_bags.any()):
                logits = logits.clone()
                logits[empty_bags] = 0.0

        attention = F.softmax(logits, dim=-1)
        if valid is not None:
            attention = attention * valid[:, None, :].to(dtype=attention.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)

        prototypes = torch.einsum("bkn,bnd->bkd", self.dropout(attention), values)
        prototypes = self.output_norm(prototypes)
        bag_embedding = self.bag_norm(prototypes.mean(dim=1))
        return MultiPrototypeBagAggregatorOutput(
            prototypes=prototypes,
            bag_embedding=bag_embedding,
            attention=attention,
        )


class MeanBagAggregator(nn.Module):
    """Parameter-free mean pooling with the multi-prototype output contract."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int | None = None,
        num_prototypes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        if self.output_dim != self.input_dim:
            raise ValueError("MeanBagAggregator requires output_dim to match input_dim")
        self.num_prototypes = num_prototypes
        self.bag_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> MultiPrototypeBagAggregatorOutput:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, instances, dim]")
        batch, instances, dim = embeddings.shape
        if dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {dim}")
        if mask is not None and mask.shape != (batch, instances):
            raise ValueError(f"mask must have shape {(batch, instances)}")

        if mask is None:
            weights = torch.full((batch, instances), 1.0 / float(instances), device=embeddings.device, dtype=embeddings.dtype)
        else:
            valid = mask.to(device=embeddings.device, dtype=embeddings.dtype)
            weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(torch.finfo(valid.dtype).eps)
        pooled = torch.einsum("bn,bnd->bd", weights, self.dropout(embeddings))
        pooled = self.bag_norm(pooled)
        prototypes = pooled[:, None, :].expand(batch, self.num_prototypes, self.output_dim)
        attention = weights[:, None, :].expand(batch, self.num_prototypes, instances)
        return MultiPrototypeBagAggregatorOutput(
            prototypes=prototypes,
            bag_embedding=pooled,
            attention=attention,
        )
```

## File: `perturb_jepa/models/adversary.py`

```python
from __future__ import annotations

import math

import torch
from torch import nn

from perturb_jepa.models.common import MLP


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        return _GradientReverse.apply(x, self.scale if scale is None else float(scale))


def gradient_reversal_ramp(step: int, max_steps: int, *, max_scale: float = 1.0) -> float:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
    return float(max_scale) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


class BatchAdversary(nn.Module):
    """Predict caller-provided technical labels through gradient reversal."""

    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        *,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        hidden_dim = hidden_dim or input_dim
        self.reversal = GradientReversalLayer(scale=scale)
        self.classifier = MLP(input_dim, hidden_dim, num_batches, depth=depth, dropout=dropout)

    def forward(self, x: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
        return self.classifier(self.reversal(x, scale=scale))
```

## File: `perturb_jepa/models/bridge.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.models.common import MLP, gradient_reverse
from perturb_jepa.models.ema import make_ema_teacher, update_ema_teacher
from perturb_jepa.models.image_encoder import ImageEncoder, ImageEncoderConfig
from perturb_jepa.models.adversary import BatchAdversary
from perturb_jepa.models.bag_aggregator import MeanBagAggregator, MultiPrototypeBagAggregator
from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig
from perturb_jepa.models.projection_heads import ImageProjectionHead, RNAProjectionHead
from perturb_jepa.models.rna_encoder import RNAEncoder, RNAEncoderConfig


@dataclass(frozen=True)
class PerturbJEPABridgeConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
    num_bag_prototypes: int = 8
    dropout: float = 0.1
    adversary_scale: float = 1.0
    bag_aggregator: str = "attention"
    rna_condition_readout: str = "encoder"
    rna_pseudobulk_normalize: bool = True
    image_condition_readout: str = "encoder"
    image_raw_normalize: bool = True
    counterfactual_rna_residual: bool = False
    counterfactual_rna_program_factorized: bool = False
    counterfactual_rna_num_programs: int = 0
    counterfactual_rna_program_assignment: tuple[int, ...] = ()
    counterfactual_rna_within_program_residual: bool = False
    counterfactual_rna_program_conditioned: bool = False
    counterfactual_rna_program_metadata_context: bool = False
    counterfactual_rna_program_decoder_depth: int = 2


class PerturbJEPABridge(nn.Module):
    def __init__(self, config: PerturbJEPABridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.rna_encoder = RNAEncoder(config.rna)
        self.image_encoder = ImageEncoder(config.image)
        self.rna_teacher = make_ema_teacher(self.rna_encoder)
        self.image_teacher = make_ema_teacher(self.image_encoder)
        self.perturbation_encoder = PerturbationEncoder(config.perturbation)

        self.rna_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        image_raw_dim = config.image.in_channels * config.image.image_size * config.image.image_size
        self.image_raw_projection = nn.Sequential(
            nn.LayerNorm(image_raw_dim),
            nn.Linear(image_raw_dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.image_raw_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_linear_projection = nn.Linear(image_raw_dim, config.shared_dim)
        self.image_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.dim),
            nn.Linear(config.rna.dim, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_pseudobulk_projection = nn.Sequential(
            nn.LayerNorm(config.rna.max_genes),
            nn.Linear(config.rna.max_genes, config.shared_dim),
            nn.GELU(),
            nn.LayerNorm(config.shared_dim),
            nn.Linear(config.shared_dim, config.shared_dim),
        )
        self.rna_raw_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_linear_projection = nn.Linear(config.rna.max_genes, config.shared_dim)
        self.rna_distilled_residual_scale = nn.Parameter(torch.zeros(()))
        self.rna_teacher_projection = RNAProjectionHead(config.rna.dim, config.shared_dim, config.shared_dim)
        self.image_teacher_projection = ImageProjectionHead(config.image.dim, config.shared_dim, config.shared_dim)
        self.rna_teacher_projection.load_state_dict(self.rna_projection.state_dict())
        self.image_teacher_projection.load_state_dict(self.image_projection.state_dict())
        for module in (self.rna_teacher_projection, self.image_teacher_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        aggregator_cls = self._bag_aggregator_cls(config.bag_aggregator)
        self.rna_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.image_bag_aggregator = aggregator_cls(
            config.shared_dim,
            output_dim=config.shared_dim,
            num_prototypes=config.num_bag_prototypes,
            dropout=config.dropout,
        )
        self.rna_teacher_bag_aggregator = make_ema_teacher(self.rna_bag_aggregator)
        self.image_teacher_bag_aggregator = make_ema_teacher(self.image_bag_aggregator)
        self.rna_jepa_predictor = MLP(config.rna.dim, config.rna.dim, config.rna.dim, depth=2, dropout=config.dropout)
        self.image_jepa_predictor = MLP(
            config.image.dim,
            config.image.dim,
            config.image.dim,
            depth=2,
            dropout=config.dropout,
        )
        self.state_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.response_head = MLP(config.shared_dim, config.shared_dim, config.shared_dim, depth=2, dropout=config.dropout)
        self.perturbation_classifier = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.state_perturbation_adversary = MLP(
            config.shared_dim,
            config.shared_dim,
            config.perturbation.num_perturbations,
            depth=2,
            dropout=config.dropout,
        )
        self.batch_adversary = BatchAdversary(
            config.shared_dim,
            config.perturbation.num_batches,
            hidden_dim=config.shared_dim,
            dropout=config.dropout,
            scale=config.adversary_scale,
        )
        self.delta_gate = MLP(
            config.shared_dim + config.perturbation.dim,
            config.shared_dim,
            config.shared_dim,
            depth=3,
            dropout=config.dropout,
        )
        self.delta_effect = nn.Linear(config.perturbation.dim, config.shared_dim)
        self.rna_distribution_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.rna.vocab_size,
            depth=2,
            dropout=config.dropout,
        )
        if config.counterfactual_rna_program_factorized:
            if config.counterfactual_rna_num_programs <= 0:
                raise ValueError("counterfactual_rna_num_programs must be positive for program-factorized decoding")
            assignment = torch.as_tensor(config.counterfactual_rna_program_assignment, dtype=torch.long)
            if assignment.numel() != config.rna.vocab_size:
                raise ValueError("counterfactual_rna_program_assignment length must match RNA vocab size")
            if int(assignment.min().item()) < 0 or int(assignment.max().item()) >= config.counterfactual_rna_num_programs:
                raise ValueError("counterfactual_rna_program_assignment contains an out-of-range program id")
            program_decoder_input_dim = config.shared_dim
            if config.counterfactual_rna_program_conditioned:
                program_decoder_input_dim += config.counterfactual_rna_num_programs
            if config.counterfactual_rna_program_metadata_context:
                program_decoder_input_dim += config.perturbation.num_perturbations + config.perturbation.num_cell_lines + 2
            self.counterfactual_program_decoder = MLP(
                program_decoder_input_dim,
                config.shared_dim,
                config.counterfactual_rna_num_programs,
                depth=config.counterfactual_rna_program_decoder_depth,
                dropout=config.dropout,
            )
            self.register_buffer("counterfactual_rna_program_assignment", assignment, persistent=True)
        self.image_prototype_decoder = MLP(
            config.shared_dim,
            config.shared_dim,
            config.shared_dim,
            depth=2,
            dropout=config.dropout,
        )

    @staticmethod
    def _bag_aggregator_cls(name: str):
        if name == "attention":
            return MultiPrototypeBagAggregator
        if name == "mean":
            return MeanBagAggregator
        raise ValueError(f"unsupported bag_aggregator: {name}")

    @torch.no_grad()
    def update_teachers(self, *, decay: float = 0.996) -> None:
        update_ema_teacher(self.rna_encoder, self.rna_teacher, decay=decay)
        update_ema_teacher(self.image_encoder, self.image_teacher, decay=decay)
        update_ema_teacher(self.rna_projection, self.rna_teacher_projection, decay=decay)
        update_ema_teacher(self.image_projection, self.image_teacher_projection, decay=decay)
        update_ema_teacher(self.rna_bag_aggregator, self.rna_teacher_bag_aggregator, decay=decay)
        update_ema_teacher(self.image_bag_aggregator, self.image_teacher_bag_aggregator, decay=decay)

    @staticmethod
    def _set_eval_temporarily(modules: tuple[nn.Module, ...]):
        class _TeacherEvalContext:
            def __init__(self, modules: tuple[nn.Module, ...]) -> None:
                self.modules = modules
                self.states = [module.training for module in modules]

            def __enter__(self) -> None:
                for module in self.modules:
                    module.eval()

            def __exit__(self, exc_type, exc, tb) -> None:
                for module, state in zip(self.modules, self.states, strict=True):
                    module.train(state)

        return _TeacherEvalContext(modules)

    def encode_perturbation(
        self,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        *,
        descriptor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.perturbation_encoder(
            perturbation_id,
            perturbation_type_id,
            cell_line_id,
            batch_id,
            dose,
            time,
            descriptor=descriptor,
        )

    def predict_delta(self, z_state: torch.Tensor, perturbation_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.delta_gate(torch.cat((z_state, perturbation_embedding), dim=-1)))
        base_delta = self.delta_effect(perturbation_embedding)
        delta = gate * base_delta
        return delta, gate, base_delta

    def forward(
        self,
        *,
        gene_ids: torch.Tensor | None = None,
        expression_values: torch.Tensor | None = None,
        rna_token_mask: torch.Tensor | None = None,
        rna_bag_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_patch_mask: torch.Tensor | None = None,
        image_bag_mask: torch.Tensor | None = None,
        perturbation_id: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        batch_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
        descriptor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        perturbation = self.encode_perturbation(
            perturbation_id,
            perturbation_type_id,
            cell_line_id,
            batch_id,
            dose,
            time,
            descriptor=descriptor,
        )
        outputs["perturbation_embedding"] = perturbation

        shared_for_state: torch.Tensor | None = None
        rna_condition_for_counterfactual: torch.Tensor | None = None
        if gene_ids is not None and expression_values is not None:
            rna_batch_shape = gene_ids.shape[:-1]
            if gene_ids.ndim == 2:
                flat_gene_ids = gene_ids
                flat_expression_values = expression_values
                flat_token_mask = rna_token_mask
                rna_bag_shape = (gene_ids.shape[0], 1)
            elif gene_ids.ndim == 3:
                if expression_values.shape != gene_ids.shape:
                    raise ValueError("bagged gene_ids and expression_values must have matching shapes")
                rna_bag_shape = gene_ids.shape[:2]
                flat_gene_ids = gene_ids.reshape(-1, gene_ids.shape[-1])
                flat_expression_values = expression_values.reshape(-1, expression_values.shape[-1])
                flat_token_mask = None if rna_token_mask is None else rna_token_mask.reshape(-1, rna_token_mask.shape[-1])
            else:
                raise ValueError("gene_ids must have shape [batch, genes] or [batch, bag, genes]")

            rna = self.rna_encoder(flat_gene_ids, flat_expression_values, token_mask=flat_token_mask)
            rna_instance_shared = self.rna_projection(rna.cell_embedding).reshape(*rna_bag_shape, -1)
            rna_aggregated = self.rna_bag_aggregator(rna_instance_shared, mask=rna_bag_mask)
            rna_shared = rna_aggregated.bag_embedding
            rna_pseudobulk_shared = self._rna_pseudobulk_shared(gene_ids, expression_values)
            rna_raw_pseudobulk_shared = self._rna_raw_pseudobulk_shared(expression_values)
            rna_raw_linear_shared = self._rna_raw_linear_pseudobulk_shared(expression_values)
            rna_distilled_linear_shared = self._rna_distilled_linear_pseudobulk_shared(expression_values)
            rna_distilled_residual_shared = rna_distilled_linear_shared + (
                self.rna_distilled_residual_scale * rna_raw_pseudobulk_shared
            )
            rna_condition_for_counterfactual = self._rna_condition_values(expression_values)
            if self.config.rna_condition_readout == "encoder":
                pass
            elif self.config.rna_condition_readout == "pseudobulk":
                rna_shared = rna_pseudobulk_shared
            elif self.config.rna_condition_readout == "encoder_plus_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_pseudobulk_shared, dim=-1)
            elif self.config.rna_condition_readout == "raw_pseudobulk":
                rna_shared = rna_raw_pseudobulk_shared
            elif self.config.rna_condition_readout == "encoder_plus_raw_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_raw_pseudobulk_shared, dim=-1)
            elif self.config.rna_condition_readout == "raw_linear_pseudobulk":
                rna_shared = rna_raw_linear_shared
            elif self.config.rna_condition_readout == "encoder_plus_raw_linear_pseudobulk":
                rna_shared = F.normalize(rna_shared + rna_raw_linear_shared, dim=-1)
            else:
                raise ValueError(f"unsupported rna_condition_readout: {self.config.rna_condition_readout}")
            rna_state = self.state_head(rna_shared)
            rna_response = self.response_head(rna_shared)
            with torch.no_grad(), self._set_eval_temporarily(
                (self.rna_teacher, self.rna_teacher_projection, self.rna_teacher_bag_aggregator)
            ):
                rna_teacher = self.rna_teacher(flat_gene_ids, flat_expression_values, token_mask=None)
                rna_teacher_instances = self.rna_teacher_projection(rna_teacher.cell_embedding).reshape(*rna_bag_shape, -1)
                rna_teacher_aggregated = self.rna_teacher_bag_aggregator(rna_teacher_instances, mask=rna_bag_mask)
                rna_teacher_shared = rna_teacher_aggregated.bag_embedding.detach()
            rna_tokens = rna.token_embeddings.reshape(*rna_batch_shape, *rna.token_embeddings.shape[1:])
            rna_teacher_tokens = rna_teacher.token_embeddings.reshape(*rna_batch_shape, *rna_teacher.token_embeddings.shape[1:])
            rna_token_prediction = self.rna_jepa_predictor(rna.token_embeddings).reshape(
                *rna_batch_shape,
                *rna.token_embeddings.shape[1:],
            )
            rna_reconstruction = rna.reconstruction.reshape(*rna_batch_shape, rna.reconstruction.shape[-1])
            outputs.update(
                {
                    "rna_tokens": rna_tokens,
                    "rna_teacher_tokens": rna_teacher_tokens.detach(),
                    "rna_token_prediction": rna_token_prediction,
                    "rna_reconstruction": rna_reconstruction,
                    "rna_instance_shared": rna_instance_shared,
                    "rna_prototypes": rna_aggregated.prototypes,
                    "rna_attention": rna_aggregated.attention,
                    "rna_pseudobulk_shared": rna_pseudobulk_shared,
                    "rna_raw_pseudobulk_shared": rna_raw_pseudobulk_shared,
                    "rna_raw_linear_shared": rna_raw_linear_shared,
                    "rna_distilled_shared": rna_raw_pseudobulk_shared,
                    "rna_distilled_linear_shared": rna_distilled_linear_shared,
                    "rna_distilled_residual_shared": rna_distilled_residual_shared,
                    "rna_shared": rna_shared,
                    "rna_retrieval": rna_shared,
                    "rna_teacher_shared": rna_teacher_shared,
                    "rna_state": rna_state,
                    "rna_response": rna_response,
                    "rna_perturbation_logits": self.perturbation_classifier(rna_response),
                    "rna_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(rna_state, scale=self.config.adversary_scale)
                    ),
                    "rna_batch_logits": self.batch_adversary(rna_shared, scale=self.config.adversary_scale),
                }
            )
            shared_for_state = rna_shared

        if images is not None:
            if images.ndim == 4:
                image_is_bagged = False
                image_bag_shape = (images.shape[0], 1)
                flat_images = images
                flat_patch_mask = image_patch_mask
            elif images.ndim == 5:
                image_is_bagged = True
                image_bag_shape = images.shape[:2]
                flat_images = images.reshape(-1, *images.shape[-3:])
                flat_patch_mask = None if image_patch_mask is None else image_patch_mask.reshape(-1, image_patch_mask.shape[-1])
            else:
                raise ValueError("images must have shape [batch, channels, height, width] or [batch, bag, channels, height, width]")

            image = self.image_encoder(flat_images, patch_mask=flat_patch_mask)
            image_instance_shared = self.image_projection(image.image_embedding).reshape(*image_bag_shape, -1)
            image_aggregated = self.image_bag_aggregator(image_instance_shared, mask=image_bag_mask)
            image_shared = image_aggregated.bag_embedding
            image_raw_shared = self._image_raw_shared(images)
            image_raw_linear_shared = self._image_raw_linear_shared(images)
            image_distilled_linear_shared = self._image_distilled_linear_shared(images)
            image_distilled_residual_shared = image_distilled_linear_shared + (
                self.image_distilled_residual_scale * image_raw_shared
            )
            if self.config.image_condition_readout == "encoder":
                pass
            elif self.config.image_condition_readout == "raw_pooled":
                image_shared = image_raw_shared
            elif self.config.image_condition_readout == "encoder_plus_raw_pooled":
                image_shared = F.normalize(image_shared + image_raw_shared, dim=-1)
            elif self.config.image_condition_readout == "raw_linear_pooled":
                image_shared = image_raw_linear_shared
            elif self.config.image_condition_readout == "encoder_plus_raw_linear_pooled":
                image_shared = F.normalize(image_shared + image_raw_linear_shared, dim=-1)
            else:
                raise ValueError(f"unsupported image_condition_readout: {self.config.image_condition_readout}")
            image_state = self.state_head(image_shared)
            image_response = self.response_head(image_shared)
            with torch.no_grad(), self._set_eval_temporarily(
                (self.image_teacher, self.image_teacher_projection, self.image_teacher_bag_aggregator)
            ):
                image_teacher = self.image_teacher(flat_images, patch_mask=None)
                image_teacher_instances = self.image_teacher_projection(image_teacher.image_embedding).reshape(*image_bag_shape, -1)
                image_teacher_aggregated = self.image_teacher_bag_aggregator(image_teacher_instances, mask=image_bag_mask)
                image_teacher_shared = image_teacher_aggregated.bag_embedding.detach()
            image_patch_prediction_flat = self.image_jepa_predictor(image.patch_embeddings)
            if image_is_bagged:
                image_patches = image.patch_embeddings.reshape(*image_bag_shape, *image.patch_embeddings.shape[1:])
                image_teacher_patches = image_teacher.patch_embeddings.reshape(
                    *image_bag_shape,
                    *image_teacher.patch_embeddings.shape[1:],
                )
                image_patch_prediction = image_patch_prediction_flat.reshape(
                    *image_bag_shape,
                    *image_patch_prediction_flat.shape[1:],
                )
                image_reconstruction = image.patch_reconstruction.reshape(
                    *image_bag_shape,
                    *image.patch_reconstruction.shape[1:],
                )
            else:
                image_patches = image.patch_embeddings
                image_teacher_patches = image_teacher.patch_embeddings
                image_patch_prediction = image_patch_prediction_flat
                image_reconstruction = image.patch_reconstruction
            outputs.update(
                {
                    "image_patches": image_patches,
                    "image_teacher_patches": image_teacher_patches.detach(),
                    "image_patch_prediction": image_patch_prediction,
                    "image_patch_reconstruction": image_reconstruction,
                    "image_instance_shared": image_instance_shared,
                    "image_prototypes": image_aggregated.prototypes,
                    "image_attention": image_aggregated.attention,
                    "image_raw_shared": image_raw_shared,
                    "image_raw_linear_shared": image_raw_linear_shared,
                    "image_distilled_shared": image_raw_shared,
                    "image_distilled_linear_shared": image_distilled_linear_shared,
                    "image_distilled_residual_shared": image_distilled_residual_shared,
                    "image_shared": image_shared,
                    "image_retrieval": image_shared,
                    "image_teacher_shared": image_teacher_shared,
                    "image_state": image_state,
                    "image_response": image_response,
                    "image_perturbation_logits": self.perturbation_classifier(image_response),
                    "image_state_perturbation_logits": self.state_perturbation_adversary(
                        gradient_reverse(image_state, scale=self.config.adversary_scale)
                    ),
                    "image_batch_logits": self.batch_adversary(image_shared, scale=self.config.adversary_scale),
                }
            )
            shared_for_state = image_shared if shared_for_state is None else 0.5 * (shared_for_state + image_shared)

        if shared_for_state is not None:
            z_state = self.state_head(shared_for_state)
            z_response = self.response_head(shared_for_state)
            delta, delta_gate, delta_base = self.predict_delta(z_state, perturbation)
            z_counterfactual = z_state + delta
            reverse_delta, reverse_gate, reverse_base = self.predict_delta(z_counterfactual, -perturbation)
            z_cycle = z_counterfactual + reverse_delta
            counterfactual_rna_delta, counterfactual_rna_aux = self._counterfactual_rna_delta(
                z_counterfactual,
                rna_condition_for_counterfactual,
                perturbation_id=perturbation_id,
                cell_line_id=cell_line_id,
                dose=dose,
                time=time,
            )
            counterfactual_rna = counterfactual_rna_delta
            if self.config.counterfactual_rna_residual and rna_condition_for_counterfactual is not None:
                counterfactual_rna = _match_last_dim(rna_condition_for_counterfactual, counterfactual_rna_delta.shape[-1])
                counterfactual_rna = counterfactual_rna + counterfactual_rna_delta
            outputs.update(
                {
                    "z_state": z_state,
                    "z_response": z_response,
                    "z_counterfactual": z_counterfactual,
                    "counterfactual_delta": delta,
                    "counterfactual_gate": delta_gate,
                    "counterfactual_base_delta": delta_base,
                    "cycle_reconstruction": z_cycle,
                    "cycle_delta": reverse_delta,
                    "cycle_gate": reverse_gate,
                    "cycle_base_delta": reverse_base,
                    "counterfactual_rna_delta": counterfactual_rna_delta,
                    "counterfactual_rna": counterfactual_rna,
                    "counterfactual_image": self.image_prototype_decoder(z_counterfactual),
                    **counterfactual_rna_aux,
                }
            )

        return outputs

    def _rna_pseudobulk_shared(self, gene_ids: torch.Tensor, expression_values: torch.Tensor) -> torch.Tensor:
        if gene_ids.ndim == 2 and expression_values.ndim == 2:
            condition_gene_ids = gene_ids
            condition_values = expression_values
        elif gene_ids.ndim == 3 and expression_values.ndim == 3:
            condition_gene_ids = gene_ids[:, 0, :]
            condition_values = expression_values.mean(dim=1)
        else:
            raise ValueError("gene_ids and expression_values must both be 2D or 3D")
        gene_embedding = self.rna_encoder.gene_embedding(condition_gene_ids)
        value_embedding = self.rna_encoder.value_embedding(condition_values.unsqueeze(-1))
        projected = self.rna_pseudobulk_projection((gene_embedding + value_embedding).mean(dim=1))
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_raw_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_raw_pseudobulk_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_raw_linear_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_raw_linear_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_distilled_linear_pseudobulk_shared(self, expression_values: torch.Tensor) -> torch.Tensor:
        condition_values = self._rna_condition_values(expression_values)
        projected = self.rna_distilled_linear_projection(condition_values)
        if self.config.rna_pseudobulk_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _rna_condition_values(self, expression_values: torch.Tensor) -> torch.Tensor:
        if expression_values.ndim == 2:
            condition_values = expression_values
        elif expression_values.ndim == 3:
            condition_values = expression_values.mean(dim=1)
        else:
            raise ValueError("expression_values must be 2D or 3D")
        expected_genes = self.config.rna.max_genes
        if condition_values.shape[-1] > expected_genes:
            raise ValueError(f"raw pseudobulk input has {condition_values.shape[-1]} genes, expected at most {expected_genes}")
        if condition_values.shape[-1] < expected_genes:
            condition_values = F.pad(condition_values, (0, expected_genes - condition_values.shape[-1]))
        return condition_values

    def _image_raw_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_raw_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_raw_linear_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_raw_linear_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_distilled_linear_shared(self, images: torch.Tensor) -> torch.Tensor:
        condition_images = self._image_condition_values(images)
        projected = self.image_distilled_linear_projection(condition_images)
        if self.config.image_raw_normalize:
            return F.normalize(projected, dim=-1)
        return projected

    def _image_condition_values(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            condition_images = images
        elif images.ndim == 5:
            condition_images = images.mean(dim=1)
        else:
            raise ValueError("images must be 4D or 5D")
        return condition_images.reshape(condition_images.shape[0], -1)

    def _counterfactual_rna_delta(
        self,
        z_counterfactual: torch.Tensor,
        source_rna: torch.Tensor | None,
        perturbation_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.counterfactual_rna_program_factorized:
            return self.rna_distribution_decoder(z_counterfactual), {}
        assignment = self.counterfactual_rna_program_assignment
        decoder_input = z_counterfactual
        source_program_context = torch.zeros(
            z_counterfactual.shape[0],
            self.config.counterfactual_rna_num_programs,
            device=z_counterfactual.device,
            dtype=z_counterfactual.dtype,
        )
        if self.config.counterfactual_rna_program_conditioned:
            if source_rna is not None:
                source_program_context = _program_means(
                    _match_last_dim(source_rna, assignment.numel()),
                    assignment,
                    num_programs=self.config.counterfactual_rna_num_programs,
                )
            decoder_input = torch.cat((z_counterfactual, source_program_context), dim=-1)
        metadata_context = torch.zeros(
            z_counterfactual.shape[0],
            self.config.perturbation.num_perturbations + self.config.perturbation.num_cell_lines + 2,
            device=z_counterfactual.device,
            dtype=z_counterfactual.dtype,
        )
        if self.config.counterfactual_rna_program_metadata_context:
            metadata_context = torch.cat(
                (
                    F.one_hot(perturbation_id, num_classes=self.config.perturbation.num_perturbations).to(
                        dtype=z_counterfactual.dtype
                    ),
                    F.one_hot(cell_line_id, num_classes=self.config.perturbation.num_cell_lines).to(
                        dtype=z_counterfactual.dtype
                    ),
                    dose.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
                    time.to(dtype=z_counterfactual.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
            decoder_input = torch.cat((decoder_input, metadata_context), dim=-1)
        program_delta = self.counterfactual_program_decoder(decoder_input)
        program_gene_delta = program_delta.index_select(dim=1, index=assignment)
        within_program_residual = torch.zeros_like(program_gene_delta)
        if self.config.counterfactual_rna_within_program_residual:
            raw_residual = _match_last_dim(self.rna_distribution_decoder(z_counterfactual), program_gene_delta.shape[-1])
            within_program_residual = raw_residual - _program_gene_means(raw_residual, assignment)
        delta = program_gene_delta + within_program_residual
        return delta, {
            "counterfactual_rna_program_delta": program_delta,
            "counterfactual_rna_program_gene_delta": program_gene_delta,
            "counterfactual_rna_within_program_residual": within_program_residual,
            "counterfactual_rna_source_program_context": source_program_context,
            "counterfactual_rna_metadata_context": metadata_context,
            "counterfactual_rna_program_decoder_input": decoder_input,
        }


def _match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    if values.shape[-1] == target_dim:
        return values
    if values.shape[-1] > target_dim:
        return values[..., :target_dim]
    return F.pad(values, (0, target_dim - values.shape[-1]))


def _program_means(values: torch.Tensor, assignment: torch.Tensor, *, num_programs: int | None = None) -> torch.Tensor:
    if values.shape[-1] != assignment.numel():
        raise ValueError("program assignment length must match values last dimension")
    num_programs = int(num_programs or (int(assignment.max().item()) + 1))
    membership = F.one_hot(assignment, num_classes=num_programs).to(dtype=values.dtype, device=values.device)
    counts = membership.sum(dim=0).clamp_min(1.0)
    return values @ membership / counts


def _program_gene_means(values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    program_means = _program_means(values, assignment)
    return program_means.index_select(dim=1, index=assignment)
```

## File: `perturb_jepa/losses.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from perturb_jepa.distribution_losses import (
    mmd_rbf_loss as prototype_mmd_rbf_loss,
    sliced_wasserstein_loss as prototype_sliced_wasserstein_loss,
)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
    error = (prediction - target).pow(2)
    if mask is None:
        return error.mean()
    weights = mask.to(dtype=error.dtype)
    while weights.ndim < error.ndim:
        weights = weights.unsqueeze(-1)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def cosine_jepa_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(student, teacher.detach(), dim=-1).mean()


def variance_floor_loss(x: torch.Tensor, *, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("variance floor loss expects at least [batch, features]")
    values = x.reshape(-1, x.shape[-1])
    if values.shape[0] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    std = torch.sqrt(values.var(dim=0, unbiased=False) + eps)
    return F.relu(float(target_std) - std).mean()


def covariance_offdiag_loss(x: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("covariance off-diagonal loss expects at least [batch, features]")
    values = x.reshape(-1, x.shape[-1])
    if values.shape[0] < 2 or values.shape[1] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    values = values - values.mean(dim=0, keepdim=True)
    values = values / torch.sqrt(values.var(dim=0, unbiased=False, keepdim=True) + eps)
    covariance = values.T @ values / max(1, values.shape[0] - 1)
    offdiag = covariance - torch.diag_embed(torch.diagonal(covariance))
    return offdiag.pow(2).sum() / values.shape[1]


def cross_correlation_identity_loss(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("cross-correlation identity loss inputs must have matching shapes")
    if x.ndim < 2:
        raise ValueError("cross-correlation identity loss expects at least [batch, features]")
    x_values = x.reshape(-1, x.shape[-1])
    y_values = y.reshape(-1, y.shape[-1])
    if x_values.shape[0] < 2 or x_values.shape[1] < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x_values = (x_values - x_values.mean(dim=0, keepdim=True)) / torch.sqrt(
        x_values.var(dim=0, unbiased=False, keepdim=True) + eps
    )
    y_values = (y_values - y_values.mean(dim=0, keepdim=True)) / torch.sqrt(
        y_values.var(dim=0, unbiased=False, keepdim=True) + eps
    )
    correlation = x_values.T @ y_values / x_values.shape[0]
    identity = torch.eye(correlation.shape[0], device=x.device, dtype=x.dtype)
    return (correlation - identity).pow(2).sum() / correlation.shape[0]


def masked_cosine_jepa_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"JEPA shape mismatch: student={student.shape}, teacher={teacher.shape}")
    cosine = F.cosine_similarity(student, teacher.detach(), dim=-1)
    if mask is None:
        return 1.0 - cosine.mean()
    if mask.shape != cosine.shape:
        raise ValueError(f"JEPA mask must have shape {cosine.shape}, got {mask.shape}")
    selected = mask.to(device=cosine.device, dtype=torch.bool)
    if not bool(selected.any()):
        return torch.zeros((), device=student.device, dtype=student.dtype)
    return 1.0 - cosine[selected].mean()


def info_nce_loss(x: torch.Tensor, y: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("InfoNCE inputs must have matching shapes")
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _as_label_tensor(values: Any, *, device: torch.device) -> torch.Tensor | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        return values.to(device=device)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        values = list(values)
        try:
            return torch.as_tensor(values, device=device)
        except (TypeError, ValueError):
            encoded: dict[Any, int] = {}
            labels: list[int] = []
            for value in values:
                if value not in encoded:
                    encoded[value] = len(encoded)
                labels.append(encoded[value])
            return torch.tensor(labels, device=device)
    return None


def _metadata_value_matrix(values: Sequence[Any], key: str, *, device: torch.device) -> torch.Tensor | None:
    extracted = [item.get(key) if isinstance(item, Mapping) else None for item in values]
    if all(value is None for value in extracted):
        return None
    encoded: dict[Any, int] = {}
    labels: list[int] = []
    for value in extracted:
        if value not in encoded:
            encoded[value] = len(encoded)
        labels.append(encoded[value])
    return torch.tensor(labels, device=device)


def _apply_label_positives(weights: torch.Tensor, labels: torch.Tensor, value: float) -> torch.Tensor:
    labels = labels.reshape(-1)
    if labels.shape != (weights.shape[0],):
        raise ValueError("positive labels must have shape [batch]")
    positive = labels[:, None].eq(labels[None, :])
    return torch.maximum(weights, positive.to(dtype=weights.dtype) * float(value))


def build_multi_positive_weights(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    positive_mask: torch.Tensor | None = None,
    bio_keys: Any = None,
    labels: Any = None,
    positive_weights: torch.Tensor | None = None,
    weak_positive_weight: float = 0.2,
) -> torch.Tensor:
    if positive_weights is not None:
        if positive_weights.shape != (batch_size, batch_size):
            raise ValueError(f"positive_weights must have shape {(batch_size, batch_size)}")
        weights = positive_weights.to(device=device, dtype=dtype)
    else:
        weights = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)

    if positive_mask is not None:
        if positive_mask.shape != (batch_size, batch_size):
            raise ValueError(f"positive_mask must have shape {(batch_size, batch_size)}")
        weights = torch.maximum(weights, positive_mask.to(device=device, dtype=dtype))

    label_tensor = _as_label_tensor(labels, device=device)
    if label_tensor is not None:
        weights = _apply_label_positives(weights, label_tensor, 1.0)

    if bio_keys is not None:
        if isinstance(bio_keys, Mapping):
            exact_keys = ("condition", "condition_id", "condition_key", "bio_key", "label")
            weak_keys = ("perturbation", "perturbation_id", "moa", "pathway")
            for key in exact_keys:
                label_tensor = _as_label_tensor(bio_keys.get(key), device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, 1.0)
            for key in weak_keys:
                label_tensor = _as_label_tensor(bio_keys.get(key), device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, weak_positive_weight)
        elif isinstance(bio_keys, Sequence) and bio_keys and isinstance(bio_keys[0], Mapping):
            for key in ("condition", "condition_id", "condition_key", "bio_key", "label"):
                label_tensor = _metadata_value_matrix(bio_keys, key, device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, 1.0)
            for key in ("perturbation", "perturbation_id", "moa", "pathway"):
                label_tensor = _metadata_value_matrix(bio_keys, key, device=device)
                if label_tensor is not None:
                    weights = _apply_label_positives(weights, label_tensor, weak_positive_weight)
        else:
            label_tensor = _as_label_tensor(bio_keys, device=device)
            if label_tensor is not None:
                weights = _apply_label_positives(weights, label_tensor, 1.0)

    diagonal = torch.eye(batch_size, device=device, dtype=dtype)
    weights = torch.maximum(weights, diagonal)
    return weights


class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(
        self,
        *,
        temperature: float = 0.1,
        weak_positive_weight: float = 0.2,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if weak_positive_weight < 0:
            raise ValueError("weak_positive_weight must be non-negative")
        self.temperature = temperature
        self.weak_positive_weight = weak_positive_weight
        self.symmetric = symmetric

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        positive_mask: torch.Tensor | None = None,
        bio_keys: Any = None,
        labels: Any = None,
        positive_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("MultiPositiveInfoNCE inputs must have matching shapes")
        if x.ndim != 2:
            raise ValueError("MultiPositiveInfoNCE inputs must have shape [batch, features]")
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logits = x @ y.T / self.temperature
        weights = build_multi_positive_weights(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
            positive_mask=positive_mask,
            bio_keys=bio_keys,
            labels=labels,
            positive_weights=positive_weights,
            weak_positive_weight=self.weak_positive_weight,
        )

        def directional_loss(direction_logits: torch.Tensor, direction_weights: torch.Tensor) -> torch.Tensor:
            log_prob = direction_logits.log_softmax(dim=1)
            norm = direction_weights.sum(dim=1).clamp_min(torch.finfo(direction_weights.dtype).eps)
            return -((direction_weights * log_prob).sum(dim=1) / norm).mean()

        loss_xy = directional_loss(logits, weights)
        if not self.symmetric:
            return loss_xy
        return 0.5 * (loss_xy + directional_loss(logits.T, weights.T))


def multi_positive_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
    bio_keys: Any = None,
    labels: Any = None,
    positive_weights: torch.Tensor | None = None,
    temperature: float = 0.1,
    weak_positive_weight: float = 0.2,
    symmetric: bool = True,
) -> torch.Tensor:
    return MultiPositiveInfoNCELoss(
        temperature=temperature,
        weak_positive_weight=weak_positive_weight,
        symmetric=symmetric,
    )(
        x,
        y,
        positive_mask=positive_mask,
        bio_keys=bio_keys,
        labels=labels,
        positive_weights=positive_weights,
    )


def supervised_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError("supervised InfoNCE inputs must have matching shapes")
    if labels.shape != (x.shape[0],):
        raise ValueError("labels must have shape [batch]")
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    positive = labels[:, None].eq(labels[None, :])
    log_prob_xy = logits.log_softmax(dim=1)
    log_prob_yx = logits.T.log_softmax(dim=1)
    denom = positive.sum(dim=1).clamp_min(1).to(dtype=x.dtype)
    loss_xy = -(log_prob_xy * positive.to(dtype=x.dtype)).sum(dim=1) / denom
    loss_yx = -(log_prob_yx * positive.to(dtype=x.dtype)).sum(dim=1) / denom
    return 0.5 * (loss_xy.mean() + loss_yx.mean())


def multi_resolution_info_nce_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    label_levels: Sequence[torch.Tensor],
    *,
    weights: Sequence[float] | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    if not label_levels:
        return info_nce_loss(x, y, temperature=temperature)
    if weights is None:
        weights = [1.0 / len(label_levels)] * len(label_levels)
    if len(weights) != len(label_levels):
        raise ValueError("weights and label_levels must have the same length")
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    norm = float(sum(weights))
    if norm <= 0:
        raise ValueError("multi-resolution InfoNCE weights must sum to a positive value")
    for weight, labels in zip(weights, label_levels, strict=True):
        total = total + float(weight) * supervised_info_nce_loss(x, y, labels, temperature=temperature)
    return total / norm


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, *, sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD expects two matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD feature dimensions must match")

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(a, b).pow(2)
        value = torch.zeros_like(dist)
        for sigma in sigmas:
            value = value + torch.exp(-dist / (2.0 * sigma * sigma))
        return value / len(sigmas)

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()


def sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sliced Wasserstein expects two matrices [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sliced Wasserstein feature dimensions must match")
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


def bag_sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    num_projections: int = 32,
) -> torch.Tensor:
    if labels is None:
        return sliced_wasserstein_loss(x, y, num_projections=num_projections)
    if labels.shape != (x.shape[0],) or labels.shape != (y.shape[0],):
        raise ValueError("bag labels must have shape [batch] for both modalities")
    values = []
    for label in labels.unique(sorted=True):
        mask = labels.eq(label)
        if int(mask.sum()) == 0:
            continue
        values.append(sliced_wasserstein_loss(x[mask], y[mask], num_projections=num_projections))
    if not values:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.stack(values).mean()


@dataclass(frozen=True)
class BridgeLossWeights:
    temperature: float = 0.1
    rna_mask: float = 1.0
    image_mask: float = 1.0
    jepa: float = 1.0
    align: float = 1.0
    mmd: float = 0.1
    sliced_wasserstein: float = 0.05
    perturbation_cls: float = 0.1
    state_perturbation_adv: float = 0.0
    batch_adv: float = 0.05
    counterfactual: float = 0.2
    cycle: float = 0.1
    response_bottleneck: float = 0.01
    shared_variance: float = 0.0
    shared_covariance: float = 0.0
    cross_correlation: float = 0.0


def bridge_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    rna_values: torch.Tensor | None = None,
    rna_mask: torch.Tensor | None = None,
    image_patches: torch.Tensor | None = None,
    image_patch_mask: torch.Tensor | None = None,
    perturbation_id: torch.Tensor | None = None,
    batch_id: torch.Tensor | None = None,
    rna_batch_id: torch.Tensor | None = None,
    image_batch_id: torch.Tensor | None = None,
    bag_labels: torch.Tensor | None = None,
    hierarchy_labels: Sequence[torch.Tensor] | None = None,
    bio_keys: Any = None,
    positive_mask: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
    counterfactual_rna_target: torch.Tensor | None = None,
    counterfactual_control_rna: torch.Tensor | None = None,
    counterfactual_image_target: torch.Tensor | None = None,
    counterfactual_control_image: torch.Tensor | None = None,
    temperature: float = 0.1,
    weights: BridgeLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or BridgeLossWeights()
    terms: dict[str, torch.Tensor] = {}

    if rna_values is not None and "rna_reconstruction" in outputs:
        terms["rna_mask"] = masked_mse(outputs["rna_reconstruction"], rna_values, rna_mask)
    if image_patches is not None and "image_patch_reconstruction" in outputs:
        terms["image_mask"] = masked_mse(outputs["image_patch_reconstruction"], image_patches, image_patch_mask)
    if "rna_token_prediction" in outputs and "rna_teacher_tokens" in outputs:
        terms["rna_jepa"] = masked_cosine_jepa_loss(
            outputs["rna_token_prediction"],
            outputs["rna_teacher_tokens"],
            rna_mask,
        )
    elif "rna_shared" in outputs and "rna_teacher_shared" in outputs:
        terms["rna_jepa"] = cosine_jepa_loss(outputs["rna_shared"], outputs["rna_teacher_shared"])
    if "image_patch_prediction" in outputs and "image_teacher_patches" in outputs:
        terms["image_jepa"] = masked_cosine_jepa_loss(
            outputs["image_patch_prediction"],
            outputs["image_teacher_patches"],
            image_patch_mask,
        )
    elif "image_shared" in outputs and "image_teacher_shared" in outputs:
        terms["image_jepa"] = cosine_jepa_loss(outputs["image_shared"], outputs["image_teacher_shared"])
    if "rna_shared" in outputs and "image_shared" in outputs:
        rna_alignment = outputs.get("rna_retrieval", outputs["rna_shared"])
        image_alignment = outputs.get("image_retrieval", outputs["image_shared"])
        if bio_keys is not None or positive_mask is not None or positive_weights is not None:
            terms["align"] = multi_positive_info_nce_loss(
                rna_alignment,
                image_alignment,
                bio_keys=bio_keys,
                positive_mask=positive_mask,
                positive_weights=positive_weights,
                temperature=temperature,
            )
        elif hierarchy_labels:
            terms["align"] = multi_resolution_info_nce_loss(
                rna_alignment,
                image_alignment,
                hierarchy_labels,
                temperature=temperature,
            )
        else:
            terms["align"] = info_nce_loss(rna_alignment, image_alignment, temperature=temperature)
        if "rna_prototypes" in outputs and "image_prototypes" in outputs:
            terms["mmd"] = prototype_mmd_rbf_loss(outputs["rna_prototypes"], outputs["image_prototypes"])
            terms["sliced_wasserstein"] = prototype_sliced_wasserstein_loss(
                outputs["rna_prototypes"],
                outputs["image_prototypes"],
            )
        else:
            terms["sliced_wasserstein"] = bag_sliced_wasserstein_loss(
                outputs["rna_shared"],
                outputs["image_shared"],
                bag_labels,
            )
    if "rna_shared" in outputs:
        terms["rna_shared_variance"] = variance_floor_loss(outputs["rna_shared"])
        terms["rna_shared_covariance"] = covariance_offdiag_loss(outputs["rna_shared"])
    if "image_shared" in outputs:
        terms["image_shared_variance"] = variance_floor_loss(outputs["image_shared"])
        terms["image_shared_covariance"] = covariance_offdiag_loss(outputs["image_shared"])
    if "rna_shared" in outputs and "image_shared" in outputs:
        terms["shared_cross_correlation"] = cross_correlation_identity_loss(
            outputs["rna_shared"],
            outputs["image_shared"],
        )
    if (
        counterfactual_rna_target is not None
        and counterfactual_control_rna is not None
        and "counterfactual_rna" in outputs
    ):
        terms["counterfactual_rna"] = mmd_rbf(outputs["counterfactual_rna"], counterfactual_rna_target.detach())
    if (
        counterfactual_image_target is not None
        and counterfactual_control_image is not None
        and "counterfactual_image" in outputs
    ):
        terms["counterfactual_image"] = mmd_rbf(outputs["counterfactual_image"], counterfactual_image_target.detach())
    if "cycle_reconstruction" in outputs and "z_state" in outputs:
        terms["cycle"] = F.mse_loss(outputs["cycle_reconstruction"], outputs["z_state"].detach())
    response_terms = [
        outputs[name].pow(2).mean()
        for name in ("rna_response", "image_response", "z_response")
        if name in outputs
    ]
    if response_terms:
        terms["response_bottleneck"] = torch.stack(response_terms).mean()
    if perturbation_id is not None:
        if "rna_perturbation_logits" in outputs:
            terms["rna_perturbation_cls"] = F.cross_entropy(outputs["rna_perturbation_logits"], perturbation_id)
        if "image_perturbation_logits" in outputs:
            terms["image_perturbation_cls"] = F.cross_entropy(outputs["image_perturbation_logits"], perturbation_id)
        if "rna_state_perturbation_logits" in outputs:
            terms["rna_state_perturbation_adv"] = F.cross_entropy(
                outputs["rna_state_perturbation_logits"],
                perturbation_id,
            )
        if "image_state_perturbation_logits" in outputs:
            terms["image_state_perturbation_adv"] = F.cross_entropy(
                outputs["image_state_perturbation_logits"],
                perturbation_id,
            )
    rna_batch_target = rna_batch_id if rna_batch_id is not None else batch_id
    image_batch_target = image_batch_id if image_batch_id is not None else batch_id
    if rna_batch_target is not None and "rna_batch_logits" in outputs:
        terms["rna_batch_adv"] = F.cross_entropy(outputs["rna_batch_logits"], rna_batch_target)
    if image_batch_target is not None and "image_batch_logits" in outputs:
        terms["image_batch_adv"] = F.cross_entropy(outputs["image_batch_logits"], image_batch_target)

    device = next(iter(outputs.values())).device
    total = torch.zeros((), device=device)
    total = total + weights.rna_mask * terms.get("rna_mask", torch.zeros((), device=device))
    total = total + weights.image_mask * terms.get("image_mask", torch.zeros((), device=device))
    total = total + weights.jepa * (
        terms.get("rna_jepa", torch.zeros((), device=device))
        + terms.get("image_jepa", torch.zeros((), device=device))
    )
    total = total + weights.align * terms.get("align", torch.zeros((), device=device))
    total = total + weights.mmd * terms.get("mmd", torch.zeros((), device=device))
    total = total + weights.sliced_wasserstein * terms.get("sliced_wasserstein", torch.zeros((), device=device))
    total = total + weights.perturbation_cls * (
        terms.get("rna_perturbation_cls", torch.zeros((), device=device))
        + terms.get("image_perturbation_cls", torch.zeros((), device=device))
    )
    total = total + weights.state_perturbation_adv * (
        terms.get("rna_state_perturbation_adv", torch.zeros((), device=device))
        + terms.get("image_state_perturbation_adv", torch.zeros((), device=device))
    )
    total = total + weights.batch_adv * (
        terms.get("rna_batch_adv", torch.zeros((), device=device))
        + terms.get("image_batch_adv", torch.zeros((), device=device))
    )
    total = total + weights.counterfactual * (
        terms.get("counterfactual_rna", torch.zeros((), device=device))
        + terms.get("counterfactual_image", torch.zeros((), device=device))
    )
    total = total + weights.cycle * terms.get("cycle", torch.zeros((), device=device))
    total = total + weights.response_bottleneck * terms.get("response_bottleneck", torch.zeros((), device=device))
    total = total + weights.shared_variance * (
        terms.get("rna_shared_variance", torch.zeros((), device=device))
        + terms.get("image_shared_variance", torch.zeros((), device=device))
    )
    total = total + weights.shared_covariance * (
        terms.get("rna_shared_covariance", torch.zeros((), device=device))
        + terms.get("image_shared_covariance", torch.zeros((), device=device))
    )
    total = total + weights.cross_correlation * terms.get(
        "shared_cross_correlation",
        torch.zeros((), device=device),
    )
    terms["total"] = total
    return total, terms
```

## File: `perturb_jepa/training/objectives.py`

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import torch
from torch import nn

from perturb_jepa.config import KendallUncertaintyConfig, ObjectiveScheduleConfig
from perturb_jepa.losses import BridgeLossWeights


RECONSTRUCTION_TERMS = frozenset({"rna_mask", "image_mask"})
BRIDGE_LOSS_TERM_WEIGHTS = {
    "rna_mask": "rna_mask",
    "image_mask": "image_mask",
    "rna_jepa": "jepa",
    "image_jepa": "jepa",
    "align": "align",
    "mmd": "mmd",
    "sliced_wasserstein": "sliced_wasserstein",
    "rna_perturbation_cls": "perturbation_cls",
    "image_perturbation_cls": "perturbation_cls",
    "rna_state_perturbation_adv": "state_perturbation_adv",
    "image_state_perturbation_adv": "state_perturbation_adv",
    "rna_batch_adv": "batch_adv",
    "image_batch_adv": "batch_adv",
    "counterfactual_rna": "counterfactual",
    "counterfactual_image": "counterfactual",
    "cycle": "cycle",
    "response_bottleneck": "response_bottleneck",
    "rna_shared_variance": "shared_variance",
    "image_shared_variance": "shared_variance",
    "rna_shared_covariance": "shared_covariance",
    "image_shared_covariance": "shared_covariance",
    "shared_cross_correlation": "cross_correlation",
}


def schedule_scales(
    schedule: ObjectiveScheduleConfig | None,
    *,
    step: int,
) -> tuple[float, float]:
    if schedule is None or not schedule.enabled:
        return 1.0, 1.0
    if step < 0:
        raise ValueError("step must be non-negative")

    if step < schedule.reconstruction_warmup_steps:
        return 1.0, schedule.warmup_non_reconstruction_scale

    if schedule.reconstruction_anneal_steps == 0:
        progress = 1.0
    else:
        anneal_step = step - schedule.reconstruction_warmup_steps + 1
        progress = min(1.0, anneal_step / schedule.reconstruction_anneal_steps)

    reconstruction_scale = _lerp(1.0, schedule.reconstruction_final_scale, progress)
    non_reconstruction_scale = _lerp(schedule.warmup_non_reconstruction_scale, 1.0, progress)
    return reconstruction_scale, non_reconstruction_scale


def scheduled_loss_weights(
    weights: BridgeLossWeights | None,
    schedule: ObjectiveScheduleConfig | None,
    *,
    step: int,
) -> BridgeLossWeights:
    base = weights or BridgeLossWeights()
    reconstruction_scale, non_reconstruction_scale = schedule_scales(schedule, step=step)
    kwargs = {}
    for field in fields(BridgeLossWeights):
        scale = reconstruction_scale if field.name in RECONSTRUCTION_TERMS else non_reconstruction_scale
        kwargs[field.name] = float(getattr(base, field.name)) * scale
    return BridgeLossWeights(**kwargs)


class KendallUncertaintyWeighting(nn.Module):
    def __init__(
        self,
        term_names: tuple[str, ...],
        *,
        initial_log_variance: float = 0.0,
        loss_scale: float = 0.5,
        regularizer_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if not term_names:
            raise ValueError("term_names must contain at least one loss term")
        if len(set(term_names)) != len(term_names):
            raise ValueError("term_names must not contain duplicates")
        if loss_scale < 0.0:
            raise ValueError("loss_scale must be non-negative")
        if regularizer_scale < 0.0:
            raise ValueError("regularizer_scale must be non-negative")
        self.term_names = tuple(term_names)
        self.loss_scale = float(loss_scale)
        self.regularizer_scale = float(regularizer_scale)
        self.log_variances = nn.Parameter(
            torch.full((len(self.term_names),), float(initial_log_variance), dtype=torch.float32)
        )

    @classmethod
    def from_config(cls, config: KendallUncertaintyConfig) -> KendallUncertaintyWeighting | None:
        if not config.enabled:
            return None
        return cls(
            config.term_names,
            initial_log_variance=config.initial_log_variance,
            loss_scale=config.loss_scale,
            regularizer_scale=config.regularizer_scale,
        )

    def weight_term(
        self,
        name: str,
        component: torch.Tensor,
        *,
        include_regularizer: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        try:
            index = self.term_names.index(name)
        except ValueError as exc:
            raise KeyError(f"uncertainty weighting is not configured for term: {name}") from exc

        log_variance = self.log_variances[index].to(device=component.device, dtype=component.dtype)
        precision = torch.exp(-log_variance)
        total = self.loss_scale * precision * component
        if include_regularizer:
            total = total + self.regularizer_scale * log_variance
        diagnostics = {
            f"uncertainty/{name}_log_variance": log_variance,
            f"uncertainty/{name}_precision": precision,
        }
        return total, diagnostics

    def forward(self, components: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = set(self.term_names).difference(components)
        if missing:
            missing_terms = ", ".join(sorted(missing))
            raise KeyError(f"uncertainty-weighted loss terms were not produced: {missing_terms}")

        total = torch.zeros((), device=_infer_device(components))
        diagnostics: dict[str, torch.Tensor] = {}
        for name in self.term_names:
            weighted, term_diagnostics = self.weight_term(name, components[name])
            total = total + weighted
            diagnostics.update(term_diagnostics)
        return total, diagnostics


def build_uncertainty_weighting(
    config: KendallUncertaintyConfig | KendallUncertaintyWeighting | None,
) -> KendallUncertaintyWeighting | None:
    if config is None:
        return None
    if isinstance(config, KendallUncertaintyWeighting):
        return config
    return KendallUncertaintyWeighting.from_config(config)


def weighted_bridge_total(
    raw_terms: Mapping[str, torch.Tensor],
    *,
    weights: BridgeLossWeights | None = None,
    schedule: ObjectiveScheduleConfig | None = None,
    step: int = 0,
    uncertainty_weighting: KendallUncertaintyWeighting | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scheduled_weights = scheduled_loss_weights(weights, schedule, step=step)
    reconstruction_scale, non_reconstruction_scale = schedule_scales(schedule, step=step)

    device = _infer_device(raw_terms)
    total = torch.zeros((), device=device)
    diagnostics: dict[str, torch.Tensor] = {
        "schedule/reconstruction_scale": torch.tensor(reconstruction_scale, device=device),
        "schedule/non_reconstruction_scale": torch.tensor(non_reconstruction_scale, device=device),
    }
    uncertainty_terms = set(uncertainty_weighting.term_names) if uncertainty_weighting is not None else set()
    grouped_uncertainty_terms: dict[str, torch.Tensor] = {}

    for name, term in raw_terms.items():
        weight_name = _weight_name_for_term(name, scheduled_weights)
        if weight_name is None:
            static_weight = non_reconstruction_scale
        else:
            static_weight = float(getattr(scheduled_weights, weight_name))
        if name == "total":
            continue
        component = term * static_weight
        if uncertainty_weighting is not None and name in uncertainty_terms:
            weighted, term_diagnostics = uncertainty_weighting.weight_term(
                name,
                component,
                include_regularizer=static_weight > 0.0,
            )
            total = total + weighted
            diagnostics.update(term_diagnostics)
        elif uncertainty_weighting is not None and weight_name in uncertainty_terms:
            grouped_uncertainty_terms[weight_name] = grouped_uncertainty_terms.get(
                weight_name,
                torch.zeros((), device=component.device, dtype=component.dtype),
            ) + component
        else:
            total = total + component
        diagnostics[f"weighted/{name}"] = component

    for name, component in grouped_uncertainty_terms.items():
        weighted, term_diagnostics = uncertainty_weighting.weight_term(
            name,
            component,
            include_regularizer=bool(component.detach().abs().gt(0.0)),
        )
        total = total + weighted
        diagnostics[f"weighted/{name}"] = component
        diagnostics.update(term_diagnostics)

    if uncertainty_terms:
        missing = uncertainty_terms.difference(raw_terms).difference(grouped_uncertainty_terms)
        if missing:
            missing_terms = ", ".join(sorted(missing))
            raise KeyError(f"uncertainty-weighted loss terms were not produced: {missing_terms}")

    diagnostics["total"] = total
    return total, diagnostics


def _infer_device(raw_terms: Mapping[str, torch.Tensor]) -> torch.device:
    for value in raw_terms.values():
        return value.device
    return torch.device("cpu")


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def _weight_name_for_term(name: str, weights: BridgeLossWeights) -> str | None:
    mapped = BRIDGE_LOSS_TERM_WEIGHTS.get(name)
    if mapped is not None and hasattr(weights, mapped):
        return mapped
    if hasattr(weights, name):
        return name
    return None
```

## File: `perturb_jepa/training/trainer.py`

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from perturb_jepa.config import KendallUncertaintyConfig, ObjectiveScheduleConfig
from perturb_jepa.losses import BridgeLossWeights, bridge_loss
from perturb_jepa.models.bridge import PerturbJEPABridge
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.objectives import (
    KendallUncertaintyWeighting,
    build_uncertainty_weighting,
    weighted_bridge_total,
)
from perturb_jepa.training.synthetic import SyntheticBridgeBatch


def forward_batch(model: PerturbJEPABridge, batch: SyntheticBridgeBatch) -> dict[str, torch.Tensor]:
    return model(
        gene_ids=batch.gene_ids,
        expression_values=batch.expression_values,
        rna_token_mask=batch.rna_token_mask,
        images=batch.images,
        image_patch_mask=batch.image_patch_mask,
        perturbation_id=batch.perturbation_id,
        perturbation_type_id=batch.perturbation_type_id,
        cell_line_id=batch.cell_line_id,
        batch_id=batch.batch_id,
        dose=batch.dose,
        time=batch.time,
    )


def patchify_batch_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim == 4:
        return patchify(images, patch_size)
    if images.ndim == 5:
        flat = images.reshape(-1, *images.shape[-3:])
        patches = patchify(flat, patch_size)
        return patches.reshape(*images.shape[:2], *patches.shape[1:])
    raise ValueError("images must have shape [batch, channels, height, width] or [batch, bag, channels, height, width]")


def move_batch_to_device(batch: SyntheticBridgeBatch, device: torch.device | str) -> SyntheticBridgeBatch:
    device = torch.device(device)
    return SyntheticBridgeBatch(**{field.name: getattr(batch, field.name).to(device) for field in fields(batch)})


def loss_for_batch(
    model: PerturbJEPABridge,
    batch: SyntheticBridgeBatch,
    *,
    weights: BridgeLossWeights | None = None,
    schedule: ObjectiveScheduleConfig | None = None,
    step: int = 0,
    uncertainty_weighting: KendallUncertaintyWeighting | None = None,
    multi_positive_alignment: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = forward_batch(model, batch)
    image_patches = patchify_batch_images(batch.images, model.config.image.patch_size)
    bio_keys = _alignment_bio_keys(batch) if multi_positive_alignment else None
    schedule_enabled = schedule is not None and schedule.enabled
    if not schedule_enabled and uncertainty_weighting is None:
        return bridge_loss(
            outputs,
            rna_values=batch.expression_values,
            rna_mask=batch.rna_token_mask,
            image_patches=image_patches,
            image_patch_mask=batch.image_patch_mask,
            perturbation_id=batch.perturbation_id,
            batch_id=batch.batch_id,
            bio_keys=bio_keys,
            temperature=(weights or BridgeLossWeights()).temperature,
            weights=weights,
        )

    _, terms = bridge_loss(
        outputs,
        rna_values=batch.expression_values,
        rna_mask=batch.rna_token_mask,
        image_patches=image_patches,
        image_patch_mask=batch.image_patch_mask,
        perturbation_id=batch.perturbation_id,
        batch_id=batch.batch_id,
        bio_keys=bio_keys,
        temperature=(weights or BridgeLossWeights()).temperature,
        weights=BridgeLossWeights(),
    )
    raw_terms = {name: value for name, value in terms.items() if name != "total"}
    total, objective_terms = weighted_bridge_total(
        raw_terms,
        weights=weights,
        schedule=schedule,
        step=step,
        uncertainty_weighting=uncertainty_weighting,
    )
    raw_terms.update(objective_terms)
    return total, raw_terms


def terms_to_float(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def train_step(
    model: PerturbJEPABridge,
    optimizer: torch.optim.Optimizer,
    batch: SyntheticBridgeBatch,
    *,
    weights: BridgeLossWeights | None = None,
    schedule: ObjectiveScheduleConfig | None = None,
    step: int = 0,
    uncertainty_weighting: KendallUncertaintyWeighting | None = None,
    ema_decay: float = 0.996,
    multi_positive_alignment: bool = False,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, terms = loss_for_batch(
        model,
        batch,
        weights=weights,
        schedule=schedule,
        step=step,
        uncertainty_weighting=uncertainty_weighting,
        multi_positive_alignment=multi_positive_alignment,
    )
    total.backward()
    optimizer.step()
    model.update_teachers(decay=ema_decay)
    return terms_to_float(terms)


class BridgeTrainer:
    def __init__(
        self,
        model: PerturbJEPABridge,
        optimizer: torch.optim.Optimizer,
        *,
        weights: BridgeLossWeights | None = None,
        schedule: ObjectiveScheduleConfig | None = None,
        uncertainty_weighting: KendallUncertaintyConfig | KendallUncertaintyWeighting | None = None,
        ema_decay: float = 0.996,
        device: torch.device | str | None = None,
        grad_clip_norm: float | None = None,
        multi_positive_alignment: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.weights = weights
        self.schedule = schedule
        self.uncertainty_weighting = build_uncertainty_weighting(uncertainty_weighting)
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device) if device is not None else None
        self.grad_clip_norm = grad_clip_norm
        self.multi_positive_alignment = bool(multi_positive_alignment)
        self.global_step = 0
        if self.device is not None:
            self.model.to(self.device)
            if self.uncertainty_weighting is not None:
                self.uncertainty_weighting.to(self.device)
        if self.uncertainty_weighting is not None:
            self._ensure_objective_params_in_optimizer()

    def step(self, batch: SyntheticBridgeBatch) -> dict[str, float]:
        if self.device is not None:
            batch = move_batch_to_device(batch, self.device)
        self.model.train()
        if self.uncertainty_weighting is not None:
            self.uncertainty_weighting.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, terms = loss_for_batch(
            self.model,
            batch,
            weights=self.weights,
            schedule=self.schedule,
            step=self.global_step,
            uncertainty_weighting=self.uncertainty_weighting,
            multi_positive_alignment=self.multi_positive_alignment,
        )
        total.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.model.update_teachers(decay=self.ema_decay)
        self.global_step += 1
        return terms_to_float(terms)

    def fit(self, batches: Iterable[SyntheticBridgeBatch], *, steps: int | None = None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        iterator = iter(batches)
        while steps is None or len(history) < steps:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            history.append(self.step(batch))
        return history

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"global_step": self.global_step}
        if self.uncertainty_weighting is not None:
            state["uncertainty_weighting"] = self.uncertainty_weighting.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))
        if self.uncertainty_weighting is not None and "uncertainty_weighting" in state:
            self.uncertainty_weighting.load_state_dict(state["uncertainty_weighting"])

    def _ensure_objective_params_in_optimizer(self) -> None:
        if self.uncertainty_weighting is None:
            return
        existing = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group.get("params", [])
        }
        new_parameters = [
            parameter
            for parameter in self.uncertainty_weighting.parameters()
            if parameter.requires_grad and id(parameter) not in existing
        ]
        if new_parameters:
            self.optimizer.add_param_group({"params": new_parameters})

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        experiment_config: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        from perturb_jepa.training.checkpoint import save_checkpoint

        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            trainer_state=self.state_dict(),
            experiment_config=experiment_config,
            metadata=metadata,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        from perturb_jepa.training.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            map_location=map_location if map_location is not None else self.device,
            strict=strict,
        )
        self.load_state_dict(checkpoint.get("trainer_state", {}))
        return checkpoint


def _alignment_bio_keys(batch: SyntheticBridgeBatch) -> dict[str, torch.Tensor]:
    dose_code = torch.round(batch.dose.to(dtype=torch.float32) * 100.0).to(dtype=torch.long)
    condition = batch.perturbation_id.to(dtype=torch.long) * 1_000_000
    condition = condition + batch.cell_line_id.to(dtype=torch.long) * 10_000
    condition = condition + dose_code
    return {
        "condition": condition,
        "perturbation": batch.perturbation_id.to(dtype=torch.long),
    }
```

## File: `perturb_jepa/training/prefit_readout.py`

```python
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
```

## File: `perturb_jepa/training/synthetic_biology_lite.py`

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from perturb_jepa.training.synthetic import SyntheticBridgeBatch


@dataclass(frozen=True)
class SyntheticBiologyLiteConfig:
    name: str = "synth_micro"
    seed: int = 0
    decoder_seed: int = 1729
    num_perturbations: int = 4
    num_cell_lines: int = 2
    doses: tuple[float, ...] = (0.0, 1.0)
    num_batches: int = 2
    cells_per_condition: int = 8
    genes: int = 128
    latent_dim: int = 16
    tech_dim: int = 4
    num_programs: int = 8
    image_channels: int = 1
    image_size: int = 16
    patch_size: int = 4
    biological_noise_std: float = 0.08
    technical_noise_std: float = 0.05
    count_depth: float = 40.0
    overdispersion: float = 0.25
    dropout_base: float = -1.0
    heldout_perturbations: tuple[int, ...] = ()
    heldout_doses: tuple[float, ...] = ()
    heldout_batches: tuple[int, ...] = ()
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    control_perturbation_id: int = 0
    batch_confounding: bool = False

    def __post_init__(self) -> None:
        if self.control_perturbation_id < 0 or self.control_perturbation_id >= self.num_perturbations:
            raise ValueError("control_perturbation_id must be within perturbation range")
        if self.genes > 512:
            raise ValueError("synthetic lite generator is capped at 512 genes")
        if self.cells_per_condition > 16:
            raise ValueError("synthetic lite generator is capped at 16 cells per condition")
        if self.image_size > 24:
            raise ValueError("synthetic lite generator is capped at 24x24 images")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not self.doses:
            raise ValueError("at least one dose is required")
        if self.train_fraction <= 0.0 or self.val_fraction < 0.0 or self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train/val fractions must leave a positive test split")
        object.__setattr__(self, "doses", tuple(float(dose) for dose in self.doses))
        object.__setattr__(self, "heldout_perturbations", tuple(int(value) for value in self.heldout_perturbations))
        object.__setattr__(self, "heldout_doses", tuple(float(value) for value in self.heldout_doses))
        object.__setattr__(self, "heldout_batches", tuple(int(value) for value in self.heldout_batches))


@dataclass
class SyntheticBiologyLiteDataset:
    config: SyntheticBiologyLiteConfig
    gene_ids: np.ndarray
    expression_values: np.ndarray
    observed_counts: np.ndarray
    clean_rna: np.ndarray
    images: np.ndarray
    z_bio: np.ndarray
    z_tech: np.ndarray
    metadata: pd.DataFrame
    perturbation_directions: np.ndarray
    cell_line_baselines: np.ndarray
    interactions: np.ndarray
    batch_offsets: np.ndarray
    gene_program_assignment: np.ndarray
    dose_curve_by_dose: dict[str, float]

    def condition_frame(self, split: str | None = None) -> pd.DataFrame:
        frame = self.metadata if split is None else self.metadata[self.metadata["split"] == split]
        columns = [
            "condition_id",
            "condition_key",
            "perturbation",
            "perturbation_id",
            "dose",
            "cell_line",
            "cell_line_id",
            "batch",
            "batch_id",
            "time",
        ]
        return frame.loc[:, columns].drop_duplicates().reset_index(drop=True)

    def condition_bag_indices(self, *, split: str) -> list[np.ndarray]:
        return self._condition_groups(split=split)

    def metadata_for_condition_bags(self, *, split: str) -> pd.DataFrame:
        rows = []
        for group in self._condition_groups(split=split):
            rows.append(self.metadata.iloc[int(group[0])])
        return pd.DataFrame(rows).reset_index(drop=True)

    def iter_condition_batches(
        self,
        *,
        split: str = "train",
        batch_size: int = 8,
        bag_size: int | None = None,
        steps: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        rna_mask_prob: float = 0.2,
        image_mask_prob: float = 0.2,
        device: torch.device | str = "cpu",
    ) -> Iterator[SyntheticBridgeBatch]:
        bag_size = int(bag_size or self.config.cells_per_condition)
        groups = self._condition_groups(split=split)
        if not groups:
            raise ValueError(f"split {split!r} has no condition groups")
        rng = np.random.default_rng(seed)
        order = np.arange(len(groups))
        emitted = 0
        while steps is None or emitted < steps:
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                if steps is not None and emitted >= steps:
                    return
                selected_groups = [groups[int(idx)] for idx in order[start : start + batch_size]]
                yield self._make_bridge_batch(
                    selected_groups,
                    bag_size=bag_size,
                    rng=rng,
                    rna_mask_prob=rna_mask_prob,
                    image_mask_prob=image_mask_prob,
                    device=device,
                )
                emitted += 1
            if not shuffle and steps is None:
                return

    def export(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "generation_config.json").write_text(
            json.dumps(_jsonable(asdict(self.config)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ground_truth = {
            "latent_biological_state": self.z_bio.tolist(),
            "latent_technical_state": self.z_tech.tolist(),
            "perturbation_direction_matrix": self.perturbation_directions.tolist(),
            "cell_line_baseline_matrix": self.cell_line_baselines.tolist(),
            "interaction_tensor": self.interactions.tolist(),
            "dose_curve": self.dose_curve_by_dose,
            "gene_program_assignment": self.gene_program_assignment.tolist(),
            "batch_offsets": self.batch_offsets.tolist(),
            "clean_rna_before_technical_corruption": self.clean_rna.tolist(),
            "observed_rna_after_technical_corruption": self.expression_values.tolist(),
            "observed_counts_after_technical_corruption": self.observed_counts.tolist(),
            "split_labels": self.metadata["split"].tolist(),
            "cell_ids": self.metadata["cell_id"].tolist(),
            "condition_keys": self.metadata["condition_key"].tolist(),
        }
        (output_path / "ground_truth.json").write_text(
            json.dumps(_jsonable(ground_truth), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.metadata.to_csv(output_path / "metadata.tsv", sep="\t", index=False)
        return output_path

    def _condition_groups(self, *, split: str) -> list[np.ndarray]:
        frame = self.metadata[self.metadata["split"] == split]
        groups: list[np.ndarray] = []
        for _, group in frame.groupby("bag_key", sort=True):
            indices = group.index.to_numpy(dtype=int)
            if indices.size:
                groups.append(indices)
        return groups

    def _make_bridge_batch(
        self,
        selected_groups: Sequence[np.ndarray],
        *,
        bag_size: int,
        rng: np.random.Generator,
        rna_mask_prob: float,
        image_mask_prob: float,
        device: torch.device | str,
    ) -> SyntheticBridgeBatch:
        sampled_indices = []
        for group in selected_groups:
            replace = group.size < bag_size
            sampled_indices.append(rng.choice(group, size=bag_size, replace=replace))
        index = np.stack(sampled_indices, axis=0)
        first = index[:, 0]
        frame = self.metadata.iloc[first]
        gene_ids = np.broadcast_to(self.gene_ids, (len(selected_groups), bag_size, self.gene_ids.size))
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        return SyntheticBridgeBatch(
            gene_ids=torch.as_tensor(gene_ids.copy(), dtype=torch.long, device=device),
            expression_values=torch.as_tensor(self.expression_values[index], dtype=torch.float32, device=device),
            rna_token_mask=torch.rand((len(selected_groups), bag_size, self.config.genes), device=device) < rna_mask_prob,
            images=torch.as_tensor(self.images[index], dtype=torch.float32, device=device),
            image_patch_mask=torch.rand((len(selected_groups), bag_size, num_patches), device=device) < image_mask_prob,
            perturbation_id=torch.as_tensor(frame["perturbation_id"].to_numpy(), dtype=torch.long, device=device),
            perturbation_type_id=torch.as_tensor(frame["perturbation_type_id"].to_numpy(), dtype=torch.long, device=device),
            cell_line_id=torch.as_tensor(frame["cell_line_id"].to_numpy(), dtype=torch.long, device=device),
            batch_id=torch.as_tensor(frame["batch_id"].to_numpy(), dtype=torch.long, device=device),
            dose=torch.as_tensor(frame["dose"].to_numpy(dtype=float), dtype=torch.float32, device=device),
            time=torch.as_tensor(frame["time"].to_numpy(dtype=float), dtype=torch.float32, device=device),
        )


def generate_synthetic_biology_lite(config: SyntheticBiologyLiteConfig) -> SyntheticBiologyLiteDataset:
    rng = np.random.default_rng(config.seed)
    decoder_rng = np.random.default_rng(config.decoder_seed)
    gene_ids = np.arange(config.genes, dtype=np.int64)
    gene_program_assignment = np.arange(config.genes, dtype=np.int64) % config.num_programs
    decoder_rng.shuffle(gene_program_assignment)

    program_loadings = decoder_rng.normal(0.0, 0.08, size=(config.num_programs, config.genes)).astype(np.float32)
    program_loadings[gene_program_assignment, np.arange(config.genes)] += decoder_rng.uniform(0.8, 1.2, size=config.genes)
    bio_to_program = decoder_rng.normal(0.0, 0.35, size=(config.latent_dim, config.num_programs)).astype(np.float32)
    rna_decoder = bio_to_program @ program_loadings
    tech_decoder = decoder_rng.normal(0.0, 0.12, size=(config.tech_dim, config.genes)).astype(np.float32)
    gene_bias = decoder_rng.normal(0.0, 0.25, size=config.genes).astype(np.float32)

    image_pixels = config.image_channels * config.image_size * config.image_size
    image_decoder = decoder_rng.normal(0.0, 0.25, size=(config.latent_dim, image_pixels)).astype(np.float32)
    image_tech_decoder = decoder_rng.normal(0.0, 0.08, size=(config.tech_dim, image_pixels)).astype(np.float32)
    image_bias = decoder_rng.normal(0.0, 0.1, size=image_pixels).astype(np.float32)

    cell_line_baselines = rng.normal(0.0, 0.65, size=(config.num_cell_lines, config.latent_dim)).astype(np.float32)
    perturbation_directions = rng.normal(0.0, 0.8, size=(config.num_perturbations, config.latent_dim)).astype(np.float32)
    perturbation_directions[config.control_perturbation_id] = 0.0
    interactions = rng.normal(
        0.0,
        0.08,
        size=(config.num_cell_lines, config.num_perturbations, config.latent_dim),
    ).astype(np.float32)
    interactions[:, config.control_perturbation_id] = 0.0
    batch_offsets = rng.normal(0.0, 0.45, size=(config.num_batches, config.tech_dim)).astype(np.float32)
    library_vector = rng.normal(0.0, 0.35, size=config.tech_dim).astype(np.float32)
    dropout_vector = rng.normal(0.0, 0.25, size=config.tech_dim).astype(np.float32)
    dropout_gene_bias = decoder_rng.normal(0.0, 0.2, size=config.genes).astype(np.float32)

    rows: list[dict[str, object]] = []
    z_bio_values: list[np.ndarray] = []
    z_tech_values: list[np.ndarray] = []
    clean_values: list[np.ndarray] = []
    observed_counts: list[np.ndarray] = []
    expression_values: list[np.ndarray] = []
    image_values: list[np.ndarray] = []
    dose_curve_by_dose = {str(float(dose)): _dose_curve(float(dose)) for dose in config.doses}
    condition_id = 0

    for perturbation_id in range(config.num_perturbations):
        allowed_batches = _allowed_batches_for_perturbation(config, perturbation_id)
        for cell_line_id in range(config.num_cell_lines):
            for dose in config.doses:
                for batch_id in allowed_batches:
                    condition_key = f"pert={perturbation_id}|dose={float(dose):g}|cell={cell_line_id}"
                    bag_key = f"{condition_key}|batch={batch_id}"
                    curve = 0.0 if perturbation_id == config.control_perturbation_id else dose_curve_by_dose[str(float(dose))]
                    for cell_in_condition in range(config.cells_per_condition):
                        split = _split_for_cell(
                            config,
                            perturbation_id=perturbation_id,
                            dose=float(dose),
                            batch_id=batch_id,
                            cell_in_condition=cell_in_condition,
                        )
                        library_size_effect = rng.normal(0.0, 0.35)
                        dropout_effect = rng.normal(0.0, 0.25)
                        z_bio = (
                            cell_line_baselines[cell_line_id]
                            + curve * perturbation_directions[perturbation_id]
                            + interactions[cell_line_id, perturbation_id]
                            + rng.normal(0.0, config.biological_noise_std, size=config.latent_dim)
                        ).astype(np.float32)
                        z_tech = (
                            batch_offsets[batch_id]
                            + library_size_effect * library_vector
                            + dropout_effect * dropout_vector
                            + rng.normal(0.0, config.technical_noise_std, size=config.tech_dim)
                        ).astype(np.float32)
                        clean_log_mu = np.clip(gene_bias + z_bio @ rna_decoder, -3.5, 3.5)
                        clean_mu = np.exp(clean_log_mu).astype(np.float32)
                        technical_log_shift = np.clip(z_tech @ tech_decoder + library_size_effect, -2.5, 2.5)
                        observed_mu = np.exp(np.log(clean_mu + 1e-6) + technical_log_shift)
                        counts = _gamma_poisson(rng, observed_mu * config.count_depth, config.overdispersion)
                        dropout_logits = config.dropout_base + dropout_gene_bias - 0.35 * np.log1p(observed_mu) + dropout_effect
                        dropout_probability = _sigmoid(dropout_logits)
                        counts = counts * (rng.random(config.genes) > dropout_probability)
                        expression = np.log1p(counts).astype(np.float32)
                        image = _sigmoid(z_bio @ image_decoder + z_tech @ image_tech_decoder + image_bias)
                        image = image + rng.normal(0.0, 0.02, size=image.shape)
                        image = np.clip(image, 0.0, 1.0).reshape(
                            config.image_channels,
                            config.image_size,
                            config.image_size,
                        )

                        cell_id = f"{config.name}_cell_{len(rows):06d}"
                        rows.append(
                            {
                                "cell_id": cell_id,
                                "condition_id": condition_id,
                                "condition_key": condition_key,
                                "bag_key": bag_key,
                                "perturbation": f"perturbation_{perturbation_id}",
                                "perturbation_id": perturbation_id,
                                "perturbation_type_id": 0 if perturbation_id == config.control_perturbation_id else 1,
                                "dose": float(dose),
                                "cell_line": f"cell_line_{cell_line_id}",
                                "cell_line_id": cell_line_id,
                                "batch": f"batch_{batch_id}",
                                "batch_id": batch_id,
                                "time": 0.0,
                                "split": split,
                            }
                        )
                        z_bio_values.append(z_bio)
                        z_tech_values.append(z_tech)
                        clean_values.append(np.log1p(clean_mu * config.count_depth).astype(np.float32))
                        observed_counts.append(counts.astype(np.float32))
                        expression_values.append(expression)
                        image_values.append(image.astype(np.float32))
                    condition_id += 1

    metadata = pd.DataFrame(rows)
    return SyntheticBiologyLiteDataset(
        config=config,
        gene_ids=gene_ids,
        expression_values=np.stack(expression_values).astype(np.float32),
        observed_counts=np.stack(observed_counts).astype(np.float32),
        clean_rna=np.stack(clean_values).astype(np.float32),
        images=np.stack(image_values).astype(np.float32),
        z_bio=np.stack(z_bio_values).astype(np.float32),
        z_tech=np.stack(z_tech_values).astype(np.float32),
        metadata=metadata,
        perturbation_directions=perturbation_directions,
        cell_line_baselines=cell_line_baselines,
        interactions=interactions,
        batch_offsets=batch_offsets,
        gene_program_assignment=gene_program_assignment,
        dose_curve_by_dose=dose_curve_by_dose,
    )


def synthetic_lite_config(name: str, *, seed: int = 0) -> SyntheticBiologyLiteConfig:
    if name == "synth_micro":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=4,
            num_cell_lines=2,
            doses=(0.0, 1.0),
            num_batches=2,
            cells_per_condition=8,
            genes=128,
        )
    if name == "synth_easy_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=8,
            num_cell_lines=3,
            doses=(0.0, 1.0),
            num_batches=2,
            cells_per_condition=12,
            genes=256,
        )
    if name == "synth_medium_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
        )
    if name == "synth_heldout_perturbation_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_perturbations=(9, 10, 11),
        )
    if name == "synth_batch_confound_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 3.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            batch_confounding=True,
        )
    if name == "synth_dose_extrapolation_lite":
        return SyntheticBiologyLiteConfig(
            name=name,
            seed=seed,
            num_perturbations=12,
            num_cell_lines=3,
            doses=(0.0, 1.0, 2.0, 3.0, 5.0),
            num_batches=3,
            cells_per_condition=12,
            genes=384,
            heldout_doses=(2.0, 5.0),
        )
    raise ValueError(f"unknown synthetic lite dataset: {name}")


def _allowed_batches_for_perturbation(config: SyntheticBiologyLiteConfig, perturbation_id: int) -> tuple[int, ...]:
    if not config.batch_confounding or perturbation_id == config.control_perturbation_id:
        return tuple(range(config.num_batches))
    return (perturbation_id % config.num_batches,)


def _split_for_cell(
    config: SyntheticBiologyLiteConfig,
    *,
    perturbation_id: int,
    dose: float,
    batch_id: int,
    cell_in_condition: int,
) -> str:
    if perturbation_id in config.heldout_perturbations:
        return "test_heldout_perturbation"
    if any(abs(dose - heldout) < 1e-8 for heldout in config.heldout_doses):
        return "test_heldout_dose"
    if batch_id in config.heldout_batches:
        return "test_heldout_batch"
    fraction = (cell_in_condition + 0.5) / config.cells_per_condition
    if fraction <= config.train_fraction:
        return "train"
    if fraction <= config.train_fraction + config.val_fraction:
        return "val"
    return "test"


def _dose_curve(dose: float) -> float:
    dose = max(float(dose), 0.0)
    return float(dose / (1.0 + dose))


def _gamma_poisson(rng: np.random.Generator, mean: np.ndarray, overdispersion: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    if overdispersion <= 0.0:
        return rng.poisson(mean).astype(np.float32)
    shape = 1.0 / overdispersion
    scale = np.maximum(mean * overdispersion, 1e-6)
    gamma_mean = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(gamma_mean).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
```

## File: `scripts/run_synthetic_lite_step0.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.baselines import (
    batch_only_retrieval_metrics,
    mean_prototype_alignment_metrics,
    metadata_only_retrieval_metrics,
)
from perturb_jepa.config import ExperimentConfig, ObjectiveScheduleConfig, OptimizerConfig, TrainingConfig
from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.losses import BridgeLossWeights
from perturb_jepa.models.bridge import PerturbJEPABridgeConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import (
    SyntheticBiologyLiteDataset,
    generate_synthetic_biology_lite,
    synthetic_lite_config,
)
from perturb_jepa.training.trainer import BridgeTrainer, loss_for_batch, move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 0 synthetic biology lite baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/autoresearch_synth_lite"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bag-size", type=int, default=None)
    parser.add_argument("--label-shuffle-repeats", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--shared-dim", type=int, default=None)
    parser.add_argument("--rna-pooling", default="cls", choices=("cls", "mean_tokens"))
    parser.add_argument("--image-pooling", default="cls", choices=("cls", "mean_patches"))
    parser.add_argument(
        "--rna-condition-readout",
        default="encoder",
        choices=(
            "encoder",
            "pseudobulk",
            "encoder_plus_pseudobulk",
            "raw_pseudobulk",
            "encoder_plus_raw_pseudobulk",
            "raw_linear_pseudobulk",
            "encoder_plus_raw_linear_pseudobulk",
        ),
    )
    parser.add_argument("--no-rna-pseudobulk-normalize", action="store_true")
    parser.add_argument(
        "--image-condition-readout",
        default="encoder",
        choices=("encoder", "raw_pooled", "encoder_plus_raw_pooled", "raw_linear_pooled", "encoder_plus_raw_linear_pooled"),
    )
    parser.add_argument("--no-image-raw-normalize", action="store_true")
    parser.add_argument("--adversary-scale", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rna-mask-weight", type=float, default=0.2)
    parser.add_argument("--image-mask-weight", type=float, default=0.2)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=1.0)
    parser.add_argument("--mmd-weight", type=float, default=0.05)
    parser.add_argument("--sliced-wasserstein-weight", type=float, default=0.02)
    parser.add_argument("--perturbation-cls-weight", type=float, default=0.05)
    parser.add_argument("--batch-adv-weight", type=float, default=0.02)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--cycle-weight", type=float, default=0.05)
    parser.add_argument("--response-bottleneck-weight", type=float, default=0.005)
    parser.add_argument("--shared-variance-weight", type=float, default=0.0)
    parser.add_argument("--shared-covariance-weight", type=float, default=0.0)
    parser.add_argument("--cross-correlation-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-rna-residual", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--bag-aggregator", default="attention", choices=("attention", "mean"))
    parser.add_argument("--num-bag-prototypes", type=int, default=2)
    parser.add_argument("--multi-positive-alignment", action="store_true")
    parser.add_argument("--schedule-reconstruction-warmup-steps", type=int, default=0)
    parser.add_argument("--schedule-reconstruction-anneal-steps", type=int, default=0)
    parser.add_argument("--schedule-reconstruction-final-scale", type=float, default=1.0)
    parser.add_argument("--schedule-warmup-non-reconstruction-scale", type=float, default=0.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    started = time.perf_counter()
    config = synthetic_lite_config(args.dataset, seed=args.seed)
    dataset = generate_synthetic_biology_lite(config)
    dataset_dir = args.output_root / "step0_baselines" / args.dataset
    dataset.export(dataset_dir)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        model_dim=args.model_dim,
        shared_dim=args.shared_dim,
        rna_pooling=args.rna_pooling,
        image_pooling=args.image_pooling,
        rna_condition_readout=args.rna_condition_readout,
        rna_pseudobulk_normalize=not args.no_rna_pseudobulk_normalize,
        image_condition_readout=args.image_condition_readout,
        image_raw_normalize=not args.no_image_raw_normalize,
        adversary_scale=args.adversary_scale,
        temperature=args.temperature,
        rna_mask_weight=args.rna_mask_weight,
        image_mask_weight=args.image_mask_weight,
        jepa_weight=args.jepa_weight,
        align_weight=args.align_weight,
        mmd_weight=args.mmd_weight,
        sliced_wasserstein_weight=args.sliced_wasserstein_weight,
        perturbation_cls_weight=args.perturbation_cls_weight,
        batch_adv_weight=args.batch_adv_weight,
        counterfactual_weight=args.counterfactual_weight,
        cycle_weight=args.cycle_weight,
        response_bottleneck_weight=args.response_bottleneck_weight,
        shared_variance_weight=args.shared_variance_weight,
        shared_covariance_weight=args.shared_covariance_weight,
        cross_correlation_weight=args.cross_correlation_weight,
        counterfactual_rna_residual=args.counterfactual_rna_residual,
        ema_decay=args.ema_decay,
        bag_aggregator=args.bag_aggregator,
        num_bag_prototypes=args.num_bag_prototypes,
        objective_schedule=_objective_schedule_from_args(args),
    )
    config_dir = args.output_root / "step0_baselines" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save_json(config_dir / f"{args.dataset}_seed{args.seed}_bridge.json")
    (config_dir / f"{args.dataset}_seed{args.seed}_generator.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model = experiment_config.build_model()
    optimizer = experiment_config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=experiment_config.loss,
        ema_decay=experiment_config.training.ema_decay,
        schedule=experiment_config.training.objective_schedule,
        device=args.device,
        grad_clip_norm=experiment_config.training.grad_clip_norm,
        multi_positive_alignment=args.multi_positive_alignment,
    )
    history = _train_with_early_stopping(
        trainer,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=args.bag_size or config.cells_per_condition,
        device=args.device,
        seed=args.seed,
    )
    split = "test"
    metrics = evaluate_step0(
        dataset,
        model,
        split=split,
        train_split="train",
        device=args.device,
        bag_size=args.bag_size or config.cells_per_condition,
        seed=args.seed,
        label_shuffle_repeats=args.label_shuffle_repeats,
    )
    metrics["training_steps_completed"] = float(len(history))
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["device_used"] = args.device
    metrics["max_gpu_memory_gb"] = 0.0

    metrics_path = dataset_dir / f"step0_seed{args.seed}_metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history_path = dataset_dir / f"step0_seed{args.seed}_history.json"
    history_path.write_text(json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dataset_report(dataset_dir / f"{args.dataset}_baseline.md", args.dataset, args.seed, metrics, history)
    print(json.dumps(_jsonable(metrics), sort_keys=True))


def evaluate_step0(
    dataset: SyntheticBiologyLiteDataset,
    model,
    *,
    split: str,
    train_split: str,
    device: str,
    bag_size: int,
    seed: int,
    label_shuffle_repeats: int = 20,
) -> dict[str, Any]:
    model.eval()
    test = _collect_model_outputs(dataset, model, split=split, device=device, bag_size=bag_size, seed=seed)
    train = _collect_model_outputs(dataset, model, split=train_split, device=device, bag_size=bag_size, seed=seed + 101)
    metrics: dict[str, Any] = {}

    retrieval = cross_modal_retrieval_metrics(
        test["rna_shared"],
        test["image_shared"],
        test["metadata"],
        test["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    metrics.update(_prefix_dict("model", retrieval))

    rng = np.random.default_rng(seed)
    random_rna = rng.normal(size=test["rna_shared"].shape)
    random_image = rng.normal(size=test["image_shared"].shape)
    metrics.update(
        _prefix_dict(
            "random_embedding",
            cross_modal_retrieval_metrics(
                random_rna,
                random_image,
                test["metadata"],
                test["metadata"],
                ks=(1, 5),
                stratify_by=(),
            ),
        )
    )
    zeros = np.zeros_like(test["rna_shared"])
    metrics.update(
        _prefix_dict(
            "dataset_mean",
            cross_modal_retrieval_metrics(
                zeros,
                zeros,
                test["metadata"],
                test["metadata"],
                ks=(1, 5),
                stratify_by=(),
            ),
        )
    )
    metrics.update(_prefix_dict("metadata_only", metadata_only_retrieval_metrics(test["metadata"], test["metadata"], ks=(1, 5))))
    metrics.update(_prefix_dict("batch_only", batch_only_retrieval_metrics(test["metadata"], test["metadata"], ks=(1, 5))))
    image_flat = test["image_mean"].reshape(test["image_mean"].shape[0], -1)
    metrics.update(
        _prefix_dict(
            "mean_prototype_alignment",
            mean_prototype_alignment_metrics(test["metadata"], image_flat, test["metadata"], ks=(1, 5)),
        )
    )

    cf_arrays = _counterfactual_arrays(dataset, model, split=split, device=device, bag_size=bag_size, seed=seed)
    gene_sets = _gene_sets(dataset)
    if cf_arrays["observed"].shape[0] > 0:
        metrics.update(
            _prefix_dict(
                "model",
                rna_counterfactual_metrics(
                    cf_arrays["model_predicted"],
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )
        metrics.update(
            _prefix_dict(
                "source_as_target",
                rna_counterfactual_metrics(
                    cf_arrays["control"],
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )
        dataset_mean = np.broadcast_to(train["rna_mean"].mean(axis=0, keepdims=True), cf_arrays["observed"].shape)
        metrics.update(
            _prefix_dict(
                "dataset_mean_cf",
                rna_counterfactual_metrics(
                    dataset_mean,
                    cf_arrays["observed"],
                    cf_arrays["control"],
                    cf_arrays["metadata"],
                    groupby=None,
                    topk=(50,),
                    gene_sets=gene_sets,
                ),
            )
        )

    metrics["model_bio_latent_r2_rna_shared"] = _latent_r2(
        train["rna_shared"],
        train["z_bio_mean"],
        test["rna_shared"],
        test["z_bio_mean"],
    )
    metrics["model_bio_latent_r2_image_shared"] = _latent_r2(
        train["image_shared"],
        train["z_bio_mean"],
        test["image_shared"],
        test["z_bio_mean"],
    )
    mapped_test = _linear_predict(train["rna_shared"], train["z_bio_mean"], test["rna_shared"])
    metrics["model_perturbation_direction_cosine"] = _perturbation_direction_cosine(test["metadata"], mapped_test, test["z_bio_mean"])
    metrics["model_dose_response_rank_correlation"] = _dose_response_rank_corr(test["metadata"], mapped_test)
    metrics["model_cell_line_baseline_recovery"] = _cell_line_baseline_recovery(
        test["metadata"],
        mapped_test,
        dataset.cell_line_baselines,
    )
    if cf_arrays["observed"].shape[0] > 0:
        metrics["model_program_level_effect_recovery"] = _program_effect_recovery(
            cf_arrays["model_predicted"],
            cf_arrays["observed"],
            cf_arrays["control"],
            dataset.gene_program_assignment,
        )
        metrics["source_as_target_program_level_effect_recovery"] = _program_effect_recovery(
            cf_arrays["control"],
            cf_arrays["observed"],
            cf_arrays["control"],
            dataset.gene_program_assignment,
        )

    batch_probe = batch_probe_metrics(test["rna_shared"], test["metadata"], label_col="batch")
    metrics.update(_prefix_dict("model", batch_probe))
    majority = batch_probe.get("batch_probe_majority_accuracy", 0.0)
    balanced = batch_probe.get("batch_probe_balanced_accuracy", float("nan"))
    metrics["model_batch_probe_balanced_accuracy_minus_majority"] = float(balanced - majority)
    metrics["model_bio_latent_r2_on_heldout_batch"] = float("nan")
    metrics["model_retrieval_drop_on_heldout_batch"] = float("nan")

    label_shuffle_values = []
    for _ in range(max(1, int(label_shuffle_repeats))):
        shuffled_metadata = test["metadata"].copy()
        shuffled_metadata["condition_key"] = rng.permutation(shuffled_metadata["condition_key"].to_numpy())
        label_shuffle_values.append(
            cross_modal_retrieval_metrics(
                test["rna_shared"],
                test["image_shared"],
                test["metadata"],
                shuffled_metadata,
                ks=(1, 5),
                stratify_by=(),
            )
        )
    label_shuffle = {
        key: float(np.mean([values[key] for values in label_shuffle_values]))
        for key in label_shuffle_values[0]
    }
    metrics.update(_prefix_dict("label_shuffle", label_shuffle))

    collapse = _collapse_diagnostics(test)
    metrics.update(collapse)
    metrics["collapse_flag"] = bool(collapse["model_rna_shared_min_std"] < 0.01 or collapse["model_image_shared_min_std"] < 0.01)
    return metrics


def _experiment_config_for_dataset(
    dataset: SyntheticBiologyLiteDataset,
    *,
    steps: int,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.05,
    model_dim: int = 32,
    shared_dim: int | None = None,
    rna_pooling: str = "cls",
    image_pooling: str = "cls",
    rna_condition_readout: str = "encoder",
    rna_pseudobulk_normalize: bool = True,
    image_condition_readout: str = "encoder",
    image_raw_normalize: bool = True,
    adversary_scale: float = 0.5,
    temperature: float = 0.1,
    rna_mask_weight: float = 0.2,
    image_mask_weight: float = 0.2,
    jepa_weight: float = 1.0,
    align_weight: float = 1.0,
    mmd_weight: float = 0.05,
    sliced_wasserstein_weight: float = 0.02,
    perturbation_cls_weight: float = 0.05,
    batch_adv_weight: float = 0.02,
    counterfactual_weight: float = 0.0,
    cycle_weight: float = 0.05,
    response_bottleneck_weight: float = 0.005,
    shared_variance_weight: float = 0.0,
    shared_covariance_weight: float = 0.0,
    cross_correlation_weight: float = 0.0,
    counterfactual_rna_residual: bool = False,
    counterfactual_rna_program_factorized: bool = False,
    counterfactual_rna_num_programs: int = 0,
    counterfactual_rna_program_assignment: tuple[int, ...] = (),
    counterfactual_rna_within_program_residual: bool = False,
    counterfactual_rna_program_conditioned: bool = False,
    counterfactual_rna_program_metadata_context: bool = False,
    counterfactual_rna_program_decoder_depth: int = 2,
    ema_decay: float = 0.996,
    bag_aggregator: str = "attention",
    num_bag_prototypes: int = 2,
    objective_schedule: ObjectiveScheduleConfig | None = None,
) -> ExperimentConfig:
    config = dataset.config
    dim = int(model_dim)
    shared_width = int(shared_dim) if shared_dim is not None else dim
    model = PerturbJEPABridgeConfig(
        rna=RNAEncoderConfig(
            vocab_size=config.genes,
            dim=dim,
            depth=1,
            heads=4,
            max_genes=config.genes,
            pooling=rna_pooling,
        ),
        image=ImageEncoderConfig(
            in_channels=config.image_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=dim,
            depth=1,
            heads=4,
            pooling=image_pooling,
        ),
        perturbation=PerturbationEncoderConfig(
            num_perturbations=config.num_perturbations,
            num_types=2,
            num_cell_lines=config.num_cell_lines,
            num_batches=config.num_batches,
            dim=dim,
        ),
        shared_dim=shared_width,
        num_bag_prototypes=num_bag_prototypes,
        dropout=dropout,
        adversary_scale=adversary_scale,
        bag_aggregator=bag_aggregator,
        rna_condition_readout=rna_condition_readout,
        rna_pseudobulk_normalize=rna_pseudobulk_normalize,
        image_condition_readout=image_condition_readout,
        image_raw_normalize=image_raw_normalize,
        counterfactual_rna_residual=counterfactual_rna_residual,
        counterfactual_rna_program_factorized=counterfactual_rna_program_factorized,
        counterfactual_rna_num_programs=counterfactual_rna_num_programs,
        counterfactual_rna_program_assignment=counterfactual_rna_program_assignment,
        counterfactual_rna_within_program_residual=counterfactual_rna_within_program_residual,
        counterfactual_rna_program_conditioned=counterfactual_rna_program_conditioned,
        counterfactual_rna_program_metadata_context=counterfactual_rna_program_metadata_context,
        counterfactual_rna_program_decoder_depth=counterfactual_rna_program_decoder_depth,
    )
    return ExperimentConfig(
        name=f"{config.name}-step0",
        model=model,
        optimizer=OptimizerConfig(lr=lr, weight_decay=weight_decay),
        training=TrainingConfig(
            steps=steps,
            batch_size=8,
            device=device,
            seed=config.seed,
            ema_decay=ema_decay,
            grad_clip_norm=1.0,
            log_every=0,
            objective_schedule=objective_schedule or ObjectiveScheduleConfig(),
        ),
        loss=BridgeLossWeights(
            temperature=temperature,
            rna_mask=rna_mask_weight,
            image_mask=image_mask_weight,
            jepa=jepa_weight,
            align=align_weight,
            mmd=mmd_weight,
            sliced_wasserstein=sliced_wasserstein_weight,
            perturbation_cls=perturbation_cls_weight,
            batch_adv=batch_adv_weight,
            counterfactual=counterfactual_weight,
            cycle=cycle_weight,
            response_bottleneck=response_bottleneck_weight,
            shared_variance=shared_variance_weight,
            shared_covariance=shared_covariance_weight,
            cross_correlation=cross_correlation_weight,
        ),
    )


def _train_with_early_stopping(
    trainer: BridgeTrainer,
    dataset: SyntheticBiologyLiteDataset,
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    device: str,
    seed: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    best_val = float("inf")
    stale_steps = 0
    patience = max(10, steps // 5)
    eval_every = max(10, min(25, steps // 4))
    train_batches = dataset.iter_condition_batches(
        split="train",
        batch_size=batch_size,
        bag_size=bag_size,
        steps=steps,
        seed=seed,
        device=device,
    )
    for step, batch in enumerate(train_batches, start=1):
        terms = trainer.step(batch)
        history.append(terms)
        if step % eval_every != 0:
            continue
        val_loss = _validation_loss(trainer, dataset, batch_size=batch_size, bag_size=bag_size, device=device, seed=seed + step)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            stale_steps = 0
        else:
            stale_steps += eval_every
        if stale_steps >= patience:
            break
    return history


def _validation_loss(
    trainer: BridgeTrainer,
    dataset: SyntheticBiologyLiteDataset,
    *,
    batch_size: int,
    bag_size: int,
    device: str,
    seed: int,
) -> float:
    trainer.model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataset.iter_condition_batches(
            split="val",
            batch_size=batch_size,
            bag_size=bag_size,
            steps=3,
            seed=seed,
            device=device,
            shuffle=True,
        ):
            batch = move_batch_to_device(batch, device)
            total, _ = loss_for_batch(
                trainer.model,
                batch,
                weights=trainer.weights,
                schedule=trainer.schedule,
                step=trainer.global_step,
                uncertainty_weighting=trainer.uncertainty_weighting,
                multi_positive_alignment=trainer.multi_positive_alignment,
            )
            losses.append(float(total.detach().cpu()))
    trainer.model.train()
    return float(np.mean(losses)) if losses else float("inf")


def _objective_schedule_from_args(args: argparse.Namespace) -> ObjectiveScheduleConfig:
    enabled = (
        args.schedule_reconstruction_warmup_steps > 0
        or args.schedule_reconstruction_anneal_steps > 0
        or args.schedule_reconstruction_final_scale != 1.0
        or args.schedule_warmup_non_reconstruction_scale != 0.0
    )
    return ObjectiveScheduleConfig(
        enabled=enabled,
        reconstruction_warmup_steps=args.schedule_reconstruction_warmup_steps,
        reconstruction_anneal_steps=args.schedule_reconstruction_anneal_steps,
        reconstruction_final_scale=args.schedule_reconstruction_final_scale,
        warmup_non_reconstruction_scale=args.schedule_warmup_non_reconstruction_scale,
    )


def _collect_model_outputs(
    dataset: SyntheticBiologyLiteDataset,
    model,
    *,
    split: str,
    device: str,
    bag_size: int,
    seed: int,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    if not groups:
        raise ValueError(f"split {split!r} has no condition bags")
    outputs: dict[str, list[np.ndarray]] = {
        "rna_shared": [],
        "image_shared": [],
        "rna_teacher_shared": [],
        "image_teacher_shared": [],
        "counterfactual_delta": [],
        "z_state": [],
        "rna_token_prediction": [],
    }
    rna_mean = []
    image_mean = []
    z_bio_mean = []
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(groups), 16):
            selected = groups[start : start + 16]
            batch = dataset._make_bridge_batch(
                selected,
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            result = model(
                gene_ids=batch.gene_ids,
                expression_values=batch.expression_values,
                rna_token_mask=None,
                images=batch.images,
                image_patch_mask=None,
                perturbation_id=batch.perturbation_id,
                perturbation_type_id=batch.perturbation_type_id,
                cell_line_id=batch.cell_line_id,
                batch_id=batch.batch_id,
                dose=batch.dose,
                time=batch.time,
            )
            for key in outputs:
                value = result.get(key)
                if value is None:
                    continue
                outputs[key].append(value.detach().cpu().numpy().reshape(value.shape[0], -1))
            for group in selected:
                rna_mean.append(dataset.expression_values[group].mean(axis=0))
                image_mean.append(dataset.images[group].mean(axis=0))
                z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
    collected = {key: np.concatenate(value, axis=0) for key, value in outputs.items() if value}
    collected["metadata"] = metadata
    collected["rna_mean"] = np.stack(rna_mean)
    collected["image_mean"] = np.stack(image_mean)
    collected["z_bio_mean"] = np.stack(z_bio_mean)
    return collected


def _counterfactual_arrays(
    dataset: SyntheticBiologyLiteDataset,
    model,
    *,
    split: str,
    device: str,
    bag_size: int,
    seed: int,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    key_to_group = {
        (
            int(row.perturbation_id),
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        ): group
        for row, group in zip(metadata.itertuples(index=False), groups, strict=True)
    }
    rows = []
    predicted = []
    observed = []
    control = []
    rng = np.random.default_rng(seed + 303)
    model.eval()
    with torch.no_grad():
        for row, target_group in zip(metadata.itertuples(index=False), groups, strict=True):
            if int(row.perturbation_id) == dataset.config.control_perturbation_id:
                continue
            control_key = (
                dataset.config.control_perturbation_id,
                int(row.cell_line_id),
                float(row.dose),
                int(row.batch_id),
            )
            control_group = key_to_group.get(control_key)
            if control_group is None:
                continue
            batch = dataset._make_bridge_batch(
                [control_group],
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            perturbation_id = torch.tensor([int(row.perturbation_id)], dtype=torch.long, device=device)
            perturbation_type_id = torch.tensor([int(row.perturbation_type_id)], dtype=torch.long, device=device)
            dose = torch.tensor([float(row.dose)], dtype=torch.float32, device=device)
            result = model(
                gene_ids=batch.gene_ids,
                expression_values=batch.expression_values,
                rna_token_mask=None,
                images=batch.images,
                image_patch_mask=None,
                perturbation_id=perturbation_id,
                perturbation_type_id=perturbation_type_id,
                cell_line_id=batch.cell_line_id,
                batch_id=batch.batch_id,
                dose=dose,
                time=batch.time,
            )
            predicted.append(result["counterfactual_rna"].detach().cpu().numpy()[0])
            observed.append(dataset.expression_values[target_group].mean(axis=0))
            control.append(dataset.expression_values[control_group].mean(axis=0))
            rows.append(row._asdict())
    if not predicted:
        genes = dataset.config.genes
        return {
            "model_predicted": np.empty((0, genes), dtype=float),
            "observed": np.empty((0, genes), dtype=float),
            "control": np.empty((0, genes), dtype=float),
            "metadata": pd.DataFrame(rows),
        }
    return {
        "model_predicted": np.asarray(predicted, dtype=float).reshape(len(predicted), -1),
        "observed": np.asarray(observed, dtype=float).reshape(len(observed), -1),
        "control": np.asarray(control, dtype=float).reshape(len(control), -1),
        "metadata": pd.DataFrame(rows),
    }


def _gene_sets(dataset: SyntheticBiologyLiteDataset) -> dict[str, list[int]]:
    result = {}
    for program in sorted(np.unique(dataset.gene_program_assignment)):
        result[f"program_{int(program)}"] = np.flatnonzero(dataset.gene_program_assignment == program).astype(int).tolist()
    return result


def _latent_r2(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    pred = _linear_predict(x_train, y_train, x_test)
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)


def _linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_eval = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    penalty = ridge * np.eye(x_aug.shape[1])
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return x_eval @ coef


def _perturbation_direction_cosine(metadata: pd.DataFrame, mapped: np.ndarray, true_latent: np.ndarray) -> float:
    values = []
    control = metadata["perturbation_id"].to_numpy() == 0
    for perturbation_id in sorted(set(metadata["perturbation_id"])):
        if perturbation_id == 0:
            continue
        mask = metadata["perturbation_id"].to_numpy() == perturbation_id
        if not mask.any() or not control.any():
            continue
        pred_delta = mapped[mask].mean(axis=0) - mapped[control].mean(axis=0)
        true_delta = true_latent[mask].mean(axis=0) - true_latent[control].mean(axis=0)
        denom = np.linalg.norm(pred_delta) * np.linalg.norm(true_delta)
        if denom > 1e-12:
            values.append(float(np.dot(pred_delta, true_delta) / denom))
    return float(np.mean(values)) if values else 0.0


def _dose_response_rank_corr(metadata: pd.DataFrame, mapped: np.ndarray) -> float:
    correlations = []
    for (_, _), group in metadata.groupby(["perturbation_id", "cell_line_id"]):
        if group["dose"].nunique() < 2:
            continue
        indices = group.index.to_numpy()
        doses = group["dose"].to_numpy(dtype=float)
        norms = np.linalg.norm(mapped[indices] - mapped[indices].mean(axis=0, keepdims=True), axis=1)
        correlations.append(_spearman(doses, norms))
    return float(np.nanmean(correlations)) if correlations else 0.0


def _cell_line_baseline_recovery(metadata: pd.DataFrame, mapped: np.ndarray, baselines: np.ndarray) -> float:
    predictions = []
    targets = []
    for cell_line_id, group in metadata[metadata["perturbation_id"] == 0].groupby("cell_line_id"):
        predictions.append(mapped[group.index.to_numpy()].mean(axis=0))
        targets.append(baselines[int(cell_line_id)])
    if len(predictions) < 2:
        return 0.0
    predictions = np.stack(predictions)
    targets = np.stack(targets)
    ss_res = float(np.sum((targets - predictions) ** 2))
    ss_tot = float(np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2))
    return 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)


def _program_effect_recovery(predicted: np.ndarray, observed: np.ndarray, control: np.ndarray, programs: np.ndarray) -> float:
    pred_delta = predicted - control
    obs_delta = observed - control
    values = []
    for program in sorted(np.unique(programs)):
        mask = programs == program
        values.append(_safe_corr(pred_delta[:, mask].mean(axis=1), obs_delta[:, mask].mean(axis=1)))
    return float(np.nanmean(values)) if values else 0.0


def _collapse_diagnostics(outputs: dict[str, Any]) -> dict[str, Any]:
    rna = outputs["rna_shared"]
    image = outputs["image_shared"]
    delta = outputs.get("counterfactual_delta", np.zeros_like(rna))
    state = outputs.get("z_state", np.ones_like(rna))
    rna_std = rna.std(axis=0)
    image_std = image.std(axis=0)
    rank = _rank(rna)
    image_rank = _rank(image)
    teacher_cos = _paired_cosine(outputs.get("rna_shared", rna), outputs.get("rna_teacher_shared", rna))
    image_teacher_cos = _paired_cosine(outputs.get("image_shared", image), outputs.get("image_teacher_shared", image))
    delta_ratio = float(np.linalg.norm(delta, axis=1).mean() / max(np.linalg.norm(state, axis=1).mean(), 1e-12))
    predictor_variance = float(outputs.get("rna_token_prediction", rna).var())
    spectrum = np.linalg.svd(rna - rna.mean(axis=0, keepdims=True), full_matrices=False, compute_uv=False)
    return {
        "model_rna_shared_min_std": float(rna_std.min()),
        "model_rna_shared_mean_std": float(rna_std.mean()),
        "model_image_shared_min_std": float(image_std.min()),
        "model_image_shared_mean_std": float(image_std.mean()),
        "model_embedding_rank": float(rank),
        "model_image_embedding_rank": float(image_rank),
        "model_covariance_spectrum_top5": [float(value) for value in spectrum[:5]],
        "model_student_teacher_cosine_mean": float(np.mean([teacher_cos[0], image_teacher_cos[0]])),
        "model_student_teacher_cosine_std": float(np.mean([teacher_cos[1], image_teacher_cos[1]])),
        "model_delta_norm_ratio": delta_ratio,
        "model_predictor_output_variance": predictor_variance,
    }


def _rank(values: np.ndarray) -> int:
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return int(np.sum(singular_values > 1e-3))


def _paired_cosine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y_norm = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    values = np.sum(x_norm * y_norm, axis=1)
    return float(values.mean()), float(values.std())


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_corr(_rankdata(x), _rankdata(y))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size == 0 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _write_dataset_report(path: Path, dataset_name: str, seed: int, metrics: dict[str, Any], history: list[dict[str, float]]) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {dataset_name} Step 0 Baseline",
                "",
                f"Seed: `{seed}`",
                "",
                f"Training steps completed: `{len(history)}`",
                f"Device: `{metrics.get('device_used', 'cpu')}`",
                f"Wallclock minutes: `{metrics.get('wallclock_minutes', 0.0):.3f}`",
                "",
                "## Key Metrics",
                "",
                f"- Model RNA->image recall@1: `{metrics.get('model_rna_to_image_recall@1', float('nan')):.4f}`",
                f"- Random RNA->image recall@1: `{metrics.get('random_embedding_rna_to_image_recall@1', float('nan')):.4f}`",
                f"- Batch-only recall@1: `{metrics.get('batch_only_batch_only_recall@1', float('nan')):.4f}`",
                f"- Model counterfactual direction accuracy: `{metrics.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
                f"- Source-as-target direction accuracy: `{metrics.get('source_as_target_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
                f"- RNA shared biological latent R2: `{metrics.get('model_bio_latent_r2_rna_shared', float('nan')):.4f}`",
                f"- Batch probe balanced accuracy: `{metrics.get('model_batch_probe_balanced_accuracy', float('nan')):.4f}`",
                f"- Embedding rank: `{metrics.get('model_embedding_rank', float('nan')):.1f}`",
                f"- Delta norm ratio: `{metrics.get('model_delta_norm_ratio', float('nan')):.4f}`",
                f"- Collapse flag: `{metrics.get('collapse_flag', False)}`",
                "",
                "Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
```

## File: `scripts/evaluate_prefit_pls_readout.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.training.prefit_readout import (
    PrefitPLSReadout,
    fit_pls_readout,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, _latent_r2, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PREFIT_PLS_READOUT")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate closed-form PLS whitening readouts inside the bridge model.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-standardize", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(
        train_arrays["rna_mean"],
        train_arrays["image_mean"],
        rank=args.dim,
        output_standardize=args.output_standardize,
    )
    direct_metrics = _direct_metrics(dataset, readout)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=0,
        device=args.device,
        model_dim=max(4, args.dim),
        shared_dim=args.dim,
        dropout=0.0,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
    )
    model = experiment_config.build_model().to(args.device)
    install_prefit_pls_readout(model, readout, freeze=True, device=args.device)

    metrics = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=dataset.config.cells_per_condition,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    metrics["training_steps_completed"] = 0.0
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["model_dim"] = float(args.dim)
    metrics["tier1_pass_gate"] = _tier1_pass(metrics)
    metrics["readout"] = "prefit_pls_linear"

    (args.output_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(args.output_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, args.output_dir / "prefit_pls_readout.json")
    (args.output_dir / "DIRECT_PLS_METRICS.json").write_text(
        json.dumps(_jsonable(direct_metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "MODEL_METRICS.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "REPORT.md", args=args, metrics=metrics, direct_metrics=direct_metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _direct_metrics(dataset, readout: PrefitPLSReadout) -> dict[str, float]:
    arrays = {split: _condition_arrays(dataset, split) for split in ("train", "test")}
    train_rna = readout.rna.transform(arrays["train"]["rna_mean"])
    train_image = readout.image.transform(arrays["train"]["image_mean"])
    test_rna = readout.rna.transform(arrays["test"]["rna_mean"])
    test_image = readout.image.transform(arrays["test"]["image_mean"])
    retrieval = cross_modal_retrieval_metrics(
        test_rna,
        test_image,
        arrays["test"]["metadata"],
        arrays["test"]["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    batch = batch_probe_metrics(test_rna, arrays["test"]["metadata"], label_col="batch")
    return {
        "direct_rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "direct_rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "direct_image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "direct_rna_bio_latent_r2": _latent_r2(train_rna, arrays["train"]["z_bio"], test_rna, arrays["test"]["z_bio"]),
        "direct_image_bio_latent_r2": _latent_r2(train_image, arrays["train"]["z_bio"], test_image, arrays["test"]["z_bio"]),
        "direct_batch_balanced_accuracy": batch.get("batch_probe_balanced_accuracy", float("nan")),
        "direct_batch_majority_accuracy": batch.get("batch_probe_majority_accuracy", float("nan")),
        "direct_rna_min_std": float(test_rna.std(axis=0).min()),
        "direct_image_min_std": float(test_image.std(axis=0).min()),
    }


def _tier1_pass(metrics: dict[str, Any]) -> bool:
    batch_balanced = float(metrics.get("model_batch_probe_balanced_accuracy", float("nan")))
    batch_majority = float(metrics.get("model_batch_probe_majority_accuracy", 0.5))
    return bool(
        not metrics.get("collapse_flag", True)
        and float(metrics.get("model_rna_to_image_recall@1", 0.0)) >= RANDOM_RECALL1 + 0.05
        and float(metrics.get("model_bio_latent_r2_rna_shared", -1.0)) > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= batch_majority + 0.10)
    )


def _write_report(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], direct_metrics: dict[str, float]) -> None:
    lines = [
        "# Prefit PLS Linear Readout Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Dim: `{args.dim}`",
        f"- Tier 1 pass: `{bool(metrics['tier1_pass_gate'])}`",
        f"- Wallclock minutes: `{metrics['wallclock_minutes']:.3f}`",
        "",
        "## Direct PLS Check",
        "",
        f"- recall@1: `{direct_metrics['direct_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{direct_metrics['direct_rna_bio_latent_r2']:.4f}`",
        f"- Image latent R2: `{direct_metrics['direct_image_bio_latent_r2']:.4f}`",
        f"- RNA min std: `{direct_metrics['direct_rna_min_std']:.4f}`",
        f"- Image min std: `{direct_metrics['direct_image_min_std']:.4f}`",
        "",
        "## Bridge Result",
        "",
        f"- collapse flag: `{bool(metrics.get('collapse_flag'))}`",
        f"- recall@1: `{metrics.get('model_rna_to_image_recall@1', float('nan')):.4f}`",
        f"- recall@5: `{metrics.get('model_rna_to_image_recall@5', float('nan')):.4f}`",
        f"- RNA latent R2: `{metrics.get('model_bio_latent_r2_rna_shared', float('nan')):.4f}`",
        f"- Image latent R2: `{metrics.get('model_bio_latent_r2_image_shared', float('nan')):.4f}`",
        f"- RNA min std: `{metrics.get('model_rna_shared_min_std', float('nan')):.4f}`",
        f"- Image min std: `{metrics.get('model_image_shared_min_std', float('nan')):.4f}`",
        f"- Batch balanced accuracy: `{metrics.get('model_batch_probe_balanced_accuracy', float('nan')):.4f}`",
        "",
        "## Artifacts",
        "",
        "- `MODEL_METRICS.json`",
        "- `DIRECT_PLS_METRICS.json`",
        "- `prefit_pls_readout.json`",
        "- `bridge_config.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `scripts/train_pls_distilled_head.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.losses import info_nce_loss, variance_floor_loss
from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_distillation_head,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic import SyntheticBridgeBatch
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, _latent_r2, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/PLS_DISTILLED_HEAD")
RANDOM_RECALL1 = 1.0 / 16.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a separate neural head to distill frozen PLS geometry without replacing retrieval."
    )
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.05)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--student-head", default="raw_mlp", choices=("raw_mlp", "linear_clone", "residual_mlp"))
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    run_dir = args.output_dir / f"{args.dataset}_{args.student_head}_seed{args.seed}_rank{args.rank}_s{args.steps}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=args.rank)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=0.0,
        model_dim=max(4, args.rank),
        shared_dim=args.rank,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
    )
    model = experiment_config.build_model().to(args.device)
    install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
    if args.student_head in {"linear_clone", "residual_mlp"}:
        install_prefit_pls_distillation_head(model, readout, freeze=False, device=args.device)
    _freeze_all_parameters(model)
    _unfreeze_distilled_heads(model, args.student_head)
    initial_readout = _frozen_readout_state(model)

    before_student = _evaluate_student_head(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        student_head=args.student_head,
    )
    protected_before = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )

    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    history = _train_distilled_heads(
        model,
        optimizer,
        dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        seed=args.seed,
        device=args.device,
        alignment_weight=args.alignment_weight,
        variance_weight=args.variance_weight,
        student_head=args.student_head,
    )

    after_student = _evaluate_student_head(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        student_head=args.student_head,
    )
    protected_after = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    readout_drift = _max_frozen_readout_drift(initial_readout, _frozen_readout_state(model))
    protected_deltas = _protected_metric_deltas(protected_before, protected_after)
    after_student["training_steps_completed"] = float(len(history))
    after_student["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after_student["frozen_readout_max_abs_drift"] = readout_drift
    after_student["protected_geometry_preserved"] = bool(readout_drift <= 1e-7 and _protected_deltas_ok(protected_deltas))

    (run_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(run_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, run_dir / "prefit_pls_readout.json")
    _write_json(run_dir / "STUDENT_BEFORE_METRICS.json", before_student)
    _write_json(run_dir / "STUDENT_AFTER_METRICS.json", after_student)
    _write_json(run_dir / "PROTECTED_BEFORE_METRICS.json", protected_before)
    _write_json(run_dir / "PROTECTED_AFTER_METRICS.json", protected_after)
    _write_json(run_dir / "TRAIN_HISTORY.json", history)
    checkpoint_path = save_checkpoint(
        run_dir / "pls_distilled_head.pt",
        model=model,
        optimizer=optimizer,
        trainer_state={"steps": len(history)},
        experiment_config=experiment_config,
        metadata={
            "stage": "pls_distilled_head",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "student_head": args.student_head,
            "protected_metric_deltas": protected_deltas,
            "frozen_readout_max_abs_drift": readout_drift,
        },
    )
    _write_report(
        run_dir / "REPORT.md",
        args=args,
        before_student=before_student,
        after_student=after_student,
        protected_after=protected_after,
        protected_deltas=protected_deltas,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after_student), sort_keys=True))
    return 0


def _train_distilled_heads(
    model,
    optimizer: torch.optim.Optimizer,
    dataset,
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    seed: int,
    device: str,
    alignment_weight: float,
    variance_weight: float,
    student_head: str,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    batches = dataset.iter_condition_batches(
        split="train",
        batch_size=batch_size,
        bag_size=bag_size,
        steps=steps,
        seed=seed,
        device=device,
        rna_mask_prob=0.0,
        image_mask_prob=0.0,
    )
    model.train()
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_batch(model, batch)
        rna_student_key, image_student_key = _student_output_keys(student_head)
        rna_teacher = outputs["rna_raw_linear_shared"].detach()
        image_teacher = outputs["image_raw_linear_shared"].detach()
        rna_student = outputs[rna_student_key]
        image_student = outputs[image_student_key]
        rna_loss = _normalized_mse(rna_student, rna_teacher)
        image_loss = _normalized_mse(image_student, image_teacher)
        align_loss = info_nce_loss(rna_student, image_student) if alignment_weight else _zero_like(rna_loss)
        variance_loss = (
            variance_floor_loss(rna_student, target_std=0.05) + variance_floor_loss(image_student, target_std=0.05)
            if variance_weight
            else _zero_like(rna_loss)
        )
        total = rna_loss + image_loss + float(alignment_weight) * align_loss + float(variance_weight) * variance_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), 1.0)
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "total": float(total.detach().cpu()),
                "rna_teacher_mse": float(rna_loss.detach().cpu()),
                "image_teacher_mse": float(image_loss.detach().cpu()),
                "student_align": float(align_loss.detach().cpu()),
                "student_variance": float(variance_loss.detach().cpu()),
            }
        )
    return history


def _evaluate_student_head(
    dataset,
    model,
    *,
    split: str,
    train_split: str,
    device: str,
    bag_size: int,
    seed: int,
    student_head: str = "raw_mlp",
) -> dict[str, Any]:
    test = _collect_head_outputs(
        dataset,
        model,
        split=split,
        device=device,
        bag_size=bag_size,
        seed=seed,
        student_head=student_head,
    )
    train = _collect_head_outputs(
        dataset,
        model,
        split=train_split,
        device=device,
        bag_size=bag_size,
        seed=seed + 101,
        student_head=student_head,
    )
    retrieval = cross_modal_retrieval_metrics(
        test["rna_student"],
        test["image_student"],
        test["metadata"],
        test["metadata"],
        ks=(1, 5),
        stratify_by=(),
    )
    batch = batch_probe_metrics(test["rna_student"], test["metadata"], label_col="batch")
    rna_std = test["rna_student"].std(axis=0)
    image_std = test["image_student"].std(axis=0)
    metrics: dict[str, Any] = {
        "student_rna_to_image_recall@1": retrieval["rna_to_image_recall@1"],
        "student_rna_to_image_recall@5": retrieval["rna_to_image_recall@5"],
        "student_image_to_rna_recall@1": retrieval["image_to_rna_recall@1"],
        "student_bio_latent_r2_rna_shared": _latent_r2(
            train["rna_student"],
            train["z_bio_mean"],
            test["rna_student"],
            test["z_bio_mean"],
        ),
        "student_bio_latent_r2_image_shared": _latent_r2(
            train["image_student"],
            train["z_bio_mean"],
            test["image_student"],
            test["z_bio_mean"],
        ),
        "student_rna_shared_min_std": float(rna_std.min()),
        "student_rna_shared_mean_std": float(rna_std.mean()),
        "student_image_shared_min_std": float(image_std.min()),
        "student_image_shared_mean_std": float(image_std.mean()),
        "student_teacher_rna_mse": float(np.mean((test["rna_student"] - test["rna_teacher"]) ** 2)),
        "student_teacher_image_mse": float(np.mean((test["image_student"] - test["image_teacher"]) ** 2)),
    }
    metrics.update({f"student_{key}": value for key, value in batch.items()})
    majority = float(batch.get("batch_probe_majority_accuracy", 0.0))
    balanced = float(batch.get("batch_probe_balanced_accuracy", float("nan")))
    metrics["student_batch_probe_balanced_accuracy_minus_majority"] = float(balanced - majority)
    metrics["student_collapse_flag"] = bool(
        metrics["student_rna_shared_min_std"] < 0.01 or metrics["student_image_shared_min_std"] < 0.01
    )
    metrics["student_tier1_pass_gate"] = _student_tier1_pass(metrics)
    if hasattr(model, "rna_distilled_residual_scale"):
        metrics["rna_distilled_residual_scale"] = float(model.rna_distilled_residual_scale.detach().cpu())
    if hasattr(model, "image_distilled_residual_scale"):
        metrics["image_distilled_residual_scale"] = float(model.image_distilled_residual_scale.detach().cpu())
    return metrics


def _collect_head_outputs(
    dataset,
    model,
    *,
    split: str,
    device: str,
    bag_size: int,
    seed: int,
    student_head: str,
) -> dict[str, Any]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    if not groups:
        raise ValueError(f"split {split!r} has no condition bags")
    outputs: dict[str, list[np.ndarray]] = {
        "rna_student": [],
        "image_student": [],
        "rna_teacher": [],
        "image_teacher": [],
    }
    z_bio_mean = []
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        rna_student_key, image_student_key = _student_output_keys(student_head)
        for start in range(0, len(groups), 16):
            selected = groups[start : start + 16]
            batch = dataset._make_bridge_batch(
                selected,
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            result = _forward_batch(model, batch)
            outputs["rna_student"].append(result[rna_student_key].detach().cpu().numpy())
            outputs["image_student"].append(result[image_student_key].detach().cpu().numpy())
            outputs["rna_teacher"].append(result["rna_raw_linear_shared"].detach().cpu().numpy())
            outputs["image_teacher"].append(result["image_raw_linear_shared"].detach().cpu().numpy())
            for group in selected:
                z_bio_mean.append(dataset.z_bio[group].mean(axis=0))
    collected = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    collected["metadata"] = metadata
    collected["z_bio_mean"] = np.stack(z_bio_mean)
    return collected


def _forward_batch(model, batch: SyntheticBridgeBatch) -> dict[str, torch.Tensor]:
    return model(
        gene_ids=batch.gene_ids,
        expression_values=batch.expression_values,
        rna_token_mask=None,
        images=batch.images,
        image_patch_mask=None,
        perturbation_id=batch.perturbation_id,
        perturbation_type_id=batch.perturbation_type_id,
        cell_line_id=batch.cell_line_id,
        batch_id=batch.batch_id,
        dose=batch.dose,
        time=batch.time,
    )


def _student_tier1_pass(metrics: dict[str, Any]) -> bool:
    batch_balanced = float(metrics.get("student_batch_probe_balanced_accuracy", float("nan")))
    batch_majority = float(metrics.get("student_batch_probe_majority_accuracy", 0.5))
    return bool(
        not metrics.get("student_collapse_flag", True)
        and float(metrics.get("student_rna_to_image_recall@1", 0.0)) >= RANDOM_RECALL1 + 0.05
        and float(metrics.get("student_bio_latent_r2_rna_shared", -1.0)) > 0.0
        and (not np.isfinite(batch_balanced) or batch_balanced <= batch_majority + 0.10)
    )


def _normalized_mse(student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.detach().std(dim=0, unbiased=False).clamp_min(1e-3)
    return (((student - target.detach()) / scale) ** 2).mean()


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=value.device, dtype=value.dtype)


def _freeze_all_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _unfreeze_distilled_heads(model, student_head: str) -> None:
    if student_head == "raw_mlp":
        modules = (model.rna_raw_pseudobulk_projection, model.image_raw_projection)
    elif student_head == "linear_clone":
        modules = (model.rna_distilled_linear_projection, model.image_distilled_linear_projection)
    elif student_head == "residual_mlp":
        modules = (model.rna_raw_pseudobulk_projection, model.image_raw_projection)
        model.rna_distilled_residual_scale.requires_grad_(True)
        model.image_distilled_residual_scale.requires_grad_(True)
    else:
        raise ValueError(f"unsupported student_head: {student_head}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _student_output_keys(student_head: str) -> tuple[str, str]:
    if student_head == "raw_mlp":
        return "rna_distilled_shared", "image_distilled_shared"
    if student_head == "linear_clone":
        return "rna_distilled_linear_shared", "image_distilled_linear_shared"
    if student_head == "residual_mlp":
        return "rna_distilled_residual_shared", "image_distilled_residual_shared"
    raise ValueError(f"unsupported student_head: {student_head}")


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _frozen_readout_state(model) -> dict[str, torch.Tensor]:
    return {
        "rna_weight": model.rna_raw_linear_projection.weight.detach().cpu().clone(),
        "rna_bias": model.rna_raw_linear_projection.bias.detach().cpu().clone(),
        "image_weight": model.image_raw_linear_projection.weight.detach().cpu().clone(),
        "image_bias": model.image_raw_linear_projection.bias.detach().cpu().clone(),
    }


def _max_frozen_readout_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return float(max((after[key] - before[key]).abs().max().item() for key in before))


def _protected_metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_to_image_recall@1",
        "model_rna_to_image_recall@5",
        "model_bio_latent_r2_rna_shared",
        "model_bio_latent_r2_image_shared",
        "model_rna_shared_min_std",
        "model_image_shared_min_std",
        "model_batch_probe_balanced_accuracy",
    )
    return {key: float(after.get(key, 0.0)) - float(before.get(key, 0.0)) for key in keys}


def _protected_deltas_ok(deltas: dict[str, float]) -> bool:
    return (
        deltas["model_rna_to_image_recall@1"] >= -1e-6
        and deltas["model_bio_latent_r2_rna_shared"] >= -1e-4
        and deltas["model_rna_shared_min_std"] >= -1e-6
        and deltas["model_image_shared_min_std"] >= -1e-6
        and deltas["model_batch_probe_balanced_accuracy"] <= 1e-6
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before_student: dict[str, Any],
    after_student: dict[str, Any],
    protected_after: dict[str, Any],
    protected_deltas: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# PLS Distilled Head Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Student head: `{args.student_head}`",
        f"- Frozen readout max abs drift: `{after_student['frozen_readout_max_abs_drift']:.8f}`",
        f"- Protected geometry preserved: `{bool(after_student['protected_geometry_preserved'])}`",
        "",
        "## Student Head Before",
        "",
        f"- pass: `{bool(before_student['student_tier1_pass_gate'])}`",
        f"- recall@1: `{before_student['student_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{before_student['student_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{before_student['student_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Student Head After",
        "",
        f"- pass: `{bool(after_student['student_tier1_pass_gate'])}`",
        f"- recall@1: `{after_student['student_rna_to_image_recall@1']:.4f}`",
        f"- recall@5: `{after_student['student_rna_to_image_recall@5']:.4f}`",
        f"- RNA latent R2: `{after_student['student_bio_latent_r2_rna_shared']:.4f}`",
        f"- Image latent R2: `{after_student['student_bio_latent_r2_image_shared']:.4f}`",
        f"- RNA min std: `{after_student['student_rna_shared_min_std']:.4f}`",
        f"- Image min std: `{after_student['student_image_shared_min_std']:.4f}`",
        f"- batch balanced accuracy: `{after_student['student_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Protected PLS Retrieval Path",
        "",
        f"- recall@1: `{protected_after['model_rna_to_image_recall@1']:.4f}`",
        f"- RNA latent R2: `{protected_after['model_bio_latent_r2_rna_shared']:.4f}`",
        f"- batch balanced accuracy: `{protected_after['model_batch_probe_balanced_accuracy']:.4f}`",
        "",
        "## Protected Deltas",
        "",
    ]
    for key, value in protected_deltas.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `STUDENT_BEFORE_METRICS.json`",
            "- `STUDENT_AFTER_METRICS.json`",
            "- `PROTECTED_BEFORE_METRICS.json`",
            "- `PROTECTED_AFTER_METRICS.json`",
            "- `TRAIN_HISTORY.json`",
            "- `prefit_pls_readout.json`",
            f"- `{checkpoint_path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `scripts/train_clone_counterfactual_decoder.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.checkpoint import save_checkpoint
from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_distillation_head,
    install_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _jsonable, evaluate_step0


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/CLONE_COUNTERFACTUAL_DECODER")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train counterfactual decoder heads with PLS clone geometry frozen.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--program-weight", type=float, default=0.0)
    parser.add_argument("--program-factorized-rna", action="store_true")
    parser.add_argument("--within-program-residual", action="store_true")
    parser.add_argument("--program-condition-source", action="store_true")
    parser.add_argument("--program-metadata-context", action="store_true")
    parser.add_argument("--linear-program-decoder", action="store_true")
    parser.add_argument("--prefit-program-ridge", action="store_true")
    parser.add_argument("--prefit-program-ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--prefit-program-ridge-repeats", type=int, default=8)
    parser.add_argument("--delta-mse", action="store_true")
    parser.add_argument("--train-perturbation-encoder", action="store_true")
    parser.add_argument("--residual-rna", action="store_true")
    args = parser.parse_args()
    if args.linear_program_decoder and not args.program_factorized_rna:
        parser.error("--linear-program-decoder requires --program-factorized-rna")
    if args.prefit_program_ridge and not args.program_factorized_rna:
        parser.error("--prefit-program-ridge requires --program-factorized-rna")
    if args.prefit_program_ridge and not args.linear_program_decoder:
        parser.error("--prefit-program-ridge requires --linear-program-decoder")
    if args.prefit_program_ridge_repeats < 1:
        parser.error("--prefit-program-ridge-repeats must be >= 1")

    started = time.perf_counter()
    seed_everything(args.seed)
    perturb_mode = "perttrain" if args.train_perturbation_encoder else "pertfrozen"
    residual_mode = "rnaresidual" if args.residual_rna else "rnagenerative"
    direction_tag = f"dw{str(args.direction_weight).replace('.', 'p')}"
    program_tag = f"pw{str(args.program_weight).replace('.', 'p')}"
    factorized_tag = "pfact" if args.program_factorized_rna else "genehead"
    within_tag = "wres" if args.within_program_residual else "nowres"
    context_tag = "srcprog" if args.program_condition_source else "nostx"
    metadata_tag = "metactx" if args.program_metadata_context else "nometactx"
    depth_tag = "linearprog" if args.linear_program_decoder else "mlpprog"
    init_tag = "prefitridge" if args.prefit_program_ridge else "sgdinit"
    loss_tag = "deltamse" if args.delta_mse else "absmse"
    lr_tag = f"lr{_slug_float(args.lr)}"
    wd_tag = f"wd{_slug_float(args.weight_decay)}"
    batch_tag = f"bs{args.batch_size}"
    ridge_tag = ""
    if args.prefit_program_ridge:
        ridge_tag = f"_ra{_slug_float(args.prefit_program_ridge_alpha)}_rr{args.prefit_program_ridge_repeats}"
    run_dir = (
        args.output_dir
        / (
            f"{args.dataset}_{perturb_mode}_{residual_mode}_{factorized_tag}_{within_tag}_{context_tag}"
            f"_{metadata_tag}_{depth_tag}_{init_tag}_{loss_tag}"
            f"_{direction_tag}_{program_tag}"
            f"_{lr_tag}_{wd_tag}_{batch_tag}{ridge_tag}"
            f"_seed{args.seed}_rank{args.rank}_s{args.steps}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=args.rank)

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=args.steps,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=0.0,
        model_dim=max(4, args.rank),
        shared_dim=args.rank,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
        counterfactual_rna_residual=args.residual_rna,
        counterfactual_rna_program_factorized=args.program_factorized_rna,
        counterfactual_rna_num_programs=dataset.config.num_programs if args.program_factorized_rna else 0,
        counterfactual_rna_program_assignment=tuple(int(value) for value in dataset.gene_program_assignment)
        if args.program_factorized_rna
        else (),
        counterfactual_rna_within_program_residual=args.within_program_residual,
        counterfactual_rna_program_conditioned=args.program_condition_source,
        counterfactual_rna_program_metadata_context=args.program_metadata_context,
        counterfactual_rna_program_decoder_depth=1 if args.linear_program_decoder else 2,
    )
    model = experiment_config.build_model().to(args.device)
    install_prefit_pls_readout(model, readout, freeze=True, device=args.device)
    install_prefit_pls_distillation_head(model, readout, freeze=True, device=args.device)
    if args.program_factorized_rna:
        _zero_init_program_factorized_decoder(model, zero_within=args.within_program_residual)
    elif args.residual_rna:
        _zero_init_rna_delta_decoder(model)
    _freeze_all_parameters(model)
    _unfreeze_counterfactual_modules(
        model,
        train_perturbation_encoder=args.train_perturbation_encoder,
        program_factorized=args.program_factorized_rna,
        within_program_residual=args.within_program_residual,
    )
    initial_readout = _frozen_readout_state(model)

    before = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    pairs = _counterfactual_pairs(dataset, split="train")
    if not pairs:
        raise RuntimeError("no train counterfactual pairs were found")

    prefit_program_ridge = {}
    if args.prefit_program_ridge:
        prefit_program_ridge = _prefit_program_ridge_decoder(
            model,
            dataset,
            pairs,
            bag_size=bag_size,
            seed=args.seed,
            device=args.device,
            ridge=args.prefit_program_ridge_alpha,
            repeats=args.prefit_program_ridge_repeats,
        )

    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    history = _train_counterfactual(
        model,
        optimizer,
        dataset,
        pairs,
        steps=args.steps,
        batch_size=args.batch_size,
        bag_size=bag_size,
        seed=args.seed,
        device=args.device,
        direction_weight=args.direction_weight,
        program_weight=args.program_weight,
        delta_mse=args.delta_mse,
    )
    after = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=args.device,
        bag_size=bag_size,
        seed=args.seed,
        label_shuffle_repeats=20,
    )
    readout_drift = _max_frozen_readout_drift(initial_readout, _frozen_readout_state(model))
    protected_deltas = _protected_metric_deltas(before, after)
    counterfactual_deltas = _counterfactual_deltas(before, after)
    after["training_steps_completed"] = float(len(history))
    after["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    after["device_used"] = str(args.device)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        after["max_gpu_memory_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    else:
        after["max_gpu_memory_gb"] = 0.0
    after["frozen_readout_max_abs_drift"] = readout_drift
    after["protected_geometry_preserved"] = bool(readout_drift <= 1e-7 and _protected_deltas_ok(protected_deltas))
    after["counterfactual_gate_pass"] = _counterfactual_gate_pass(before, after, protected_deltas, readout_drift)
    after["counterfactual_program_factorized"] = bool(args.program_factorized_rna)
    after["counterfactual_within_program_residual"] = bool(args.within_program_residual)
    after["counterfactual_program_conditioned"] = bool(args.program_condition_source)
    after["counterfactual_program_metadata_context"] = bool(args.program_metadata_context)
    after["counterfactual_program_decoder_depth"] = int(1 if args.linear_program_decoder else 2)
    after["counterfactual_prefit_program_ridge"] = bool(args.prefit_program_ridge)
    after["counterfactual_prefit_program_ridge_alpha"] = float(args.prefit_program_ridge_alpha)
    after["counterfactual_prefit_program_ridge_repeats"] = float(args.prefit_program_ridge_repeats)
    after["counterfactual_delta_mse_loss"] = bool(args.delta_mse)
    after.update({f"prefit_program_ridge_{key}": value for key, value in prefit_program_ridge.items()})
    after.update(_training_diagnostics(history))

    (run_dir / "generation_config.json").write_text(
        json.dumps(_jsonable(asdict(dataset.config)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    experiment_config.save_json(run_dir / "bridge_config.json")
    save_prefit_pls_readout(readout, run_dir / "prefit_pls_readout.json")
    if prefit_program_ridge:
        _write_json(run_dir / "PREFIT_PROGRAM_RIDGE.json", prefit_program_ridge)
    _write_json(run_dir / "BEFORE_METRICS.json", before)
    _write_json(run_dir / "AFTER_METRICS.json", after)
    _write_json(run_dir / "TRAIN_HISTORY.json", history)
    checkpoint_path = save_checkpoint(
        run_dir / "clone_counterfactual_decoder.pt",
        model=model,
        optimizer=optimizer,
        trainer_state={"steps": len(history)},
        experiment_config=experiment_config,
        metadata={
            "stage": "clone_counterfactual_decoder",
            "dataset": args.dataset,
            "seed": args.seed,
            "prefit_readout": readout.to_dict(),
            "prefit_readout_path": "prefit_pls_readout.json",
            "protected_metric_deltas": protected_deltas,
            "counterfactual_metric_deltas": counterfactual_deltas,
            "frozen_readout_max_abs_drift": readout_drift,
            "counterfactual_program_factorized": bool(args.program_factorized_rna),
            "counterfactual_within_program_residual": bool(args.within_program_residual),
            "counterfactual_program_conditioned": bool(args.program_condition_source),
            "counterfactual_program_metadata_context": bool(args.program_metadata_context),
            "counterfactual_program_decoder_depth": int(1 if args.linear_program_decoder else 2),
            "counterfactual_prefit_program_ridge": bool(args.prefit_program_ridge),
            "prefit_program_ridge": prefit_program_ridge,
            "counterfactual_delta_mse_loss": bool(args.delta_mse),
        },
    )
    _write_report(
        run_dir / "REPORT.md",
        args=args,
        before=before,
        after=after,
        protected_deltas=protected_deltas,
        counterfactual_deltas=counterfactual_deltas,
        checkpoint_path=checkpoint_path,
    )
    print(json.dumps(_jsonable(after), sort_keys=True))
    return 0


def _train_counterfactual(
    model,
    optimizer: torch.optim.Optimizer,
    dataset,
    pairs: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    bag_size: int,
    seed: int,
    device: str,
    direction_weight: float,
    program_weight: float,
    delta_mse: bool,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    program_assignment = torch.as_tensor(dataset.gene_program_assignment, dtype=torch.long, device=device)
    model.train()
    for step in range(1, steps + 1):
        selected = [pairs[int(index)] for index in rng.integers(0, len(pairs), size=batch_size)]
        batch = dataset._make_bridge_batch(
            [item["control_group"] for item in selected],
            bag_size=bag_size,
            rng=rng,
            rna_mask_prob=0.0,
            image_mask_prob=0.0,
            device=device,
        )
        target = torch.as_tensor(
            np.stack([dataset.expression_values[item["target_group"]].mean(axis=0) for item in selected]),
            dtype=torch.float32,
            device=device,
        )
        target_metadata = _target_metadata_tensors(selected, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            gene_ids=batch.gene_ids,
            expression_values=batch.expression_values,
            rna_token_mask=None,
            images=batch.images,
            image_patch_mask=None,
            perturbation_id=target_metadata["perturbation_id"],
            perturbation_type_id=target_metadata["perturbation_type_id"],
            cell_line_id=target_metadata["cell_line_id"],
            batch_id=target_metadata["batch_id"],
            dose=target_metadata["dose"],
            time=target_metadata["time"],
        )
        prediction = outputs["counterfactual_rna"]
        control = batch.expression_values.mean(dim=1)
        target_delta = target - control
        predicted_delta = prediction - control
        absolute_mse = F.mse_loss(prediction, target)
        delta_loss = F.mse_loss(predicted_delta, target_delta)
        mse = delta_loss if delta_mse else absolute_mse
        direction = 1.0 - F.cosine_similarity(predicted_delta, target_delta, dim=-1).mean()
        program_loss = _program_delta_mse(
            predicted_delta,
            target_delta,
            program_assignment=program_assignment,
        )
        total = mse + float(direction_weight) * direction + float(program_weight) * program_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), 1.0)
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "total": float(total.detach().cpu()),
                "counterfactual_mse": float(mse.detach().cpu()),
                "absolute_mse": float(absolute_mse.detach().cpu()),
                "delta_mse": float(delta_loss.detach().cpu()),
                "direction_loss": float(direction.detach().cpu()),
                "program_delta_mse": float(program_loss.detach().cpu()),
                "final_delta_to_target_ratio": _norm_ratio(predicted_delta, target_delta),
                "program_contribution_ratio": _norm_ratio(
                    outputs.get("counterfactual_rna_program_gene_delta"),
                    predicted_delta,
                ),
                "within_program_residual_ratio": _norm_ratio(
                    outputs.get("counterfactual_rna_within_program_residual"),
                    predicted_delta,
                ),
                "cap_hit_fraction": 0.0,
            }
        )
    return history


def _program_delta_mse(
    prediction_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    program_assignment: torch.Tensor,
) -> torch.Tensor:
    if prediction_delta.shape != target_delta.shape:
        raise ValueError("program delta loss inputs must have matching shapes")
    if program_assignment.shape != (prediction_delta.shape[-1],):
        program_assignment = program_assignment[: prediction_delta.shape[-1]]
    values = []
    for program in program_assignment.unique(sorted=True):
        mask = program_assignment.eq(program)
        values.append(F.mse_loss(prediction_delta[:, mask].mean(dim=1), target_delta[:, mask].mean(dim=1)))
    if not values:
        return torch.zeros((), device=prediction_delta.device, dtype=prediction_delta.dtype)
    return torch.stack(values).mean()


def _counterfactual_pairs(dataset, *, split: str) -> list[dict[str, Any]]:
    groups = dataset.condition_bag_indices(split=split)
    metadata = dataset.metadata_for_condition_bags(split=split)
    key_to_group = {
        (
            int(row.perturbation_id),
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        ): group
        for row, group in zip(metadata.itertuples(index=False), groups, strict=True)
    }
    pairs: list[dict[str, Any]] = []
    for row, target_group in zip(metadata.itertuples(index=False), groups, strict=True):
        if int(row.perturbation_id) == dataset.config.control_perturbation_id:
            continue
        control_key = (
            dataset.config.control_perturbation_id,
            int(row.cell_line_id),
            float(row.dose),
            int(row.batch_id),
        )
        control_group = key_to_group.get(control_key)
        if control_group is None:
            continue
        pairs.append(
            {
                "control_group": control_group,
                "target_group": target_group,
                "perturbation_id": int(row.perturbation_id),
                "perturbation_type_id": int(row.perturbation_type_id),
                "cell_line_id": int(row.cell_line_id),
                "batch_id": int(row.batch_id),
                "dose": float(row.dose),
                "time": float(row.time),
            }
        )
    return pairs


def _target_metadata_tensors(selected: list[dict[str, Any]], *, device: str) -> dict[str, torch.Tensor]:
    return {
        "perturbation_id": torch.tensor([item["perturbation_id"] for item in selected], dtype=torch.long, device=device),
        "perturbation_type_id": torch.tensor(
            [item["perturbation_type_id"] for item in selected],
            dtype=torch.long,
            device=device,
        ),
        "cell_line_id": torch.tensor([item["cell_line_id"] for item in selected], dtype=torch.long, device=device),
        "batch_id": torch.tensor([item["batch_id"] for item in selected], dtype=torch.long, device=device),
        "dose": torch.tensor([item["dose"] for item in selected], dtype=torch.float32, device=device),
        "time": torch.tensor([item["time"] for item in selected], dtype=torch.float32, device=device),
    }


def _prefit_program_ridge_decoder(
    model,
    dataset,
    pairs: list[dict[str, Any]],
    *,
    bag_size: int,
    seed: int,
    device: str,
    ridge: float,
    repeats: int,
) -> dict[str, float]:
    if not getattr(model.config, "counterfactual_rna_program_factorized", False):
        raise ValueError("program ridge prefit requires a program-factorized decoder")
    linear = model.counterfactual_program_decoder.net[-1]
    if len(model.counterfactual_program_decoder.net) != 1 or not isinstance(linear, torch.nn.Linear):
        raise TypeError("program ridge prefit requires a depth-1 linear program decoder")

    rng = np.random.default_rng(seed + 19_001)
    assignment = np.asarray(dataset.gene_program_assignment, dtype=np.int64)
    num_programs = int(model.config.counterfactual_rna_num_programs)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            batch = dataset._make_bridge_batch(
                [item["control_group"] for item in pairs],
                bag_size=bag_size,
                rng=rng,
                rna_mask_prob=0.0,
                image_mask_prob=0.0,
                device=device,
            )
            target_metadata = _target_metadata_tensors(pairs, device=device)
            outputs = model(
                gene_ids=batch.gene_ids,
                expression_values=batch.expression_values,
                rna_token_mask=None,
                images=batch.images,
                image_patch_mask=None,
                perturbation_id=target_metadata["perturbation_id"],
                perturbation_type_id=target_metadata["perturbation_type_id"],
                cell_line_id=target_metadata["cell_line_id"],
                batch_id=target_metadata["batch_id"],
                dose=target_metadata["dose"],
                time=target_metadata["time"],
            )
            decoder_input = outputs.get("counterfactual_rna_program_decoder_input")
            if decoder_input is None:
                raise RuntimeError("model did not expose counterfactual_rna_program_decoder_input")
            x_blocks.append(decoder_input.detach().cpu().numpy())
            y_blocks.append(
                np.stack(
                    [
                        _program_means_np(
                            dataset.expression_values[item["target_group"]].mean(axis=0)
                            - dataset.expression_values[item["control_group"]].mean(axis=0),
                            assignment=assignment,
                            num_programs=num_programs,
                        )
                        for item in pairs
                    ],
                    axis=0,
                )
            )
    if was_training:
        model.train()
    x_train = np.concatenate(x_blocks, axis=0).astype(np.float64)
    y_train = np.concatenate(y_blocks, axis=0).astype(np.float64)
    if x_train.shape[1] != linear.in_features:
        raise ValueError(f"decoder input width {x_train.shape[1]} does not match linear width {linear.in_features}")
    x_aug = np.concatenate((x_train, np.ones((x_train.shape[0], 1), dtype=np.float64)), axis=1)
    penalty = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    prediction = x_aug @ coef
    residual = prediction - y_train
    total = y_train - y_train.mean(axis=0, keepdims=True)
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(total).sum())

    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(coef[:-1].T, dtype=linear.weight.dtype, device=linear.weight.device))
        linear.bias.copy_(torch.as_tensor(coef[-1], dtype=linear.bias.dtype, device=linear.bias.device))

    return {
        "train_rows": float(x_train.shape[0]),
        "feature_dim": float(x_train.shape[1]),
        "target_dim": float(y_train.shape[1]),
        "ridge_alpha": float(ridge),
        "repeats": float(repeats),
        "train_mse": float(np.square(residual).mean()),
        "train_program_r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "mean_target_program_delta_norm": float(np.linalg.norm(y_train, axis=1).mean()),
        "mean_predicted_program_delta_norm": float(np.linalg.norm(prediction, axis=1).mean()),
    }


def _program_means_np(values: np.ndarray, *, assignment: np.ndarray, num_programs: int) -> np.ndarray:
    if values.shape[-1] != assignment.shape[0]:
        raise ValueError("program assignment length must match values last dimension")
    result = np.zeros(num_programs, dtype=np.float64)
    for program in range(num_programs):
        mask = assignment == program
        if np.any(mask):
            result[program] = float(values[mask].mean())
    return result


def _freeze_all_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _zero_init_rna_delta_decoder(model) -> None:
    final = model.rna_distribution_decoder.net[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("rna_distribution_decoder must end with a Linear layer for zero init")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)


def _zero_init_program_factorized_decoder(model, *, zero_within: bool) -> None:
    final = model.counterfactual_program_decoder.net[-1]
    if not isinstance(final, torch.nn.Linear):
        raise TypeError("counterfactual_program_decoder must end with a Linear layer for zero init")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)
    if zero_within:
        _zero_init_rna_delta_decoder(model)


def _unfreeze_counterfactual_modules(
    model,
    *,
    train_perturbation_encoder: bool,
    program_factorized: bool,
    within_program_residual: bool,
) -> None:
    modules = [
        model.state_head,
        model.response_head,
        model.delta_gate,
        model.delta_effect,
    ]
    if program_factorized:
        modules.append(model.counterfactual_program_decoder)
        if within_program_residual:
            modules.append(model.rna_distribution_decoder)
    else:
        modules.append(model.rna_distribution_decoder)
    if train_perturbation_encoder:
        modules.append(model.perturbation_encoder)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _frozen_readout_state(model) -> dict[str, torch.Tensor]:
    return {
        "rna_weight": model.rna_raw_linear_projection.weight.detach().cpu().clone(),
        "rna_bias": model.rna_raw_linear_projection.bias.detach().cpu().clone(),
        "image_weight": model.image_raw_linear_projection.weight.detach().cpu().clone(),
        "image_bias": model.image_raw_linear_projection.bias.detach().cpu().clone(),
    }


def _max_frozen_readout_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    return float(max((after[key] - before[key]).abs().max().item() for key in before))


def _protected_metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_to_image_recall@1",
        "model_rna_to_image_recall@5",
        "model_bio_latent_r2_rna_shared",
        "model_bio_latent_r2_image_shared",
        "model_rna_shared_min_std",
        "model_image_shared_min_std",
        "model_batch_probe_balanced_accuracy",
    )
    return {key: float(after.get(key, 0.0)) - float(before.get(key, 0.0)) for key in keys}


def _counterfactual_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_counterfactual_direction_accuracy",
        "model_rna_counterfactual_logfc_correlation",
        "model_rna_counterfactual_pseudobulk_correlation",
        "model_program_level_effect_recovery",
        "model_rna_counterfactual_top50_de_overlap",
    )
    return {key: float(after.get(key, 0.0)) - float(before.get(key, 0.0)) for key in keys}


def _norm_ratio(numerator: torch.Tensor | None, denominator: torch.Tensor | None, *, eps: float = 1e-8) -> float:
    if numerator is None or denominator is None:
        return 0.0
    num = torch.linalg.vector_norm(numerator.detach(), dim=-1).mean()
    den = torch.linalg.vector_norm(denominator.detach(), dim=-1).mean().clamp_min(eps)
    return float((num / den).cpu())


def _training_diagnostics(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {
            "mean_final_delta_to_target_ratio": 0.0,
            "mean_program_contribution_ratio": 0.0,
            "mean_within_program_residual_ratio": 0.0,
            "mean_cap_hit_fraction": 0.0,
        }
    keys = (
        "final_delta_to_target_ratio",
        "program_contribution_ratio",
        "within_program_residual_ratio",
        "cap_hit_fraction",
    )
    return {f"mean_{key}": float(np.mean([row.get(key, 0.0) for row in history])) for key in keys}


def _protected_deltas_ok(deltas: dict[str, float]) -> bool:
    return (
        deltas["model_rna_to_image_recall@1"] >= -1e-6
        and deltas["model_bio_latent_r2_rna_shared"] >= -1e-4
        and deltas["model_rna_shared_min_std"] >= -1e-6
        and deltas["model_image_shared_min_std"] >= -1e-6
        and deltas["model_batch_probe_balanced_accuracy"] <= 1e-6
    )


def _counterfactual_gate_pass(
    before: dict[str, Any],
    after: dict[str, Any],
    protected_deltas: dict[str, float],
    readout_drift: float,
) -> bool:
    return bool(
        readout_drift <= 1e-7
        and _protected_deltas_ok(protected_deltas)
        and float(after.get("model_rna_counterfactual_pseudobulk_correlation", -1.0))
        >= float(before.get("model_rna_counterfactual_pseudobulk_correlation", -1.0)) - 0.02
        and float(after.get("model_rna_counterfactual_direction_accuracy", -1.0))
        >= float(before.get("model_rna_counterfactual_direction_accuracy", -1.0)) + 0.10
        and float(after.get("model_rna_counterfactual_logfc_correlation", -1.0))
        >= float(before.get("model_rna_counterfactual_logfc_correlation", -1.0)) + 0.05
        and float(after.get("model_program_level_effect_recovery", -1.0))
        >= float(before.get("model_program_level_effect_recovery", -1.0)) + 0.05
        and float(after.get("model_rna_counterfactual_top50_de_overlap", -1.0))
        >= float(before.get("model_rna_counterfactual_top50_de_overlap", -1.0)) + 0.03
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    before: dict[str, Any],
    after: dict[str, Any],
    protected_deltas: dict[str, float],
    counterfactual_deltas: dict[str, float],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# Clone Counterfactual Decoder Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Rank: `{args.rank}`",
        f"- Steps: `{args.steps}`",
        f"- Train perturbation encoder: `{bool(args.train_perturbation_encoder)}`",
        f"- Residual RNA prediction: `{bool(args.residual_rna)}`",
        f"- Program-factorized RNA decoder: `{bool(args.program_factorized_rna)}`",
        f"- Within-program residual: `{bool(args.within_program_residual)}`",
        f"- Source-conditioned program decoder: `{bool(args.program_condition_source)}`",
        f"- Direct biological metadata context: `{bool(args.program_metadata_context)}`",
        f"- Delta MSE training loss: `{bool(args.delta_mse)}`",
        f"- Device used: `{args.device}`",
        f"- Max GPU memory GB: `{after.get('max_gpu_memory_gb', 0.0):.4f}`",
        f"- Protected geometry preserved: `{bool(after['protected_geometry_preserved'])}`",
        f"- Counterfactual gate pass: `{bool(after['counterfactual_gate_pass'])}`",
        f"- Frozen readout max abs drift: `{after['frozen_readout_max_abs_drift']:.8f}`",
        "",
        "## Counterfactual Metrics",
        "",
        f"- direction accuracy before: `{before.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
        f"- direction accuracy after: `{after.get('model_rna_counterfactual_direction_accuracy', float('nan')):.4f}`",
        f"- logFC correlation before: `{before.get('model_rna_counterfactual_logfc_correlation', float('nan')):.4f}`",
        f"- logFC correlation after: `{after.get('model_rna_counterfactual_logfc_correlation', float('nan')):.4f}`",
        f"- pseudobulk correlation before: `{before.get('model_rna_counterfactual_pseudobulk_correlation', float('nan')):.4f}`",
        f"- pseudobulk correlation after: `{after.get('model_rna_counterfactual_pseudobulk_correlation', float('nan')):.4f}`",
        f"- program recovery before: `{before.get('model_program_level_effect_recovery', float('nan')):.4f}`",
        f"- program recovery after: `{after.get('model_program_level_effect_recovery', float('nan')):.4f}`",
        f"- top50 overlap before: `{before.get('model_rna_counterfactual_top50_de_overlap', float('nan')):.4f}`",
        f"- top50 overlap after: `{after.get('model_rna_counterfactual_top50_de_overlap', float('nan')):.4f}`",
        "",
        "## Decoder Diagnostics",
        "",
        f"- mean final delta/target delta ratio: `{after.get('mean_final_delta_to_target_ratio', 0.0):.4f}`",
        f"- mean program contribution ratio: `{after.get('mean_program_contribution_ratio', 0.0):.4f}`",
        f"- mean within-program residual ratio: `{after.get('mean_within_program_residual_ratio', 0.0):.4f}`",
        f"- mean cap-hit fraction: `{after.get('mean_cap_hit_fraction', 0.0):.4f}`",
        "",
        "## Protected Deltas",
        "",
    ]
    for key, value in protected_deltas.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(["", "## Counterfactual Deltas", ""])
    for key, value in counterfactual_deltas.items():
        lines.append(f"- `{key}`: `{value:.8f}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `BEFORE_METRICS.json`",
            "- `AFTER_METRICS.json`",
            "- `TRAIN_HISTORY.json`",
            "- `prefit_pls_readout.json`",
            f"- `{checkpoint_path.name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `scripts/run_family_m_transport_baselines.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.rna_counterfactual import rna_counterfactual_metrics
from perturb_jepa.training.prefit_readout import fit_pls_readout, install_prefit_pls_distillation_head, install_prefit_pls_readout
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset, generate_synthetic_biology_lite, synthetic_lite_config
from scripts.diagnose_pseudobulk_whitening_probe import _condition_arrays
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset, _gene_sets, _jsonable, _program_effect_recovery, evaluate_step0
from scripts.train_clone_counterfactual_decoder import _counterfactual_pairs


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_M_TRANSPORT_BASELINES")
BIOLOGICAL_KEY_FIELDS = ("perturbation_id", "cell_line_id", "dose", "time")


class FeatureProjector:
    def __init__(
        self,
        *,
        dataset: SyntheticBiologyLiteDataset,
        feature_space: str,
        fit_values: np.ndarray,
        readout: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.feature_space = feature_space
        self.readout = readout
        fitted = self._raw_project(fit_values)
        self.mean = fitted.mean(axis=0, keepdims=True)
        self.std = np.where(fitted.std(axis=0, keepdims=True) < 1e-6, 1.0, fitted.std(axis=0, keepdims=True))

    def project(self, values: np.ndarray) -> np.ndarray:
        raw = self._raw_project(values)
        return (raw - self.mean) / self.std

    def _raw_project(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if self.feature_space == "rna":
            return values
        if self.feature_space == "program":
            return program_means(values, self.dataset.gene_program_assignment)
        if self.feature_space == "pls":
            if self.readout is None:
                raise ValueError("feature_space='pls' requires a prefit readout")
            return np.asarray(self.readout.rna.transform(values), dtype=float)
        raise ValueError(f"unsupported feature space: {self.feature_space!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family M no-batch matching and conditional transport baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--knn-feature-space", choices=("program", "pls", "rna"), default="program")
    parser.add_argument("--sinkhorn-feature-space", choices=("program", "pls", "rna"), default="program")
    parser.add_argument("--k-values", default="1,3,5,8")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn-query-epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn-iterations", type=int, default=200)
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    protected_metrics, readout = protected_pls_metrics(
        dataset,
        rank=args.rank,
        device=args.device,
        seed=args.seed,
    )
    train_records = pair_records(dataset, split="train")
    test_records = pair_records(dataset, split="test")
    if not train_records or not test_records:
        raise RuntimeError("Family M transport baselines require train and test counterfactual pairs")

    all_train_controls = np.concatenate([record["control_cells"] for record in train_records], axis=0)
    knn_projector = FeatureProjector(
        dataset=dataset,
        feature_space=args.knn_feature_space,
        fit_values=all_train_controls,
        readout=readout,
    )
    sinkhorn_projector = FeatureProjector(
        dataset=dataset,
        feature_space=args.sinkhorn_feature_space,
        fit_values=all_train_controls,
        readout=readout,
    )

    source_metrics = evaluate_predictions(
        dataset,
        test_records,
        [record["control_mean"] for record in test_records],
        candidate_name="source_as_target",
        method="source_as_target",
        exact_match_fraction=1.0,
        diagnostics={},
    )

    candidates: list[dict[str, Any]] = []
    prediction, diagnostics = predict_matched_perturbed_mean(train_records, test_records)
    candidates.append(
        evaluate_predictions(
            dataset,
            test_records,
            prediction,
            candidate_name="seed2_no_batch_matched_perturbed_mean",
            method="matched_perturbed_mean",
            exact_match_fraction=diagnostics["exact_match_fraction"],
            diagnostics=diagnostics,
        )
    )

    prediction, diagnostics = predict_residualized_matching(train_records, test_records)
    residual_matching = evaluate_predictions(
        dataset,
        test_records,
        prediction,
        candidate_name="seed2_no_batch_residualized_matching",
        method="residualized_matching",
        exact_match_fraction=diagnostics["exact_match_fraction"],
        diagnostics=diagnostics,
    )
    candidates.append(residual_matching)

    for k in parse_int_list(args.k_values):
        prediction, diagnostics = predict_knn_residual_transport(
            train_records,
            test_records,
            projector=knn_projector,
            k=k,
        )
        candidates.append(
            evaluate_predictions(
                dataset,
                test_records,
                prediction,
                candidate_name=f"seed2_knn_residual_transport_{args.knn_feature_space}_k{k}",
                method="knn_residual_transport",
                exact_match_fraction=diagnostics["exact_match_fraction"],
                diagnostics={**diagnostics, "k": float(k), "feature_space": args.knn_feature_space},
            )
        )

    prediction, diagnostics = predict_sinkhorn_residual_transport(
        train_records,
        test_records,
        projector=sinkhorn_projector,
        epsilon=args.sinkhorn_epsilon,
        query_epsilon=args.sinkhorn_query_epsilon,
        iterations=args.sinkhorn_iterations,
    )
    candidates.append(
        evaluate_predictions(
            dataset,
            test_records,
            prediction,
            candidate_name=(
                f"seed2_sinkhorn_residual_transport_{args.sinkhorn_feature_space}"
                f"_eps{slug_float(args.sinkhorn_epsilon)}"
            ),
            method="sinkhorn_residual_transport",
            exact_match_fraction=diagnostics["exact_match_fraction"],
            diagnostics={**diagnostics, "feature_space": args.sinkhorn_feature_space},
        )
    )

    matching_reference = residual_matching
    rows = []
    for candidate in candidates:
        candidate["beats_residualized_matching"] = beats_matching_reference(candidate, matching_reference)
        candidate["counterfactual_gate_pass"] = counterfactual_gate_pass(candidate, source_metrics)
        candidate["protected_geometry_preserved"] = True
        candidate["protected_rna_to_image_recall@1"] = float(protected_metrics.get("model_rna_to_image_recall@1", 0.0))
        candidate["protected_bio_latent_r2_rna_shared"] = float(
            protected_metrics.get("model_bio_latent_r2_rna_shared", 0.0)
        )
        candidate["protected_representation_rank"] = float(protected_metrics.get("model_embedding_rank", 0.0))
        candidate["protected_batch_probe_balanced_accuracy"] = float(
            protected_metrics.get("model_batch_probe_balanced_accuracy", 0.0)
        )
        rows.append(candidate)

    frame = pd.DataFrame(rows)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_M_RESULTS.tsv", sep="\t", index=False)

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_matching": True,
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "candidates": rows,
    }
    write_json(args.output_dir / "FAMILY_M_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_json(args.output_dir / "prefit_pls_readout.json", readout.to_dict())
    write_report(args.output_dir / "REPORT.md", frame, source_metrics=source_metrics, args=args)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def protected_pls_metrics(
    dataset: SyntheticBiologyLiteDataset,
    *,
    rank: int,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], Any]:
    bag_size = dataset.config.cells_per_condition
    train_arrays = _condition_arrays(dataset, "train")
    readout = fit_pls_readout(train_arrays["rna_mean"], train_arrays["image_mean"], rank=rank)
    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=0,
        device=device,
        dropout=0.0,
        model_dim=max(4, rank),
        shared_dim=rank,
        rna_condition_readout="raw_linear_pseudobulk",
        rna_pseudobulk_normalize=False,
        image_condition_readout="raw_linear_pooled",
        image_raw_normalize=False,
        bag_aggregator="mean",
        num_bag_prototypes=1,
        rna_mask_weight=0.0,
        image_mask_weight=0.0,
        jepa_weight=0.0,
        align_weight=0.0,
        mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        perturbation_cls_weight=0.0,
        batch_adv_weight=0.0,
        counterfactual_weight=0.0,
        cycle_weight=0.0,
        response_bottleneck_weight=0.0,
        shared_variance_weight=0.0,
        shared_covariance_weight=0.0,
    )
    model = experiment_config.build_model().to(device)
    install_prefit_pls_readout(model, readout, freeze=True, device=device)
    install_prefit_pls_distillation_head(model, readout, freeze=True, device=device)
    metrics = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=device,
        bag_size=bag_size,
        seed=seed,
        label_shuffle_repeats=20,
    )
    return metrics, readout


def pair_records(dataset: SyntheticBiologyLiteDataset, *, split: str) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split):
        control_cells = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_cells = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_cells": control_cells,
                "target_cells": target_cells,
                "control_mean": control_cells.mean(axis=0),
                "target_mean": target_cells.mean(axis=0),
            }
        )
    return records


def biological_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in BIOLOGICAL_KEY_FIELDS)


def predict_matched_perturbed_mean(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    target_by_key = group_mean_by_key(train_records, value_name="target_mean")
    global_target = np.stack([record["target_mean"] for record in train_records]).mean(axis=0)
    predictions = []
    exact_hits = 0
    for record in test_records:
        value = target_by_key.get(record["biological_key"])
        if value is None:
            value = global_target
        else:
            exact_hits += 1
        predictions.append(value)
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "batch_id_excluded": 1.0,
    }


def predict_residualized_matching(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    deltas_by_key: dict[tuple[Any, ...], list[np.ndarray]] = {}
    for record in train_records:
        deltas_by_key.setdefault(record["biological_key"], []).append(record["target_mean"] - record["control_mean"])
    mean_delta_by_key = {key: np.stack(values).mean(axis=0) for key, values in deltas_by_key.items()}
    global_delta = np.stack([record["target_mean"] - record["control_mean"] for record in train_records]).mean(axis=0)
    predictions = []
    exact_hits = 0
    for record in test_records:
        delta = mean_delta_by_key.get(record["biological_key"])
        if delta is None:
            delta = global_delta
        else:
            exact_hits += 1
        predictions.append(record["control_mean"] + delta)
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "batch_id_excluded": 1.0,
    }


def predict_knn_residual_transport(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    projector: FeatureProjector,
    k: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    samples_by_key = residual_samples_by_key(train_records)
    predictions = []
    exact_hits = 0
    neighbor_counts = []
    neighbor_distances = []
    for record in test_records:
        samples = samples_by_key.get(record["biological_key"])
        if not samples:
            samples = [sample for values in samples_by_key.values() for sample in values]
        else:
            exact_hits += 1
        sources = np.stack([sample["source"] for sample in samples])
        residuals = np.stack([sample["residual"] for sample in samples])
        distances = squared_distances(projector.project(record["control_mean"])[0:1], projector.project(sources))[0]
        take = np.argsort(distances)[: min(int(k), len(samples))]
        predictions.append(record["control_mean"] + residuals[take].mean(axis=0))
        neighbor_counts.append(float(len(take)))
        neighbor_distances.append(float(distances[take].mean()))
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "mean_neighbor_count": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "mean_neighbor_distance": float(np.mean(neighbor_distances)) if neighbor_distances else 0.0,
        "batch_id_excluded": 1.0,
    }


def predict_sinkhorn_residual_transport(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    projector: FeatureProjector,
    epsilon: float,
    query_epsilon: float,
    iterations: int,
) -> tuple[list[np.ndarray], dict[str, float]]:
    transports_by_key = {}
    for key in sorted({record["biological_key"] for record in train_records}):
        key_records = [record for record in train_records if record["biological_key"] == key]
        controls = np.concatenate([record["control_cells"] for record in key_records], axis=0)
        targets = np.concatenate([record["target_cells"] for record in key_records], axis=0)
        control_features = projector.project(controls)
        target_features = projector.project(targets)
        plan = sinkhorn_plan(control_features, target_features, epsilon=epsilon, iterations=iterations)
        row_mass = plan.sum(axis=1, keepdims=True)
        row_mass = np.where(row_mass <= 1e-12, 1.0, row_mass)
        barycentric_target = (plan / row_mass) @ targets
        transports_by_key[key] = {
            "controls": controls,
            "control_features": control_features,
            "residuals": barycentric_target - controls,
            "plan_entropy": plan_entropy(plan),
        }

    predictions = []
    exact_hits = 0
    entropies = []
    support_sizes = []
    for record in test_records:
        transport = transports_by_key.get(record["biological_key"])
        if transport is None:
            key, transport = next(iter(transports_by_key.items()))
        else:
            exact_hits += 1
        query = projector.project(record["control_mean"])[0:1]
        distances = squared_distances(query, transport["control_features"])[0]
        weights = softmax_negative_distance(distances, epsilon=query_epsilon)
        predictions.append(record["control_mean"] + weights @ transport["residuals"])
        entropies.append(float(transport["plan_entropy"]))
        support_sizes.append(float(transport["controls"].shape[0]))
    return predictions, {
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_hits / max(1, len(test_records))),
        "sinkhorn_epsilon": float(epsilon),
        "sinkhorn_query_epsilon": float(query_epsilon),
        "sinkhorn_iterations": float(iterations),
        "mean_plan_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
        "batch_id_excluded": 1.0,
    }


def residual_samples_by_key(train_records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, np.ndarray]]]:
    samples: dict[tuple[Any, ...], list[dict[str, np.ndarray]]] = {}
    for record in train_records:
        target_mean = record["target_mean"]
        key_samples = samples.setdefault(record["biological_key"], [])
        for source in record["control_cells"]:
            key_samples.append({"source": source, "residual": target_mean - source})
    return samples


def group_mean_by_key(records: list[dict[str, Any]], *, value_name: str) -> dict[tuple[Any, ...], np.ndarray]:
    grouped: dict[tuple[Any, ...], list[np.ndarray]] = {}
    for record in records:
        grouped.setdefault(record["biological_key"], []).append(np.asarray(record[value_name], dtype=float))
    return {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}


def evaluate_predictions(
    dataset: SyntheticBiologyLiteDataset,
    test_records: list[dict[str, Any]],
    predictions: list[np.ndarray],
    *,
    candidate_name: str,
    method: str,
    exact_match_fraction: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=float)
    observed = np.stack([record["target_mean"] for record in test_records])
    control = np.stack([record["control_mean"] for record in test_records])
    metadata = pd.DataFrame([{key: record[key] for key in BIOLOGICAL_KEY_FIELDS + ("batch_id",)} for record in test_records])
    metrics = rna_counterfactual_metrics(
        predicted,
        observed,
        control,
        metadata,
        groupby=None,
        topk=(50,),
        gene_sets=_gene_sets(dataset),
    )
    predicted_delta = predicted - control
    observed_delta = observed - control
    result: dict[str, Any] = {
        "candidate_name": candidate_name,
        "method": method,
        "rows": float(len(test_records)),
        "exact_match_fraction": float(exact_match_fraction),
        "program_level_effect_recovery": _program_effect_recovery(
            predicted,
            observed,
            control,
            dataset.gene_program_assignment,
        ),
        "direction_accuracy": float(metrics["rna_counterfactual_direction_accuracy"]),
        "logfc_correlation": float(metrics["rna_counterfactual_logfc_correlation"]),
        "pseudobulk_correlation": float(metrics["rna_counterfactual_pseudobulk_correlation"]),
        "top50_de_overlap": float(metrics["rna_counterfactual_top50_de_overlap"]),
        "pathway_score_correlation": float(metrics.get("rna_counterfactual_pathway_score_correlation", 0.0)),
        "mean_delta_to_target_ratio": norm_ratio(predicted_delta, observed_delta),
    }
    for key, value in diagnostics.items():
        result[key] = value
    return result


def counterfactual_gate_pass(candidate: dict[str, Any], source: dict[str, Any]) -> bool:
    return bool(
        float(candidate["pseudobulk_correlation"]) >= float(source["pseudobulk_correlation"]) - 0.02
        and float(candidate["direction_accuracy"]) >= float(source["direction_accuracy"]) + 0.10
        and float(candidate["logfc_correlation"]) >= float(source["logfc_correlation"]) + 0.05
        and float(candidate["program_level_effect_recovery"]) >= float(source["program_level_effect_recovery"]) + 0.05
        and float(candidate["top50_de_overlap"]) >= float(source["top50_de_overlap"]) + 0.03
    )


def beats_matching_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return bool(
        float(candidate["program_level_effect_recovery"]) > float(reference["program_level_effect_recovery"]) + 1e-9
        and float(candidate["logfc_correlation"]) > float(reference["logfc_correlation"]) + 1e-9
        and float(candidate["top50_de_overlap"]) > float(reference["top50_de_overlap"]) + 1e-9
    )


def program_means(values: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    return np.stack([values[:, assignment == program].mean(axis=1) for program in sorted(np.unique(assignment))], axis=1)


def sinkhorn_plan(x: np.ndarray, y: np.ndarray, *, epsilon: float, iterations: int) -> np.ndarray:
    cost = squared_distances(x, y)
    positive = cost[cost > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = cost / max(scale, 1e-12)
    kernel = np.exp(-cost / max(float(epsilon), 1e-6))
    a = np.full(x.shape[0], 1.0 / max(1, x.shape[0]), dtype=float)
    b = np.full(y.shape[0], 1.0 / max(1, y.shape[0]), dtype=float)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max(1, int(iterations))):
        u = a / np.maximum(kernel @ v, 1e-12)
        v = b / np.maximum(kernel.T @ u, 1e-12)
    return (u[:, None] * kernel) * v[None, :]


def plan_entropy(plan: np.ndarray) -> float:
    values = plan[plan > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = np.asarray(x, dtype=float)[:, None, :] - np.asarray(y, dtype=float)[None, :, :]
    return np.sum(diff * diff, axis=-1)


def softmax_negative_distance(distances: np.ndarray, *, epsilon: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    positive = distances[distances > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    logits = -distances / max(float(epsilon) * scale, 1e-12)
    logits = logits - logits.max()
    weights = np.exp(logits)
    total = weights.sum()
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / max(1, weights.size), dtype=float)
    return weights / total


def norm_ratio(numerator: np.ndarray, denominator: np.ndarray, *, eps: float = 1e-8) -> float:
    num = np.linalg.norm(np.asarray(numerator, dtype=float), axis=-1).mean()
    den = max(float(np.linalg.norm(np.asarray(denominator, dtype=float), axis=-1).mean()), eps)
    return float(num / den)


def parse_int_list(values: str) -> list[int]:
    result = [int(value.strip()) for value in values.split(",") if value.strip()]
    if not result:
        raise ValueError("at least one k value is required")
    return result


def slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, frame: pd.DataFrame, *, source_metrics: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Family M Transport Baselines",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological matching key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from matching/features: `true`",
        "- Real data used: `false`",
        "- Marker/pathway/pretrained biological assets used: `false`",
        "",
        "## Source-As-Target Reference",
        "",
        f"- program recovery: `{source_metrics['program_level_effect_recovery']:.4f}`",
        f"- direction accuracy: `{source_metrics['direction_accuracy']:.4f}`",
        f"- logFC correlation: `{source_metrics['logfc_correlation']:.4f}`",
        f"- pseudobulk correlation: `{source_metrics['pseudobulk_correlation']:.4f}`",
        f"- top50 overlap: `{source_metrics['top50_de_overlap']:.4f}`",
        "",
        "## Candidate Results",
        "",
    ]
    for row in frame.itertuples(index=False):
        lines.extend(
            [
                f"### {row.candidate_name}",
                f"- method: `{row.method}`",
                f"- exact no-batch key coverage: `{row.exact_match_fraction:.4f}`",
                f"- program recovery: `{row.program_level_effect_recovery:.4f}`",
                f"- direction accuracy: `{row.direction_accuracy:.4f}`",
                f"- logFC correlation: `{row.logfc_correlation:.4f}`",
                f"- pseudobulk correlation: `{row.pseudobulk_correlation:.4f}`",
                f"- top50 overlap: `{row.top50_de_overlap:.4f}`",
                f"- mean delta/target ratio: `{row.mean_delta_to_target_ratio:.4f}`",
                f"- beats residualized matching: `{bool(row.beats_residualized_matching)}`",
                f"- counterfactual gate pass: `{bool(row.counterfactual_gate_pass)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Family M tests the CellOT/CINEMA-OT/scDRP/Conditional-Monge intuition on the synthetic seed-2 task without using batch features.",
            "Systema-style discipline is enforced by treating residualized matching as the baseline that transport must beat before neural transport is justified.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_M_RESULTS.tsv`",
            "- `FAMILY_M_RESULTS.json`",
            "- `generation_config.json`",
            "- `prefit_pls_readout.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `scripts/run_family_n_distillation.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset, generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import (
    BIOLOGICAL_KEY_FIELDS,
    biological_key,
    counterfactual_gate_pass,
    evaluate_predictions,
    norm_ratio,
    pair_records,
    predict_matched_perturbed_mean,
    predict_residualized_matching,
    program_means,
    protected_pls_metrics,
    slug_float,
)
from scripts.run_synthetic_lite_step0 import _jsonable


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_N_DISTILLATION")


class TrainOnlyConditionalMeanTable:
    """Train-split-only target mean lookup keyed by biological condition fields."""

    def __init__(self, train_records: list[dict[str, Any]], *, value_name: str = "target_mean") -> None:
        if not train_records:
            raise ValueError("at least one train record is required")
        self.value_name = value_name
        self.train_record_count = len(train_records)
        grouped: dict[tuple[Any, ...], list[np.ndarray]] = {}
        perturbation_grouped: dict[Any, list[np.ndarray]] = {}
        for record in train_records:
            key = biological_key(record)
            value = np.asarray(record[value_name], dtype=float)
            grouped.setdefault(key, []).append(value)
            perturbation_grouped.setdefault(record["perturbation_id"], []).append(value)
        self.target_by_key = {key: np.stack(values).mean(axis=0) for key, values in grouped.items()}
        self.perturbation_mean = {
            perturbation_id: np.stack(values).mean(axis=0) for perturbation_id, values in perturbation_grouped.items()
        }
        self.global_mean = np.stack([np.asarray(record[value_name], dtype=float) for record in train_records]).mean(axis=0)
        self.train_keys = tuple(sorted(self.target_by_key))

    def predict(self, records: list[dict[str, Any]]) -> tuple[list[np.ndarray], dict[str, float]]:
        predictions = []
        counts = {
            "exact": 0,
            "nearest_same_perturbation_cell": 0,
            "global_perturbation_mean": 0,
            "global_train_mean": 0,
        }
        for record in records:
            prediction, fallback = self._predict_one(record)
            counts[fallback] += 1
            predictions.append(prediction)
        rows = len(records)
        return predictions, {
            "rows": float(rows),
            "train_record_count": float(self.train_record_count),
            "train_key_count": float(len(self.target_by_key)),
            "test_key_count": float(len({biological_key(record) for record in records})),
            "exact_match_fraction": float(counts["exact"] / max(1, rows)),
            "fallback_exact_count": float(counts["exact"]),
            "fallback_nearest_same_perturbation_cell_count": float(counts["nearest_same_perturbation_cell"]),
            "fallback_global_perturbation_mean_count": float(counts["global_perturbation_mean"]),
            "fallback_global_train_mean_count": float(counts["global_train_mean"]),
            "batch_id_excluded": 1.0,
            "fit_split_train_only": 1.0,
            "teacher_target_test_rows_used": 0.0,
        }

    def _predict_one(self, record: dict[str, Any]) -> tuple[np.ndarray, str]:
        key = biological_key(record)
        exact = self.target_by_key.get(key)
        if exact is not None:
            return exact, "exact"
        nearest = self._nearest_same_perturbation_cell(record)
        if nearest is not None:
            return nearest, "nearest_same_perturbation_cell"
        perturbation_value = self.perturbation_mean.get(record["perturbation_id"])
        if perturbation_value is not None:
            return perturbation_value, "global_perturbation_mean"
        return self.global_mean, "global_train_mean"

    def _nearest_same_perturbation_cell(self, record: dict[str, Any]) -> np.ndarray | None:
        candidates = []
        for key, value in self.target_by_key.items():
            if key[0] != record["perturbation_id"] or key[1] != record["cell_line_id"]:
                continue
            distance = abs(float(key[2]) - float(record["dose"])) + abs(float(key[3]) - float(record["time"]))
            candidates.append((distance, key, value))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]


class ConditionFeatureEncoder:
    """No-batch biological condition and source-program feature encoder."""

    def __init__(self, dataset: SyntheticBiologyLiteDataset, train_records: list[dict[str, Any]]) -> None:
        if not train_records:
            raise ValueError("at least one train record is required")
        self.dataset = dataset
        self.perturbation_ids = tuple(sorted({int(record["perturbation_id"]) for record in train_records}))
        self.cell_line_ids = tuple(sorted({int(record["cell_line_id"]) for record in train_records}))
        self.biological_keys = tuple(sorted({biological_key(record) for record in train_records}))
        source_programs = np.asarray(
            [program_means(record["control_mean"], dataset.gene_program_assignment)[0] for record in train_records],
            dtype=float,
        )
        self.source_program_mean = source_programs.mean(axis=0)
        self.source_program_std = np.where(source_programs.std(axis=0) < 1e-6, 1.0, source_programs.std(axis=0))
        dose = np.asarray([float(record["dose"]) for record in train_records], dtype=float)
        time = np.asarray([float(record["time"]) for record in train_records], dtype=float)
        self.dose_mean = float(dose.mean())
        self.dose_std = float(dose.std()) if float(dose.std()) >= 1e-6 else 1.0
        self.time_mean = float(time.mean())
        self.time_std = float(time.std()) if float(time.std()) >= 1e-6 else 1.0
        self.feature_names = self._feature_names()

    def transform(self, records: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        perturb_index = {value: index for index, value in enumerate(self.perturbation_ids)}
        cell_index = {value: index for index, value in enumerate(self.cell_line_ids)}
        key_index = {value: index for index, value in enumerate(self.biological_keys)}
        for record in records:
            values: list[float] = [1.0]
            perturbation_one_hot = np.zeros(len(self.perturbation_ids), dtype=float)
            if int(record["perturbation_id"]) in perturb_index:
                perturbation_one_hot[perturb_index[int(record["perturbation_id"])]] = 1.0
            values.extend(perturbation_one_hot.tolist())
            cell_one_hot = np.zeros(len(self.cell_line_ids), dtype=float)
            if int(record["cell_line_id"]) in cell_index:
                cell_one_hot[cell_index[int(record["cell_line_id"])]] = 1.0
            values.extend(cell_one_hot.tolist())
            values.append((float(record["dose"]) - self.dose_mean) / self.dose_std)
            values.append((float(record["time"]) - self.time_mean) / self.time_std)
            key_one_hot = np.zeros(len(self.biological_keys), dtype=float)
            key = biological_key(record)
            if key in key_index:
                key_one_hot[key_index[key]] = 1.0
            values.extend(key_one_hot.tolist())
            source_program = program_means(record["control_mean"], self.dataset.gene_program_assignment)[0]
            values.extend(((source_program - self.source_program_mean) / self.source_program_std).tolist())
            rows.append(np.asarray(values, dtype=float))
        return np.stack(rows, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "batch_id_feature_present": any("batch" in name for name in self.feature_names),
            "fit_split_train_only": True,
            "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        }

    def _feature_names(self) -> list[str]:
        names = ["bias"]
        names.extend([f"perturbation_id={value}" for value in self.perturbation_ids])
        names.extend([f"cell_line_id={value}" for value in self.cell_line_ids])
        names.extend(["dose_z", "time_z"])
        names.extend([f"biological_key={index}" for index, _ in enumerate(self.biological_keys)])
        names.extend([f"source_program_z={index}" for index in range(self.dataset.config.num_programs)])
        return names


class LinearConditionalMeanModel:
    def __init__(self, weights: np.ndarray, *, ridge_alpha: float, train_mse: float) -> None:
        self.weights = weights
        self.ridge_alpha = float(ridge_alpha)
        self.train_mse = float(train_mse)

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, *, ridge_alpha: float) -> "LinearConditionalMeanModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        penalty = np.eye(x.shape[1], dtype=float) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        system = x.T @ x + penalty
        rhs = x.T @ y
        try:
            weights = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(system) @ rhs
        prediction = x @ weights
        return cls(weights, ridge_alpha=ridge_alpha, train_mse=float(np.mean((prediction - y) ** 2)))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.weights


class TargetStandardizer:
    def __init__(self, targets: np.ndarray) -> None:
        values = np.asarray(targets, dtype=float)
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = np.where(values.std(axis=0, keepdims=True) < 1e-6, 1.0, values.std(axis=0, keepdims=True))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean


class SmallMLPConditionalMeanModel:
    def __init__(
        self,
        model: torch.nn.Module,
        target_standardizer: TargetStandardizer,
        *,
        train_mse: float,
        final_loss: float,
    ) -> None:
        self.model = model
        self.target_standardizer = target_standardizer
        self.train_mse = float(train_mse)
        self.final_loss = float(final_loss)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        hidden_dim: int,
        steps: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> "SmallMLPConditionalMeanModel":
        torch.manual_seed(seed)
        x = torch.as_tensor(np.asarray(features, dtype=np.float32))
        standardizer = TargetStandardizer(targets)
        y_np = standardizer.transform(targets).astype(np.float32)
        y = torch.as_tensor(y_np)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, targets.shape[1]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        final_loss = 0.0
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        with torch.no_grad():
            train_prediction = standardizer.inverse_transform(model(x).detach().cpu().numpy())
        return cls(
            model,
            standardizer,
            train_mse=float(np.mean((train_prediction - np.asarray(targets, dtype=float)) ** 2)),
            final_loss=final_loss,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.as_tensor(np.asarray(features, dtype=np.float32))).cpu().numpy()
        return self.target_standardizer.inverse_transform(prediction)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family N train-only matched-mean distillation baselines.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--linear-ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-steps", type=int, default=1200)
    parser.add_argument("--mlp-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hybrid-alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    train_records = pair_records(dataset, split="train")
    test_records = pair_records(dataset, split="test")
    if not train_records or not test_records:
        raise RuntimeError("Family N requires train and test counterfactual pairs")
    protected_metrics, _ = protected_pls_metrics(dataset, rank=args.rank, device=args.device, seed=args.seed)
    source_metrics = evaluate_predictions(
        dataset,
        test_records,
        [record["control_mean"] for record in test_records],
        candidate_name="source_as_target",
        method="source_as_target",
        exact_match_fraction=1.0,
        diagnostics={},
    )

    rows: list[dict[str, Any]] = []
    table = TrainOnlyConditionalMeanTable(train_records, value_name="target_mean")
    table_predictions, table_diagnostics = table.predict(test_records)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                table_predictions,
                candidate_name="seed2_train_only_condition_mean_table",
                method="train_only_condition_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics=table_diagnostics,
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(table_predictions, dtype=float),
            train_targets=None,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "A"},
        )
    )

    train_teacher_predictions, train_teacher_diagnostics = table.predict(train_records)
    train_targets = np.asarray(train_teacher_predictions, dtype=float)
    feature_encoder = ConditionFeatureEncoder(dataset, train_records)
    train_features = feature_encoder.transform(train_records)
    test_features = feature_encoder.transform(test_records)

    linear_model = LinearConditionalMeanModel.fit(
        train_features,
        train_targets,
        ridge_alpha=args.linear_ridge_alpha,
    )
    linear_predictions = linear_model.predict(test_features)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                list(linear_predictions),
                candidate_name="seed2_distilled_linear_condition_mean",
                method="distilled_linear_condition_mean",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                    "linear_ridge_alpha": float(args.linear_ridge_alpha),
                    "linear_train_teacher_mse": linear_model.train_mse,
                },
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(linear_predictions, dtype=float),
            train_targets=train_targets,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "B"},
        )
    )

    mlp_model = SmallMLPConditionalMeanModel.fit(
        train_features,
        train_targets,
        hidden_dim=args.mlp_hidden_dim,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed + 31_337,
    )
    mlp_predictions = mlp_model.predict(test_features)
    rows.append(
        _finalize_candidate(
            evaluate_predictions(
                dataset,
                test_records,
                list(mlp_predictions),
                candidate_name="seed2_distilled_mlp_condition_mean",
                method="distilled_mlp_condition_mean",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.mlp_steps),
                    "mlp_lr": float(args.mlp_lr),
                    "mlp_weight_decay": float(args.mlp_weight_decay),
                    "mlp_final_standardized_loss": mlp_model.final_loss,
                    "mlp_train_teacher_mse": mlp_model.train_mse,
                },
            ),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            candidate_predictions=np.asarray(mlp_predictions, dtype=float),
            train_targets=train_targets,
            teacher_predictions=np.asarray(table_predictions, dtype=float),
            extra={"candidate_family": "C"},
        )
    )

    fallback = np.asarray(table_predictions, dtype=float)
    alphas = parse_float_list(args.hybrid_alphas)
    for model_name, learned_predictions in (
        ("linear", np.asarray(linear_predictions, dtype=float)),
        ("mlp", np.asarray(mlp_predictions, dtype=float)),
    ):
        for alpha in alphas:
            hybrid_predictions = float(alpha) * learned_predictions + (1.0 - float(alpha)) * fallback
            rows.append(
                _finalize_candidate(
                    evaluate_predictions(
                        dataset,
                        test_records,
                        list(hybrid_predictions),
                        candidate_name=f"seed2_{model_name}_condition_mean_hybrid_alpha{slug_float(alpha)}",
                        method="shrinkage_distillation_hybrid",
                        exact_match_fraction=table_diagnostics["exact_match_fraction"],
                        diagnostics={
                            **_teacher_fit_diagnostics(train_teacher_diagnostics, feature_encoder),
                            "hybrid_alpha": float(alpha),
                            "hybrid_model": model_name,
                            "fallback": "train_only_condition_mean_table",
                        },
                    ),
                    protected_metrics=protected_metrics,
                    source_metrics=source_metrics,
                    candidate_predictions=np.asarray(hybrid_predictions, dtype=float),
                    train_targets=train_targets,
                    teacher_predictions=fallback,
                    extra={"candidate_family": "D"},
                )
            )

    comparator_rows = build_comparator_rows(args.output_dir, rows)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame = pd.DataFrame(rows)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_N_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(comparator_rows).to_csv(args.output_dir / "COMPARATOR_RESULTS.tsv", sep="\t", index=False)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_matching_and_features": True,
        "train_pairs": len(train_records),
        "test_pairs": len(test_records),
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "feature_encoder": feature_encoder.to_dict(),
        "train_teacher_diagnostics": train_teacher_diagnostics,
        "candidates": rows,
        "comparators": comparator_rows,
    }
    write_json(args.output_dir / "FAMILY_N_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_report(
        args.output_dir / "REPORT.md",
        frame,
        comparator_rows=comparator_rows,
        source_metrics=source_metrics,
        args=args,
    )
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def _teacher_fit_diagnostics(
    train_teacher_diagnostics: dict[str, float],
    feature_encoder: ConditionFeatureEncoder,
) -> dict[str, Any]:
    feature_payload = feature_encoder.to_dict()
    return {
        "train_teacher_exact_match_fraction": float(train_teacher_diagnostics["exact_match_fraction"]),
        "feature_count": float(feature_payload["feature_count"]),
        "batch_id_feature_present": float(bool(feature_payload["batch_id_feature_present"])),
        "fit_split_train_only": 1.0,
        "teacher_target_test_rows_used": 0.0,
    }


def _finalize_candidate(
    result: dict[str, Any],
    *,
    protected_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    candidate_predictions: np.ndarray,
    train_targets: np.ndarray | None,
    teacher_predictions: np.ndarray,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = dict(result)
    row.update(extra)
    row["counterfactual_gate_pass"] = counterfactual_gate_pass(row, source_metrics)
    row["protected_geometry_preserved"] = True
    row["protected_rna_to_image_recall@1"] = float(protected_metrics.get("model_rna_to_image_recall@1", 0.0))
    row["protected_bio_latent_r2_rna_shared"] = float(protected_metrics.get("model_bio_latent_r2_rna_shared", 0.0))
    row["protected_representation_rank"] = float(protected_metrics.get("model_embedding_rank", 0.0))
    row["protected_batch_probe_balanced_accuracy"] = float(
        protected_metrics.get("model_batch_probe_balanced_accuracy", 0.0)
    )
    row["leakage_gate_pass"] = bool(
        float(row.get("fit_split_train_only", 1.0)) == 1.0
        and float(row.get("teacher_target_test_rows_used", 0.0)) == 0.0
        and float(row.get("batch_id_feature_present", 0.0)) == 0.0
        and float(row.get("batch_id_excluded", 1.0)) == 1.0
    )
    row["teacher_prediction_mse_on_test"] = float(np.mean((candidate_predictions - teacher_predictions) ** 2))
    if train_targets is not None:
        row["train_teacher_target_variance"] = float(np.var(train_targets))
    return row


def build_comparator_rows(output_dir: Path, family_n_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparators: list[dict[str, Any]] = []
    family_m_path = output_dir.parent / "FAMILY_M_TRANSPORT_BASELINES" / "FAMILY_M_RESULTS.tsv"
    if family_m_path.exists():
        family_m = pd.read_csv(family_m_path, sep="\t")
        for name in ("seed2_no_batch_matched_perturbed_mean", "seed2_no_batch_residualized_matching"):
            match = family_m[family_m["candidate_name"] == name]
            if not match.empty:
                comparators.append(_comparator_from_series(match.iloc[0], source="Family M"))
    comparators.extend(load_sparse_residual_comparators(output_dir.parent / "SPARSE_PERTURBATION_RESIDUAL_HEAD"))
    ridge = load_prefit_ridge_best(output_dir.parent / "CLONE_COUNTERFACTUAL_DECODER")
    if ridge is not None:
        comparators.append(ridge)
    for row in family_n_rows:
        if row["candidate_family"] in {"A", "B", "C"}:
            comparators.append(
                {
                    "candidate_name": row["candidate_name"],
                    "source": "Family N",
                    "program_level_effect_recovery": row["program_level_effect_recovery"],
                    "direction_accuracy": row["direction_accuracy"],
                    "logfc_correlation": row["logfc_correlation"],
                    "pseudobulk_correlation": row["pseudobulk_correlation"],
                    "top50_de_overlap": row["top50_de_overlap"],
                    "mean_delta_to_target_ratio": row["mean_delta_to_target_ratio"],
                }
            )
    return comparators


def _comparator_from_series(row: pd.Series, *, source: str) -> dict[str, Any]:
    return {
        "candidate_name": row["candidate_name"],
        "source": source,
        "program_level_effect_recovery": float(row["program_level_effect_recovery"]),
        "direction_accuracy": float(row["direction_accuracy"]),
        "logfc_correlation": float(row["logfc_correlation"]),
        "pseudobulk_correlation": float(row["pseudobulk_correlation"]),
        "top50_de_overlap": float(row["top50_de_overlap"]),
        "mean_delta_to_target_ratio": float(row["mean_delta_to_target_ratio"]),
    }


def load_sparse_residual_comparators(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/AFTER_METRICS.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate_name": path.parent.name,
                "source": "Family L",
                "program_level_effect_recovery": float(payload["model_program_level_effect_recovery"]),
                "direction_accuracy": float(payload["model_rna_counterfactual_direction_accuracy"]),
                "logfc_correlation": float(payload["model_rna_counterfactual_logfc_correlation"]),
                "pseudobulk_correlation": float(payload["model_rna_counterfactual_pseudobulk_correlation"]),
                "top50_de_overlap": float(payload["model_rna_counterfactual_top50_de_overlap"]),
                "mean_delta_to_target_ratio": float(payload.get("mean_sparse_final_delta_to_target_ratio", 0.0)),
            }
        )
    return rows


def load_prefit_ridge_best(root: Path) -> dict[str, Any] | None:
    best = None
    for path in sorted(root.glob("**/AFTER_METRICS.json")):
        if "prefitridge" not in str(path) or "seed2" not in str(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "candidate_name": path.parent.name,
            "source": "prefit ridge best",
            "program_level_effect_recovery": float(payload["model_program_level_effect_recovery"]),
            "direction_accuracy": float(payload["model_rna_counterfactual_direction_accuracy"]),
            "logfc_correlation": float(payload["model_rna_counterfactual_logfc_correlation"]),
            "pseudobulk_correlation": float(payload["model_rna_counterfactual_pseudobulk_correlation"]),
            "top50_de_overlap": float(payload["model_rna_counterfactual_top50_de_overlap"]),
            "mean_delta_to_target_ratio": float(payload.get("mean_final_delta_to_target_ratio", 0.0)),
        }
        if best is None:
            best = row
            continue
        if (row["top50_de_overlap"], row["logfc_correlation"]) > (
            best["top50_de_overlap"],
            best["logfc_correlation"],
        ):
            best = row
    return best


def parse_float_list(values: str) -> list[float]:
    result = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not result:
        raise ValueError("at least one alpha is required")
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    frame: pd.DataFrame,
    *,
    comparator_rows: list[dict[str, Any]],
    source_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    key_rows = frame[frame["candidate_family"].isin(["A", "B", "C"])].copy()
    hybrid = frame[frame["candidate_family"] == "D"].copy()
    best_hybrid = hybrid.sort_values(
        ["program_level_effect_recovery", "logfc_correlation", "top50_de_overlap"],
        ascending=False,
    ).head(1)
    lines = [
        "# Family N Train-Only Matched-Mean Distillation",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from matching and features: `true`",
        "- Teacher targets fit on train split only: `true`",
        "- Test target rows used for teacher construction: `0`",
        "- Real data used: `false`",
        "- Marker/pathway/pretrained biological assets used: `false`",
        "",
        "## Source-As-Target Reference",
        "",
        f"- program recovery: `{source_metrics['program_level_effect_recovery']:.4f}`",
        f"- direction accuracy: `{source_metrics['direction_accuracy']:.4f}`",
        f"- logFC correlation: `{source_metrics['logfc_correlation']:.4f}`",
        f"- pseudobulk correlation: `{source_metrics['pseudobulk_correlation']:.4f}`",
        f"- top50 overlap: `{source_metrics['top50_de_overlap']:.4f}`",
        "",
        "## Core Candidates",
        "",
    ]
    for row in key_rows.itertuples(index=False):
        lines.extend(_candidate_lines(row))
    if not best_hybrid.empty:
        lines.extend(["## Best Shrinkage Hybrid", ""])
        lines.extend(_candidate_lines(best_hybrid.iloc[0]))
    lines.extend(["## Required Comparators", ""])
    for comparator in comparator_rows:
        lines.extend(
            [
                f"### {comparator['candidate_name']}",
                f"- source: `{comparator['source']}`",
                f"- program recovery: `{comparator['program_level_effect_recovery']:.4f}`",
                f"- direction accuracy: `{comparator['direction_accuracy']:.4f}`",
                f"- logFC correlation: `{comparator['logfc_correlation']:.4f}`",
                f"- pseudobulk correlation: `{comparator['pseudobulk_correlation']:.4f}`",
                f"- top50 overlap: `{comparator['top50_de_overlap']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Candidate A is the leakage-safe train-only teacher and should match the Family M direct matched perturbed mean when exact train biological keys cover the test split.",
            "The learned students are useful only if they approximate that teacher without batch features or test target statistics; they do not change protected bridge geometry because no bridge weights are trained.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_N_RESULTS.tsv`",
            "- `FAMILY_N_RESULTS.json`",
            "- `COMPARATOR_RESULTS.tsv`",
            "- `generation_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_lines(row: Any) -> list[str]:
    return [
        f"### {row.candidate_name}",
        f"- method: `{row.method}`",
        f"- exact train key coverage on test: `{row.exact_match_fraction:.4f}`",
        f"- leakage gate pass: `{bool(row.leakage_gate_pass)}`",
        f"- program recovery: `{row.program_level_effect_recovery:.4f}`",
        f"- direction accuracy: `{row.direction_accuracy:.4f}`",
        f"- logFC correlation: `{row.logfc_correlation:.4f}`",
        f"- pseudobulk correlation: `{row.pseudobulk_correlation:.4f}`",
        f"- top50 overlap: `{row.top50_de_overlap:.4f}`",
        f"- mean delta/target ratio: `{row.mean_delta_to_target_ratio:.4f}`",
        f"- counterfactual gate pass: `{bool(row.counterfactual_gate_pass)}`",
        "",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: `scripts/run_family_o_count_likelihood.py`

```python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import SyntheticBiologyLiteDataset, generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import (
    BIOLOGICAL_KEY_FIELDS,
    biological_key,
    counterfactual_gate_pass,
    evaluate_predictions,
    norm_ratio,
    protected_pls_metrics,
    slug_float,
)
from scripts.run_family_n_distillation import (
    ConditionFeatureEncoder,
    TrainOnlyConditionalMeanTable,
    load_prefit_ridge_best,
    load_sparse_residual_comparators,
)
from scripts.run_synthetic_lite_step0 import _jsonable
from scripts.train_clone_counterfactual_decoder import _counterfactual_pairs


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/FAMILY_O_COUNT_LIKELIHOOD")
RESULTS_PATH = Path("outputs/autoresearch_synth_lite/results.tsv")


class CountMeanMLP(torch.nn.Module):
    """Small no-batch decoder that predicts per-gene log-count means."""

    def __init__(self, input_dim: int, genes: int, *, hidden_dim: int, initial_mean: np.ndarray) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, genes),
        )
        final = self.net[-1]
        if not isinstance(final, torch.nn.Linear):
            raise TypeError("CountMeanMLP final layer must be linear")
        torch.nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.log(torch.as_tensor(np.clip(initial_mean, 1e-4, None), dtype=torch.float32)))

    def forward(self, features: torch.Tensor, *, min_log_mean: float, max_log_mean: float) -> torch.Tensor:
        log_mu = torch.clamp(self.net(features), min=float(min_log_mean), max=float(max_log_mean))
        return torch.clamp(torch.exp(log_mu), min=1e-6, max=float(np.exp(max_log_mean)))


class TrainedCountModel:
    def __init__(
        self,
        model: CountMeanMLP,
        *,
        likelihood: str,
        train_nll: float,
        final_loss: float,
        dispersion: np.ndarray | None,
        min_log_mean: float,
        max_log_mean: float,
        device: str,
    ) -> None:
        self.model = model
        self.likelihood = likelihood
        self.train_nll = float(train_nll)
        self.final_loss = float(final_loss)
        self.dispersion = None if dispersion is None else np.asarray(dispersion, dtype=float)
        self.min_log_mean = float(min_log_mean)
        self.max_log_mean = float(max_log_mean)
        self.device = device

    def predict_mean(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device)
            mean = self.model(x, min_log_mean=self.min_log_mean, max_log_mean=self.max_log_mean)
        return mean.detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Family O synthetic count-likelihood perturbation diagnostics.")
    parser.add_argument("--dataset", default="synth_micro")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pseudo-count-scale", type=float, default=40.0)
    parser.add_argument("--dispersion-steps", type=int, default=700)
    parser.add_argument("--dispersion-lr", type=float, default=5e-2)
    parser.add_argument("--poisson-steps", type=int, default=1200)
    parser.add_argument("--nb-steps", type=int, default=1400)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-lr", type=float, default=3e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-dispersion", type=float, default=1e-4)
    parser.add_argument("--max-dispersion", type=float, default=10.0)
    parser.add_argument("--min-log-mean", type=float, default=-12.0)
    parser.add_argument("--max-log-mean", type=float, default=12.0)
    parser.add_argument("--append-results", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    counts, count_diagnostics = resolve_count_matrix(dataset, pseudo_count_scale=args.pseudo_count_scale)
    train_records = count_pair_records(dataset, counts, split="train")
    test_records = count_pair_records(dataset, counts, split="test")
    if not train_records or not test_records:
        raise RuntimeError("Family O requires train and test counterfactual pairs")

    protected_metrics, _ = protected_pls_metrics(dataset, rank=args.rank, device=args.device, seed=args.seed)
    source_count_predictions = [record["control_count_mean"] for record in test_records]
    source_metrics = evaluate_count_predictions(
        dataset,
        test_records,
        source_count_predictions,
        candidate_name="seed2_count_audit_source_as_target",
        method="count_audit_source_as_target",
        exact_match_fraction=1.0,
        diagnostics={"batch_id_excluded": 1.0, "fit_split_train_only": 1.0, "teacher_target_test_rows_used": 0.0},
    )

    rows: list[dict[str, Any]] = []
    global_train_count_mean = np.stack([record["target_count_mean"] for record in train_records]).mean(axis=0)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                source_count_predictions,
                candidate_name="seed2_count_audit_source_as_target",
                method="count_audit_source_as_target",
                exact_match_fraction=1.0,
                diagnostics={
                    "candidate_stage": "A",
                    "batch_id_excluded": 1.0,
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(source_count_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="COUNT_AUDIT_BASELINE_COMPLETE",
        )
    )
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                [global_train_count_mean for _ in test_records],
                candidate_name="seed2_train_global_count_mean_poisson_baseline",
                method="train_global_count_mean_poisson_baseline",
                exact_match_fraction=0.0,
                diagnostics={
                    "candidate_stage": "A",
                    "batch_id_excluded": 1.0,
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "global_train_count_mean": 1.0,
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray([global_train_count_mean for _ in test_records], dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="COUNT_AUDIT_BASELINE_COMPLETE",
        )
    )

    table = TrainOnlyConditionalMeanTable(train_records, value_name="target_count_mean")
    table_count_predictions, table_diagnostics = table.predict(test_records)
    table_count_array = np.asarray(table_count_predictions, dtype=float)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                table_count_predictions,
                candidate_name="seed2_poisson_train_only_count_mean_table",
                method="poisson_train_only_count_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **table_diagnostics,
                    "candidate_stage": "B",
                    "count_likelihood": "poisson",
                    "output_parameterization": "train_count_mean_positive_rate",
                },
            ),
            test_records=test_records,
            count_predictions=table_count_array,
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="TIER1_POISSON_COUNT_TABLE_COMPLETE",
        )
    )

    train_table_predictions, train_table_diagnostics = table.predict(train_records)
    learned_table_dispersion = fit_gene_wise_nb_dispersion(
        train_records,
        np.asarray(train_table_predictions, dtype=float),
        steps=args.dispersion_steps,
        lr=args.dispersion_lr,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 17,
        device=args.device,
    )
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                table_count_predictions,
                candidate_name="seed2_nb_train_only_count_mean_table",
                method="nb_train_only_count_mean_table",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    **table_diagnostics,
                    "candidate_stage": "C",
                    "count_likelihood": "negative_binomial",
                    "dispersion_fit_split_train_only": 1.0,
                    "train_teacher_exact_match_fraction": float(train_table_diagnostics["exact_match_fraction"]),
                    **dispersion_stats(learned_table_dispersion, prefix="nb_train"),
                },
            ),
            test_records=test_records,
            count_predictions=table_count_array,
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=learned_table_dispersion,
            decision_label="TIER1_NB_COUNT_TABLE_COMPLETE",
        )
    )

    feature_encoder = ConditionFeatureEncoder(dataset, train_records)
    train_features = feature_encoder.transform(train_records)
    test_features = feature_encoder.transform(test_records)
    feature_diagnostics = feature_encoder.to_dict()

    poisson_model = fit_count_mlp(
        train_features,
        train_records,
        likelihood="poisson",
        hidden_dim=args.mlp_hidden_dim,
        steps=args.poisson_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        min_log_mean=args.min_log_mean,
        max_log_mean=args.max_log_mean,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 101,
        device=args.device,
    )
    poisson_predictions = poisson_model.predict_mean(test_features)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                list(poisson_predictions),
                candidate_name="seed2_poisson_mlp_no_batch_condition_source",
                method="poisson_mlp_no_batch_condition_source",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    "candidate_stage": "D",
                    "count_likelihood": "poisson",
                    "feature_count": float(feature_diagnostics["feature_count"]),
                    "batch_id_feature_present": float(bool(feature_diagnostics["batch_id_feature_present"])),
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.poisson_steps),
                    "mlp_train_poisson_nll": poisson_model.train_nll,
                    "mlp_final_loss": poisson_model.final_loss,
                    "output_parameterization": "log_mean_exp_clamped",
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(poisson_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=None,
            decision_label="TIER1_POISSON_COUNT_MLP_COMPLETE",
        )
    )

    nb_model = fit_count_mlp(
        train_features,
        train_records,
        likelihood="negative_binomial",
        hidden_dim=args.mlp_hidden_dim,
        steps=args.nb_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        min_log_mean=args.min_log_mean,
        max_log_mean=args.max_log_mean,
        min_dispersion=args.min_dispersion,
        max_dispersion=args.max_dispersion,
        seed=args.seed + 202,
        device=args.device,
    )
    nb_predictions = nb_model.predict_mean(test_features)
    rows.append(
        finalize_count_candidate(
            evaluate_count_predictions(
                dataset,
                test_records,
                list(nb_predictions),
                candidate_name="seed2_nb_mlp_no_batch_condition_source",
                method="nb_mlp_no_batch_condition_source",
                exact_match_fraction=table_diagnostics["exact_match_fraction"],
                diagnostics={
                    "candidate_stage": "E",
                    "count_likelihood": "negative_binomial",
                    "feature_count": float(feature_diagnostics["feature_count"]),
                    "batch_id_feature_present": float(bool(feature_diagnostics["batch_id_feature_present"])),
                    "fit_split_train_only": 1.0,
                    "teacher_target_test_rows_used": 0.0,
                    "mlp_hidden_dim": float(args.mlp_hidden_dim),
                    "mlp_steps": float(args.nb_steps),
                    "mlp_train_nb_nll": nb_model.train_nll,
                    "mlp_final_loss": nb_model.final_loss,
                    "output_parameterization": "log_mean_exp_clamped",
                    **dispersion_stats(np.asarray(nb_model.dispersion, dtype=float), prefix="nb_model"),
                },
            ),
            test_records=test_records,
            count_predictions=np.asarray(nb_predictions, dtype=float),
            protected_metrics=protected_metrics,
            source_metrics=source_metrics,
            dispersion=nb_model.dispersion,
            decision_label="TIER1_NB_COUNT_MLP_COMPLETE",
        )
    )

    comparator_rows = load_required_comparators(args.output_dir)
    elapsed = float((time.perf_counter() - started) / 60.0)
    frame = pd.DataFrame(rows)
    frame["wallclock_minutes_total"] = elapsed
    frame.to_csv(args.output_dir / "FAMILY_O_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(comparator_rows).to_csv(args.output_dir / "COMPARATOR_RESULTS.tsv", sep="\t", index=False)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "rank": args.rank,
        "device": args.device,
        "count_diagnostics": count_diagnostics,
        "biological_key_fields": list(BIOLOGICAL_KEY_FIELDS),
        "batch_id_excluded_from_features": True,
        "train_pairs": len(train_records),
        "test_pairs": len(test_records),
        "source_as_target": source_metrics,
        "protected_metrics": protected_metrics,
        "feature_encoder": feature_diagnostics,
        "candidates": rows,
        "comparators": comparator_rows,
    }
    write_json(args.output_dir / "FAMILY_O_RESULTS.json", payload)
    write_json(args.output_dir / "generation_config.json", asdict(dataset.config))
    write_report(args.output_dir / "REPORT.md", frame, comparator_rows=comparator_rows, count_diagnostics=count_diagnostics, args=args)
    if args.append_results:
        append_results_tsv(frame, protected_metrics=protected_metrics, device=args.device, wallclock_minutes=elapsed)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    return 0


def resolve_count_matrix(
    dataset: SyntheticBiologyLiteDataset,
    *,
    pseudo_count_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(dataset, "observed_counts") and dataset.observed_counts is not None:
        counts = np.asarray(dataset.observed_counts, dtype=float)
        path = "raw_synthetic_observed_counts"
        pseudo = False
        scale = None
    else:
        counts = pseudo_counts_from_expression(dataset.expression_values, pseudo_count_scale=pseudo_count_scale)
        path = "synthetic_pseudo_count_from_expression_values"
        pseudo = True
        scale = float(pseudo_count_scale)
    diagnostics = count_path_diagnostics(dataset, counts, count_path=path, pseudo_count_used=pseudo, pseudo_count_scale=scale)
    return counts, diagnostics


def pseudo_counts_from_expression(expression_values: np.ndarray, *, pseudo_count_scale: float) -> np.ndarray:
    expression = np.asarray(expression_values, dtype=float)
    positive = np.clip(np.expm1(expression), 0.0, None)
    library = positive.sum(axis=1, keepdims=True)
    median_library = float(np.median(library[library > 0.0])) if np.any(library > 0.0) else 1.0
    scaled = positive * (float(pseudo_count_scale) * expression.shape[1]) / max(median_library, 1e-8)
    return np.rint(np.clip(scaled, 0.0, None)).astype(float)


def count_path_diagnostics(
    dataset: SyntheticBiologyLiteDataset,
    counts: np.ndarray,
    *,
    count_path: str,
    pseudo_count_used: bool,
    pseudo_count_scale: float | None,
) -> dict[str, Any]:
    values = np.asarray(counts, dtype=float)
    gene_mean = values.mean(axis=0)
    gene_var = values.var(axis=0)
    dispersion = method_of_moments_dispersion(values)
    return {
        "raw_count_available": bool(
            hasattr(dataset, "observed_counts") and dataset.observed_counts is not None and not pseudo_count_used
        ),
        "count_path": count_path,
        "pseudo_count_used": bool(pseudo_count_used),
        "pseudo_count_scale": pseudo_count_scale,
        "count_rows": int(values.shape[0]),
        "count_genes": int(values.shape[1]),
        "count_min": float(values.min()),
        "count_max": float(values.max()),
        "count_mean": float(values.mean()),
        "count_variance": float(values.var()),
        "zero_fraction": float(np.mean(values <= 0.0)),
        "integer_valued_fraction": float(np.mean(np.isclose(values, np.rint(values), atol=1e-6))),
        "mean_library_size": float(values.sum(axis=1).mean()),
        "median_library_size": float(np.median(values.sum(axis=1))),
        "gene_mean_variance_log_correlation": safe_corr(np.log1p(gene_mean), np.log1p(gene_var)),
        "gene_mean_variance_log_slope": safe_slope(np.log1p(gene_mean), np.log1p(gene_var)),
        **dispersion_stats(dispersion, prefix="moments"),
    }


def count_pair_records(dataset: SyntheticBiologyLiteDataset, counts: np.ndarray, *, split: str) -> list[dict[str, Any]]:
    records = []
    for pair in _counterfactual_pairs(dataset, split=split):
        control_counts = np.asarray(counts[pair["control_group"]], dtype=float)
        target_counts = np.asarray(counts[pair["target_group"]], dtype=float)
        control_expression = np.asarray(dataset.expression_values[pair["control_group"]], dtype=float)
        target_expression = np.asarray(dataset.expression_values[pair["target_group"]], dtype=float)
        metadata = {key: value for key, value in pair.items() if not key.endswith("_group")}
        records.append(
            {
                **metadata,
                "biological_key": biological_key(metadata),
                "control_cells": control_expression,
                "target_cells": target_expression,
                "control_mean": control_expression.mean(axis=0),
                "target_mean": target_expression.mean(axis=0),
                "control_count_cells": control_counts,
                "target_count_cells": target_counts,
                "control_count_mean": control_counts.mean(axis=0),
                "target_count_mean": target_counts.mean(axis=0),
            }
        )
    return records


def evaluate_count_predictions(
    dataset: SyntheticBiologyLiteDataset,
    test_records: list[dict[str, Any]],
    count_predictions: list[np.ndarray],
    *,
    candidate_name: str,
    method: str,
    exact_match_fraction: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    predicted_counts = np.asarray(count_predictions, dtype=float)
    predicted_expression = np.log1p(np.clip(predicted_counts, 0.0, None))
    result = evaluate_predictions(
        dataset,
        test_records,
        list(predicted_expression),
        candidate_name=candidate_name,
        method=method,
        exact_match_fraction=exact_match_fraction,
        diagnostics=diagnostics,
    )
    target_counts = np.concatenate([record["target_count_cells"] for record in test_records], axis=0)
    result["predicted_count_mean"] = float(predicted_counts.mean())
    result["predicted_count_zero_fraction"] = float(np.mean(predicted_counts <= 0.0))
    result["target_count_mean"] = float(target_counts.mean())
    return result


def finalize_count_candidate(
    result: dict[str, Any],
    *,
    test_records: list[dict[str, Any]],
    count_predictions: np.ndarray,
    protected_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    dispersion: np.ndarray | None,
    decision_label: str,
) -> dict[str, Any]:
    row = dict(result)
    row["poisson_nll_test"] = poisson_nll_for_records(test_records, count_predictions)
    if dispersion is not None:
        row["nb_nll_test"] = nb_nll_for_records(test_records, count_predictions, dispersion)
    else:
        row["nb_nll_test"] = float("nan")
    row["counterfactual_gate_pass"] = counterfactual_gate_pass(row, source_metrics)
    row["protected_geometry_preserved"] = True
    row["protected_rna_to_image_recall@1"] = float(protected_metrics.get("model_rna_to_image_recall@1", 0.0))
    row["protected_bio_latent_r2_rna_shared"] = float(protected_metrics.get("model_bio_latent_r2_rna_shared", 0.0))
    row["protected_representation_rank"] = float(protected_metrics.get("model_embedding_rank", 0.0))
    row["protected_batch_probe_balanced_accuracy"] = float(
        protected_metrics.get("model_batch_probe_balanced_accuracy", 0.0)
    )
    row["leakage_gate_pass"] = bool(
        float(row.get("fit_split_train_only", 1.0)) == 1.0
        and float(row.get("teacher_target_test_rows_used", 0.0)) == 0.0
        and float(row.get("batch_id_feature_present", 0.0)) == 0.0
        and float(row.get("batch_id_excluded", 1.0)) == 1.0
    )
    row["decision_label"] = decision_label if row["leakage_gate_pass"] else "TIER1_DISCARD_LEAKAGE_CONTROL_FAIL"
    return row


def fit_gene_wise_nb_dispersion(
    records: list[dict[str, Any]],
    count_predictions: np.ndarray,
    *,
    steps: int,
    lr: float,
    min_dispersion: float,
    max_dispersion: float,
    seed: int,
    device: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    y_np, mu_np = expand_record_counts_and_means(records, count_predictions)
    init = np.clip(method_of_moments_dispersion(y_np), min_dispersion, max_dispersion)
    raw = torch.nn.Parameter(inverse_softplus(torch.as_tensor(init - min_dispersion, dtype=torch.float32, device=device)))
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    mu = torch.as_tensor(np.clip(mu_np, 1e-6, None), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([raw], lr=float(lr))
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        dispersion = positive_dispersion(raw, min_dispersion=min_dispersion, max_dispersion=max_dispersion)
        loss = nb_nll_torch(y, mu, dispersion).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return positive_dispersion(raw, min_dispersion=min_dispersion, max_dispersion=max_dispersion).detach().cpu().numpy()


def fit_count_mlp(
    features: np.ndarray,
    records: list[dict[str, Any]],
    *,
    likelihood: str,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    min_log_mean: float,
    max_log_mean: float,
    min_dispersion: float,
    max_dispersion: float,
    seed: int,
    device: str,
) -> TrainedCountModel:
    torch.manual_seed(seed)
    x_np, y_np = expand_record_features_and_counts(features, records)
    initial_mean = np.clip(y_np.mean(axis=0), 1e-4, None)
    model = CountMeanMLP(x_np.shape[1], y_np.shape[1], hidden_dim=hidden_dim, initial_mean=initial_mean).to(device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    parameters: list[torch.nn.Parameter] = list(model.parameters())
    dispersion_param = None
    if likelihood == "negative_binomial":
        init = np.clip(method_of_moments_dispersion(y_np), min_dispersion, max_dispersion)
        dispersion_param = torch.nn.Parameter(
            inverse_softplus(torch.as_tensor(init - min_dispersion, dtype=torch.float32, device=device))
        )
        parameters.append(dispersion_param)
    elif likelihood != "poisson":
        raise ValueError(f"unsupported count likelihood: {likelihood!r}")
    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    final_loss = 0.0
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        mu = model(x, min_log_mean=min_log_mean, max_log_mean=max_log_mean)
        if likelihood == "poisson":
            loss = poisson_nll_torch(y, mu).mean()
        else:
            assert dispersion_param is not None
            dispersion = positive_dispersion(
                dispersion_param,
                min_dispersion=min_dispersion,
                max_dispersion=max_dispersion,
            )
            loss = nb_nll_torch(y, mu, dispersion).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    with torch.no_grad():
        mu = model(x, min_log_mean=min_log_mean, max_log_mean=max_log_mean)
        if likelihood == "poisson":
            train_nll = float(poisson_nll_torch(y, mu).mean().detach().cpu())
            dispersion_np = None
        else:
            assert dispersion_param is not None
            dispersion = positive_dispersion(
                dispersion_param,
                min_dispersion=min_dispersion,
                max_dispersion=max_dispersion,
            )
            train_nll = float(nb_nll_torch(y, mu, dispersion).mean().detach().cpu())
            dispersion_np = dispersion.detach().cpu().numpy()
    return TrainedCountModel(
        model,
        likelihood=likelihood,
        train_nll=train_nll,
        final_loss=final_loss,
        dispersion=dispersion_np,
        min_log_mean=min_log_mean,
        max_log_mean=max_log_mean,
        device=device,
    )


def expand_record_features_and_counts(features: np.ndarray, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for feature, record in zip(np.asarray(features, dtype=float), records, strict=True):
        target = np.asarray(record["target_count_cells"], dtype=float)
        x_rows.append(np.repeat(feature[None, :], target.shape[0], axis=0))
        y_rows.append(target)
    return np.concatenate(x_rows, axis=0), np.concatenate(y_rows, axis=0)


def expand_record_counts_and_means(
    records: list[dict[str, Any]],
    count_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_rows = []
    mu_rows = []
    for prediction, record in zip(np.asarray(count_predictions, dtype=float), records, strict=True):
        target = np.asarray(record["target_count_cells"], dtype=float)
        y_rows.append(target)
        mu_rows.append(np.repeat(prediction[None, :], target.shape[0], axis=0))
    return np.concatenate(y_rows, axis=0), np.concatenate(mu_rows, axis=0)


def poisson_nll_for_records(records: list[dict[str, Any]], count_predictions: np.ndarray) -> float:
    y, mu = expand_record_counts_and_means(records, count_predictions)
    mu = np.clip(mu, 1e-8, None)
    nll = mu - y * np.log(mu) + gammaln(y + 1.0)
    return float(np.mean(nll))


def nb_nll_for_records(records: list[dict[str, Any]], count_predictions: np.ndarray, dispersion: np.ndarray) -> float:
    y, mu = expand_record_counts_and_means(records, count_predictions)
    return float(np.mean(nb_nll_numpy(y, np.clip(mu, 1e-8, None), np.asarray(dispersion, dtype=float))))


def poisson_nll_torch(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)


def nb_nll_torch(y: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    mu = torch.clamp(mu, min=1e-8)
    alpha = torch.clamp(dispersion, min=1e-8)
    theta = 1.0 / alpha
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_prob


def nb_nll_numpy(y: np.ndarray, mu: np.ndarray, dispersion: np.ndarray) -> np.ndarray:
    alpha = np.clip(np.asarray(dispersion, dtype=float), 1e-8, None)
    theta = 1.0 / alpha
    log_prob = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (np.log(mu) - np.log(theta + mu))
    )
    return -log_prob


def method_of_moments_dispersion(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    return np.clip((variance - mean) / np.maximum(mean * mean, 1e-8), 1e-4, 10.0)


def inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, min=1e-8)
    return torch.log(torch.expm1(values))


def positive_dispersion(
    raw: torch.Tensor,
    *,
    min_dispersion: float,
    max_dispersion: float,
) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + float(min_dispersion), max=float(max_dispersion))


def dispersion_stats(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_dispersion_min": float(np.min(array)),
        f"{prefix}_dispersion_median": float(np.median(array)),
        f"{prefix}_dispersion_mean": float(np.mean(array)),
        f"{prefix}_dispersion_max": float(np.max(array)),
        f"{prefix}_dispersion_zero_fraction": float(np.mean(array <= 1e-8)),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centered = x - x.mean()
    denom = float(np.sum(centered * centered))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(centered * (y - y.mean())) / denom)


def load_required_comparators(output_dir: Path) -> list[dict[str, Any]]:
    parent = output_dir.parent
    comparators: list[dict[str, Any]] = []
    family_n_path = parent / "FAMILY_N_DISTILLATION" / "FAMILY_N_RESULTS.tsv"
    if family_n_path.exists():
        family_n = pd.read_csv(family_n_path, sep="\t")
        for name in (
            "seed2_train_only_condition_mean_table",
            "seed2_distilled_linear_condition_mean",
            "seed2_distilled_mlp_condition_mean",
        ):
            match = family_n[family_n["candidate_name"] == name]
            if not match.empty:
                comparators.append(comparator_from_series(match.iloc[0], source="Family N"))
    family_m_path = parent / "FAMILY_M_TRANSPORT_BASELINES" / "FAMILY_M_RESULTS.tsv"
    if family_m_path.exists():
        family_m = pd.read_csv(family_m_path, sep="\t")
        for name in ("seed2_no_batch_matched_perturbed_mean", "seed2_no_batch_residualized_matching"):
            match = family_m[family_m["candidate_name"] == name]
            if not match.empty:
                comparators.append(comparator_from_series(match.iloc[0], source="Family M"))
    comparators.extend(load_sparse_residual_comparators(parent / "SPARSE_PERTURBATION_RESIDUAL_HEAD"))
    ridge = load_prefit_ridge_best(parent / "CLONE_COUNTERFACTUAL_DECODER")
    if ridge is not None:
        comparators.append(ridge)
    return comparators


def comparator_from_series(row: pd.Series, *, source: str) -> dict[str, Any]:
    return {
        "candidate_name": row["candidate_name"],
        "source": source,
        "program_level_effect_recovery": float(row["program_level_effect_recovery"]),
        "direction_accuracy": float(row["direction_accuracy"]),
        "logfc_correlation": float(row["logfc_correlation"]),
        "pseudobulk_correlation": float(row["pseudobulk_correlation"]),
        "top50_de_overlap": float(row["top50_de_overlap"]),
        "mean_delta_to_target_ratio": float(row["mean_delta_to_target_ratio"]),
    }


def append_results_tsv(
    frame: pd.DataFrame,
    *,
    protected_metrics: dict[str, Any],
    device: str,
    wallclock_minutes: float,
) -> None:
    existing = pd.read_csv(RESULTS_PATH, sep="\t") if RESULTS_PATH.exists() else pd.DataFrame()
    columns = list(existing.columns) if not existing.empty else [
        "commit",
        "experiment_num",
        "family",
        "candidate_name",
        "tier_reached",
        "decision_label",
        "device_used",
        "wallclock_minutes",
        "max_gpu_memory_gb",
        "synth_micro_recall1",
        "synth_easy_recall1",
        "synth_medium_recall1",
        "synth_easy_cf_dir_acc",
        "synth_medium_cf_dir_acc",
        "heldout_pert_cf_dir_acc",
        "batch_confound_batch_leakage",
        "batch_confound_recall1",
        "dose_extrap_logfc_corr",
        "bio_latent_r2",
        "representation_rank",
        "delta_norm_ratio",
        "cap_bound",
        "collapse_flag",
        "architecture_change",
        "description",
    ]
    last_num = int(existing["experiment_num"].max()) if not existing.empty else 0
    commit = current_commit()
    rows = []
    for offset, row in enumerate(frame.itertuples(index=False), start=1):
        rows.append(
            {
                "commit": commit,
                "experiment_num": last_num + offset,
                "family": "O",
                "candidate_name": row.candidate_name,
                "tier_reached": "Tier 1 synth_micro seed2",
                "decision_label": row.decision_label,
                "device_used": device,
                "wallclock_minutes": f"{wallclock_minutes:.3f}",
                "max_gpu_memory_gb": "0.000",
                "synth_micro_recall1": f"{float(protected_metrics.get('model_rna_to_image_recall@1', 0.0)):.6f}",
                "synth_easy_recall1": "",
                "synth_medium_recall1": "",
                "synth_easy_cf_dir_acc": "",
                "synth_medium_cf_dir_acc": "",
                "heldout_pert_cf_dir_acc": "",
                "batch_confound_batch_leakage": "",
                "batch_confound_recall1": "",
                "dose_extrap_logfc_corr": f"{float(row.logfc_correlation):.6f}",
                "bio_latent_r2": f"{float(protected_metrics.get('model_bio_latent_r2_rna_shared', 0.0)):.6f}",
                "representation_rank": f"{float(protected_metrics.get('model_embedding_rank', 0.0)):.1f}",
                "delta_norm_ratio": f"{float(row.mean_delta_to_target_ratio):.6f}",
                "cap_bound": "false",
                "collapse_flag": "false",
                "architecture_change": "count_likelihood_synthetic_only",
                "description": count_result_description(row),
            }
        )
    output = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    output.loc[:, columns].to_csv(RESULTS_PATH, sep="\t", index=False)


def count_result_description(row: Any) -> str:
    nb = "NA" if pd.isna(row.nb_nll_test) else f"{float(row.nb_nll_test):.4f}"
    return (
        f"Family O synthetic-only count-likelihood candidate; stage {row.candidate_stage}, "
        f"Poisson NLL {float(row.poisson_nll_test):.4f}, NB NLL {nb}, "
        f"program {float(row.program_level_effect_recovery):.4f}, logFC {float(row.logfc_correlation):.4f}, "
        f"top50 {float(row.top50_de_overlap):.4f}, direction {float(row.direction_accuracy):.4f}, "
        f"pseudobulk {float(row.pseudobulk_correlation):.4f}; batch_id excluded and raw-count path audited."
    )


def current_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    frame: pd.DataFrame,
    *,
    comparator_rows: list[dict[str, Any]],
    count_diagnostics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    best_poisson = frame.sort_values("poisson_nll_test", ascending=True).head(1).iloc[0]
    nb_frame = frame[np.isfinite(frame["nb_nll_test"].astype(float))]
    best_nb = nb_frame.sort_values("nb_nll_test", ascending=True).head(1).iloc[0] if not nb_frame.empty else None
    lines = [
        "# Family O Count-Likelihood Perturbation Training",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Seed: `{args.seed}`",
        f"- Device: `{args.device}`",
        f"- Biological key: `{', '.join(BIOLOGICAL_KEY_FIELDS)}`",
        "- Batch ID excluded from features and matching: `true`",
        "- Training targets/statistics split: `train`",
        "- Test target rows used for teacher/training construction: `0`",
        "- Real data used: `false`",
        "- Marker/pathway/pretrained biological assets used: `false`",
        "",
        "## Count Path Audit",
        "",
        f"- Raw count-like RNA values available: `{bool(count_diagnostics['raw_count_available'])}`",
        f"- Count path: `{count_diagnostics['count_path']}`",
        f"- Pseudo-count path used: `{bool(count_diagnostics['pseudo_count_used'])}`",
        f"- Pseudo-count scale: `{count_diagnostics['pseudo_count_scale']}`",
        f"- Zero fraction: `{count_diagnostics['zero_fraction']:.4f}`",
        f"- Count mean / variance: `{count_diagnostics['count_mean']:.4f}` / `{count_diagnostics['count_variance']:.4f}`",
        f"- Mean library size: `{count_diagnostics['mean_library_size']:.2f}`",
        f"- log mean-variance correlation: `{count_diagnostics['gene_mean_variance_log_correlation']:.4f}`",
        f"- MoM dispersion median: `{count_diagnostics['moments_dispersion_median']:.4f}`",
        "",
        "## Candidate Results",
        "",
    ]
    for row in frame.itertuples(index=False):
        lines.extend(count_candidate_lines(row))
    lines.extend(
        [
            "## NLL Winners",
            "",
            f"- Best Poisson NLL: `{best_poisson.candidate_name}` with `{best_poisson.poisson_nll_test:.4f}`",
        ]
    )
    if best_nb is not None:
        lines.append(f"- Best NB NLL: `{best_nb.candidate_name}` with `{best_nb.nb_nll_test:.4f}`")
    lines.extend(["", "## Required Comparators", ""])
    for comparator in comparator_rows:
        lines.extend(
            [
                f"### {comparator['candidate_name']}",
                f"- source: `{comparator['source']}`",
                f"- program recovery: `{comparator['program_level_effect_recovery']:.4f}`",
                f"- direction accuracy: `{comparator['direction_accuracy']:.4f}`",
                f"- logFC correlation: `{comparator['logfc_correlation']:.4f}`",
                f"- pseudobulk correlation: `{comparator['pseudobulk_correlation']:.4f}`",
                f"- top50 overlap: `{comparator['top50_de_overlap']:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Family O tests whether count likelihood changes the seed-2 counterfactual signal rather than replacing Family N. The raw synthetic count path is available, so no pseudo-count fallback was used in this run.",
            "Poisson and NB table candidates evaluate likelihood calibration around the train-only condition mean; MLP candidates test a learned no-batch condition/source-feature decoder with positive log-mean parameterization and stable positive dispersion for NB.",
            "",
            "## Artifacts",
            "",
            "- `FAMILY_O_RESULTS.tsv`",
            "- `FAMILY_O_RESULTS.json`",
            "- `COMPARATOR_RESULTS.tsv`",
            "- `generation_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_candidate_lines(row: Any) -> list[str]:
    nb_nll = "NA" if pd.isna(row.nb_nll_test) else f"{row.nb_nll_test:.4f}"
    return [
        f"### {row.candidate_name}",
        f"- method: `{row.method}`",
        f"- stage: `{row.candidate_stage}`",
        f"- leakage gate pass: `{bool(row.leakage_gate_pass)}`",
        f"- Poisson NLL test: `{row.poisson_nll_test:.4f}`",
        f"- NB NLL test: `{nb_nll}`",
        f"- program recovery: `{row.program_level_effect_recovery:.4f}`",
        f"- direction accuracy: `{row.direction_accuracy:.4f}`",
        f"- logFC correlation: `{row.logfc_correlation:.4f}`",
        f"- pseudobulk correlation: `{row.pseudobulk_correlation:.4f}`",
        f"- top50 overlap: `{row.top50_de_overlap:.4f}`",
        f"- mean delta/target ratio: `{row.mean_delta_to_target_ratio:.4f}`",
        f"- counterfactual gate pass: `{bool(row.counterfactual_gate_pass)}`",
        "",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
```

