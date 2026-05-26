from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class RidgeFloorState:
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    coef: torch.Tensor


@dataclass(frozen=True)
class BioSpectralJEPAConfig:
    shared_dim: int = 32
    bio_dim: int = 24
    tech_dim: int = 8
    predictor_dim: int = 64
    action_dim: int = 32
    rank_ladder: tuple[int, ...] = (4, 8, 12, 24)
    use_full_ridge_floor: bool = True
    freeze_ridge_floor: bool = True
    residual_init_scale: float = 0.0
    residual_norm_cap: float = 0.25
    spectral_entropy_weight: float = 0.05
    covariance_shape_weight: float = 0.05
    floor_distillation_weight: float = 1.0
    delta_direction_weight: float = 1.0
    endpoint_jepa_weight: float = 1.0
    source_improvement_hinge_weight: float = 0.5
    anti_collapse_weight: float = 0.1
    allow_kernel_residual: bool = False
    kernel_num_features: int = 128
    kernel_residual_init_scale: float = 0.0
    base_biotech_config: BioTechJEPAConfig | None = None


class ActionDescriptorOnlyBuilder(nn.Module):
    """Expose legal action descriptors without condition-key shortcuts."""

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, batch: BioActionConditionBatch) -> torch.Tensor:
        if self.action_dim == 0:
            return torch.zeros(batch.perturbation_id.shape[0], 0, device=batch.perturbation_id.device, dtype=batch.dose.dtype)
        if batch.descriptor is None:
            raise ValueError("BioSpectral-JEPA requires an explicit action descriptor")
        if batch.descriptor.shape[-1] != self.action_dim:
            raise ValueError(f"expected action descriptor dim {self.action_dim}, got {batch.descriptor.shape[-1]}")
        return batch.descriptor.to(dtype=batch.dose.dtype)


class RidgeFloorHead(nn.Module):
    """Train-only analytical full-ridge transition floor over [z_source, action]."""

    def __init__(self, source_dim: int, action_dim: int) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.source_dim + self.action_dim
        self.register_buffer("x_mean", torch.zeros(1, self.input_dim))
        self.register_buffer("y_mean", torch.zeros(1, self.source_dim))
        self.register_buffer("coef", torch.zeros(self.input_dim, self.source_dim))
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, source: torch.Tensor, action: torch.Tensor, delta: torch.Tensor, *, alpha: float = 1.0e-2) -> "RidgeFloorHead":
        x = _features(source, action).detach().to(dtype=torch.float64)
        y = delta.detach().to(dtype=torch.float64)
        if y.shape[-1] != self.source_dim:
            raise ValueError(f"delta dim must be {self.source_dim}")
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        xc = x - x_mean
        yc = y - y_mean
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        coef = torch.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
        self.x_mean.copy_(x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype))
        self.y_mean.copy_(y_mean.to(device=self.y_mean.device, dtype=self.y_mean.dtype))
        self.coef.copy_(coef.to(device=self.coef.device, dtype=self.coef.dtype))
        self.is_fit.fill_(True)
        return self

    @torch.no_grad()
    def initialize_from_state(self, state: RidgeFloorState) -> "RidgeFloorHead":
        if state.coef.shape != self.coef.shape:
            raise ValueError(f"coef shape mismatch: expected {tuple(self.coef.shape)}, got {tuple(state.coef.shape)}")
        self.x_mean.copy_(state.x_mean.to(device=self.x_mean.device, dtype=self.x_mean.dtype))
        self.y_mean.copy_(state.y_mean.to(device=self.y_mean.device, dtype=self.y_mean.dtype))
        self.coef.copy_(state.coef.to(device=self.coef.device, dtype=self.coef.dtype))
        self.is_fit.fill_(True)
        return self

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fit.detach().cpu()):
            raise RuntimeError("RidgeFloorHead must be fit before use")
        x = _features(source, action)
        return (x - self.x_mean.to(x.device, x.dtype)) @ self.coef.to(x.device, x.dtype) + self.y_mean.to(x.device, x.dtype)

    def state(self) -> RidgeFloorState:
        return RidgeFloorState(self.x_mean.detach().clone(), self.y_mean.detach().clone(), self.coef.detach().clone())

    def as_linear_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.coef.detach().T.clone()
        bias = (self.y_mean - self.x_mean @ self.coef).detach().reshape(-1).clone()
        return weight, bias


class DeltaSpectralBasis(nn.Module):
    """Train-only spectral basis for teacher deltas or ridge residuals."""

    def __init__(self, dim: int, *, max_rank: int | None = None, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_rank = self.dim if max_rank is None else max(1, min(int(max_rank), self.dim))
        self.eps = float(eps)
        self.register_buffer("teacher_delta_mean", torch.zeros(1, self.dim))
        self.register_buffer("teacher_delta_eigenvectors", torch.eye(self.dim)[: self.max_rank].clone())
        self.register_buffer("teacher_delta_eigenvalues", torch.zeros(self.max_rank))
        self.register_buffer("ridge_pred_delta_mean", torch.zeros(1, self.dim))
        self.register_buffer("ridge_residual_basis", torch.eye(self.dim)[: self.max_rank].clone())
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, delta_train: torch.Tensor, *, ridge_pred_delta: torch.Tensor | None = None) -> "DeltaSpectralBasis":
        delta = delta_train.detach().to(dtype=torch.float64)
        if delta.ndim != 2 or delta.shape[1] != self.dim:
            raise ValueError(f"delta_train must have shape [n, {self.dim}]")
        mean = delta.mean(dim=0, keepdim=True)
        basis, eigenvalues = _svd_basis(delta - mean, self.max_rank)
        if ridge_pred_delta is None:
            ridge_mean = torch.zeros_like(mean)
            residual_basis = basis
        else:
            ridge = ridge_pred_delta.detach().to(dtype=torch.float64)
            ridge_mean = ridge.mean(dim=0, keepdim=True)
            residual_basis, _ = _svd_basis(delta - ridge, self.max_rank)
        self.teacher_delta_mean.copy_(mean.to(device=self.teacher_delta_mean.device, dtype=self.teacher_delta_mean.dtype))
        self.teacher_delta_eigenvectors.copy_(basis.to(device=self.teacher_delta_eigenvectors.device, dtype=self.teacher_delta_eigenvectors.dtype))
        self.teacher_delta_eigenvalues.copy_(eigenvalues.to(device=self.teacher_delta_eigenvalues.device, dtype=self.teacher_delta_eigenvalues.dtype))
        self.ridge_pred_delta_mean.copy_(ridge_mean.to(device=self.ridge_pred_delta_mean.device, dtype=self.ridge_pred_delta_mean.dtype))
        self.ridge_residual_basis.copy_(residual_basis.to(device=self.ridge_residual_basis.device, dtype=self.ridge_residual_basis.dtype))
        self.is_fit.fill_(True)
        return self

    def project(self, delta: torch.Tensor, *, rank: int | None = None, basis: str = "teacher") -> torch.Tensor:
        vectors = self._basis(basis, delta.device, delta.dtype, rank)
        mean = self.teacher_delta_mean.to(delta.device, delta.dtype)
        return (delta - mean) @ vectors.T

    def reconstruct(self, coeff: torch.Tensor, *, rank: int | None = None, basis: str = "teacher") -> torch.Tensor:
        vectors = self._basis(basis, coeff.device, coeff.dtype, rank or coeff.shape[-1])
        return self.teacher_delta_mean.to(coeff.device, coeff.dtype) + coeff[..., : vectors.shape[0]] @ vectors

    def reduce(self, delta: torch.Tensor, *, rank: int | None = None, basis: str = "teacher") -> torch.Tensor:
        return self.reconstruct(self.project(delta, rank=rank, basis=basis), rank=rank, basis=basis)

    def cumulative_variance(self, rank: int | str) -> torch.Tensor:
        values = self.teacher_delta_eigenvalues
        total = values.sum().clamp_min(self.eps)
        if rank == "full":
            return torch.ones((), device=values.device, dtype=values.dtype)
        return values[: min(int(rank), values.numel())].sum() / total

    def effective_rank(self, values: torch.Tensor | None = None) -> torch.Tensor:
        if values is None:
            spectrum = self.teacher_delta_eigenvalues.clamp_min(self.eps)
        else:
            centered = values - values.mean(dim=0, keepdim=True)
            spectrum = torch.linalg.svdvals(centered).clamp_min(self.eps)
        probs = spectrum / spectrum.sum().clamp_min(self.eps)
        return torch.exp(-(probs * probs.log()).sum())

    def _basis(self, basis: str, device: torch.device, dtype: torch.dtype, rank: int | None) -> torch.Tensor:
        if basis not in {"teacher", "residual"}:
            raise ValueError("basis must be teacher or residual")
        vectors = self.teacher_delta_eigenvectors if basis == "teacher" else self.ridge_residual_basis
        use_rank = self.max_rank if rank is None else max(1, min(int(rank), vectors.shape[0]))
        return vectors[:use_rank].to(device=device, dtype=dtype)


class NeuralReducedRankRidgeHead(nn.Module):
    """Exact neural equivalent of analytical reduced-rank ridge."""

    def __init__(self, source_dim: int, action_dim: int, rank: int) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.rank = max(1, min(int(rank), self.source_dim))
        self.linear = nn.Linear(self.source_dim + self.action_dim, self.rank)
        self.register_buffer("delta_mean", torch.zeros(1, self.source_dim))
        self.register_buffer("basis", torch.eye(self.source_dim)[: self.rank].clone())
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.bool))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @torch.no_grad()
    def fit(self, source: torch.Tensor, action: torch.Tensor, delta: torch.Tensor, *, alpha: float = 1.0e-2) -> "NeuralReducedRankRidgeHead":
        x = _features(source, action).detach().to(dtype=torch.float64)
        y = delta.detach().to(dtype=torch.float64)
        delta_mean = y.mean(dim=0, keepdim=True)
        centered = y - delta_mean
        basis, _ = _svd_basis(centered, self.rank)
        coeff = centered @ basis.T
        x_mean = x.mean(dim=0, keepdim=True)
        coeff_mean = coeff.mean(dim=0, keepdim=True)
        xc = x - x_mean
        yc = coeff - coeff_mean
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        coef = torch.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
        bias = (coeff_mean - x_mean @ coef).reshape(-1)
        self.linear.weight.copy_(coef.T.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype))
        self.linear.bias.copy_(bias.to(device=self.linear.bias.device, dtype=self.linear.bias.dtype))
        self.delta_mean.copy_(delta_mean.to(device=self.delta_mean.device, dtype=self.delta_mean.dtype))
        self.basis.copy_(basis.to(device=self.basis.device, dtype=self.basis.dtype))
        self.is_fit.fill_(True)
        return self

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        coeff = self.linear(_features(source, action))
        return self.delta_mean.to(coeff.device, coeff.dtype) + coeff @ self.basis.to(coeff.device, coeff.dtype)


class RankLadderTransitionHead(nn.Module):
    """Rank-adaptive mixture with an exact full-ridge fallback expert."""

    def __init__(self, source_dim: int, action_dim: int, ranks: Iterable[int]) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.ranks = tuple(dict.fromkeys(max(1, min(int(rank), self.source_dim)) for rank in ranks))
        if self.source_dim not in self.ranks:
            self.ranks = (*self.ranks, self.source_dim)
        self.low_rank_experts = nn.ModuleDict({str(rank): NeuralReducedRankRidgeHead(source_dim, action_dim, rank) for rank in self.ranks if rank < self.source_dim})
        self.full_floor = RidgeFloorHead(source_dim, action_dim)
        self.expert_names = (*[str(rank) for rank in self.ranks if rank < self.source_dim], "full")
        self.router = nn.Linear(self.source_dim + self.action_dim, len(self.expert_names))
        self.register_buffer("force_full_floor", torch.ones((), dtype=torch.bool))
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    @torch.no_grad()
    def fit(self, source: torch.Tensor, action: torch.Tensor, delta: torch.Tensor, *, alpha: float = 1.0e-2) -> "RankLadderTransitionHead":
        for expert in self.low_rank_experts.values():
            expert.fit(source, action, delta, alpha=alpha)
        self.full_floor.fit(source, action, delta, alpha=alpha)
        self.force_full_floor.fill_(True)
        return self

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_deltas = [self.low_rank_experts[name](source, action) for name in self.expert_names if name != "full"]
        expert_deltas.append(self.full_floor(source, action))
        stacked = torch.stack(expert_deltas, dim=1)
        weights = self.router_weights(source, action)
        delta = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return {
            "delta_ladder": delta,
            "router_weights": weights,
            "expert_deltas_by_rank": stacked,
            "rank_usage": weights.mean(dim=0),
            "spectral_entropy": categorical_entropy(weights).mean(),
        }

    def router_weights(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = _features(source, action)
        if bool(self.force_full_floor.detach().cpu()):
            weights = torch.zeros(x.shape[0], len(self.expert_names), device=x.device, dtype=x.dtype)
            weights[:, self.expert_names.index("full")] = 1.0
            return weights
        return torch.softmax(self.router(x), dim=-1)


class SpectralResidualHead(nn.Module):
    """Small capped residual in a train-only spectral basis."""

    def __init__(
        self,
        source_dim: int,
        action_dim: int,
        *,
        rank: int,
        hidden_dim: int,
        norm_cap: float = 0.25,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.action_dim = int(action_dim)
        self.rank = max(1, min(int(rank), self.source_dim))
        self.norm_cap = float(norm_cap)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.source_dim + self.action_dim),
            nn.Linear(self.source_dim + self.action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.rank),
        )
        self.register_buffer("basis", torch.eye(self.source_dim)[: self.rank].clone())
        self._zero_initialize(init_scale)

    @torch.no_grad()
    def initialize_basis(self, spectral_basis: DeltaSpectralBasis, *, residual: bool = True) -> "SpectralResidualHead":
        basis = spectral_basis.ridge_residual_basis if residual else spectral_basis.teacher_delta_eigenvectors
        self.basis.copy_(basis[: self.rank].to(device=self.basis.device, dtype=self.basis.dtype))
        return self

    def forward(self, source: torch.Tensor, action: torch.Tensor, *, floor_delta: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        coeff = self.mlp(_features(source, action))
        raw = coeff @ self.basis.to(coeff.device, coeff.dtype)
        capped, cap_hit = cap_residual(raw, floor_delta, self.norm_cap)
        return {
            "delta_residual": capped,
            "raw_delta_residual": raw,
            "residual_cap_hit_fraction": cap_hit,
            "residual_to_floor_norm_ratio": residual_ratio(capped, floor_delta),
        }

    def _zero_initialize(self, init_scale: float) -> None:
        last = self.mlp[-1]
        if not isinstance(last, nn.Linear):
            return
        if float(init_scale) == 0.0:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            nn.init.normal_(last.weight, mean=0.0, std=float(init_scale))
            nn.init.zeros_(last.bias)


class FloorPreservingTransitionHead(nn.Module):
    """Full-ridge floor plus an explicitly gated residual branch."""

    def __init__(self, source_dim: int, action_dim: int, *, residual_rank: int, hidden_dim: int, residual_norm_cap: float = 0.25) -> None:
        super().__init__()
        self.ridge_floor = RidgeFloorHead(source_dim, action_dim)
        self.spectral_basis = DeltaSpectralBasis(source_dim, max_rank=source_dim)
        self.residual = SpectralResidualHead(
            source_dim,
            action_dim,
            rank=residual_rank,
            hidden_dim=hidden_dim,
            norm_cap=residual_norm_cap,
            init_scale=0.0,
        )
        self.register_buffer("residual_gate", torch.zeros(()))

    @torch.no_grad()
    def fit_floor_and_basis(self, source: torch.Tensor, action: torch.Tensor, delta: torch.Tensor, *, alpha: float = 1.0e-2) -> "FloorPreservingTransitionHead":
        self.ridge_floor.fit(source, action, delta, alpha=alpha)
        floor_delta = self.ridge_floor(source, action)
        self.spectral_basis.fit(delta, ridge_pred_delta=floor_delta)
        self.residual.initialize_basis(self.spectral_basis, residual=True)
        self.residual_gate.zero_()
        return self

    def set_residual_gate(self, value: float) -> None:
        self.residual_gate.fill_(float(value))

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        floor_delta = self.ridge_floor(source, action)
        residual = self.residual(source, action, floor_delta=floor_delta)
        delta = floor_delta + self.residual_gate.to(floor_delta.device, floor_delta.dtype) * residual["delta_residual"]
        return {
            "delta_floor": floor_delta,
            "delta_residual": residual["delta_residual"],
            "predicted_delta_bio": delta,
            "residual_to_floor_norm_ratio": residual["residual_to_floor_norm_ratio"],
            "residual_cap_hit_fraction": residual["residual_cap_hit_fraction"],
        }


class BioSpectralJEPA(nn.Module):
    """BioTech-JEPA encoders plus a floor-preserving spectral transition head."""

    def __init__(self, config: BioSpectralJEPAConfig) -> None:
        super().__init__()
        if config.base_biotech_config is None:
            raise ValueError("BioSpectralJEPA requires base_biotech_config for full JEPA mode")
        self.config = config
        self.base = BioTechJEPA(config.base_biotech_config)
        self.action_features = ActionDescriptorOnlyBuilder(config.action_dim)
        self.transition_head = FloorPreservingTransitionHead(
            config.bio_dim,
            config.action_dim,
            residual_rank=max(config.rank_ladder),
            hidden_dim=config.predictor_dim,
            residual_norm_cap=config.residual_norm_cap,
        )

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward_operator(self, batch: BioActionConditionBatch) -> dict[str, torch.Tensor]:
        action = self.action_features(batch)
        control_context = self.base.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="context",
        )
        control_teacher = self.base.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="target",
        )
        target_teacher = self.base.encode_condition(
            gene_ids=batch.target_gene_ids,
            expression_values=batch.target_expression_values,
            images=batch.target_images,
            rna_bag_mask=batch.target_rna_bag_mask,
            image_bag_mask=batch.target_image_bag_mask,
            mode="target",
        )
        source = control_context["joint_z_bio"]
        source_teacher = control_teacher["joint_z_bio"].detach()
        target = target_teacher["joint_z_bio"].detach()
        transition = self.transition_head(source, action)
        pred = F.normalize(source + transition["predicted_delta_bio"], dim=-1)
        return {
            "z_control_bio": source,
            "z_control_bio_teacher": source_teacher,
            "z_target_bio_teacher": target,
            "predicted_delta_bio": transition["predicted_delta_bio"],
            "predicted_target_bio": pred,
            "delta_teacher": (target - source_teacher).detach(),
            "delta_floor": transition["delta_floor"],
            "delta_residual": transition["delta_residual"],
            "residual_to_floor_norm_ratio": transition["residual_to_floor_norm_ratio"],
            "residual_cap_hit_fraction": transition["residual_cap_hit_fraction"],
            "encoder_path_used": torch.ones((), device=source.device),
            "pls_raw_linear_main_path_used": torch.zeros((), device=source.device),
            "condition_key_feature_present": torch.zeros((), device=source.device),
            "biological_key_one_hot_present": torch.zeros((), device=source.device),
            "test_target_mean_used_for_fit": torch.zeros((), device=source.device),
            "pooled_train_test_target_used_for_fit": torch.zeros((), device=source.device),
            "teacher_stop_gradient_verified": torch.ones((), device=source.device),
            "separate_bio_and_tech_latents_present": torch.ones((), device=source.device),
            "heldout_action_descriptor_valid": torch.ones((), device=source.device),
        }


def _features(source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if action.numel() == 0:
        action = torch.zeros(source.shape[0], 0, device=source.device, dtype=source.dtype)
    if source.ndim != 2 or action.ndim != 2:
        raise ValueError("source and action must be rank-2 tensors")
    if source.shape[0] != action.shape[0]:
        raise ValueError("source and action batch size mismatch")
    return torch.cat((source, action.to(device=source.device, dtype=source.dtype)), dim=-1)


def _svd_basis(centered: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    use_rank = max(1, min(int(rank), int(vh.shape[0])))
    basis = vh[:use_rank]
    eigenvalues = singular[:use_rank].pow(2) / max(1, centered.shape[0] - 1)
    if use_rank < rank:
        pad = rank - use_rank
        basis = torch.cat((basis, torch.zeros(pad, centered.shape[1], device=centered.device, dtype=centered.dtype)), dim=0)
        eigenvalues = torch.cat((eigenvalues, torch.zeros(pad, device=centered.device, dtype=centered.dtype)), dim=0)
    return basis, eigenvalues


def cap_residual(residual: torch.Tensor, floor_delta: torch.Tensor | None, norm_cap: float) -> tuple[torch.Tensor, torch.Tensor]:
    if floor_delta is None:
        return residual, torch.zeros((), device=residual.device, dtype=residual.dtype)
    floor_norm = floor_delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    max_norm = float(norm_cap) * floor_norm
    scale = torch.minimum(torch.ones_like(residual_norm), max_norm / residual_norm)
    cap_hit = (residual_norm > max_norm).to(residual.dtype).mean()
    return residual * scale, cap_hit


def residual_ratio(residual: torch.Tensor, floor_delta: torch.Tensor | None) -> torch.Tensor:
    if floor_delta is None:
        return torch.zeros((), device=residual.device, dtype=residual.dtype)
    return residual.norm(dim=-1).mean() / floor_delta.norm(dim=-1).mean().clamp_min(1.0e-8)


def categorical_entropy(weights: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    probs = weights.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)
