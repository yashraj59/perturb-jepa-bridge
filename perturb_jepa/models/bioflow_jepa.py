from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class BioFlowJEPAConfig:
    base_biotech_config: BioTechJEPAConfig
    action_dim: int = 0
    transition_mode: str = "vector_field"
    flow_steps: int = 4
    flow_tau_samples: int = 1
    use_delta_whitening: bool = True
    delta_whitening_rank: int | None = None
    use_tangent_projection: bool = True
    use_source_improvement_hinge: bool = True
    source_improvement_margin: float = 0.02
    use_action_conditioned_film: bool = True
    use_low_rank_operator: bool = True
    low_rank_operator_rank: int = 8
    freeze_encoders_for_transition_steps: int = 100
    transition_lr_multiplier: float = 3.0
    endpoint_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    delta_direction_loss_weight: float = 2.0
    delta_magnitude_loss_weight: float = 0.2
    source_improvement_loss_weight: float = 2.0
    delta_rank_variance_weight: float = 0.1
    action_negative_loss_weight: float = 0.2
    zero_action_identity_weight: float = 0.5
    z_tech_leakage_penalty_weight: float = 0.0


class DeltaWhitening(nn.Module):
    """Train-split-only diagonal/PCA whitener for teacher deltas."""

    def __init__(self, dim: int, *, rank: int | None = None, eps: float = 1.0e-3) -> None:
        super().__init__()
        if rank is not None and rank <= 0:
            raise ValueError("rank must be positive or None")
        self.dim = int(dim)
        self.rank = None if rank is None else min(int(rank), self.dim)
        self.eps = float(eps)
        used_rank = self.dim if self.rank is None else self.rank
        self.register_buffer("mean", torch.zeros(self.dim))
        self.register_buffer("basis", torch.eye(self.dim)[:, :used_rank].clone())
        self.register_buffer("scale", torch.ones(used_rank))
        self.register_buffer("explained_variance", torch.ones(()))
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, delta_train: torch.Tensor) -> "DeltaWhitening":
        if delta_train.ndim != 2 or delta_train.shape[1] != self.dim:
            raise ValueError(f"delta_train must have shape [batch, {self.dim}]")
        delta = delta_train.detach().to(dtype=torch.float32)
        mean = delta.mean(dim=0)
        centered = delta - mean
        if self.rank is None or self.rank >= self.dim:
            basis = torch.eye(self.dim, device=delta.device, dtype=delta.dtype)
            scale = centered.std(dim=0, unbiased=False).clamp_min(self.eps)
            explained = torch.ones((), device=delta.device, dtype=delta.dtype)
        else:
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            rank = min(self.rank, int(vh.shape[0]))
            basis = vh[:rank].T.contiguous()
            coeff = centered @ basis
            scale = coeff.std(dim=0, unbiased=False).clamp_min(self.eps)
            denom = singular.pow(2).sum().clamp_min(self.eps)
            explained = singular[:rank].pow(2).sum() / denom
        self.mean = mean.to(device=self.mean.device)
        self.basis = basis.to(device=self.basis.device)
        self.scale = scale.to(device=self.scale.device)
        self.explained_variance = explained.to(device=self.explained_variance.device)
        self.is_fit = torch.ones((), dtype=torch.bool, device=self.is_fit.device)
        return self

    def whiten(self, delta: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fit.detach().cpu()):
            return delta
        mean = self.mean.to(device=delta.device, dtype=delta.dtype)
        basis = self.basis.to(device=delta.device, dtype=delta.dtype)
        scale = self.scale.to(device=delta.device, dtype=delta.dtype)
        return ((delta - mean) @ basis) / scale

    def unwhiten(self, delta_w: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fit.detach().cpu()):
            return delta_w
        mean = self.mean.to(device=delta_w.device, dtype=delta_w.dtype)
        basis = self.basis.to(device=delta_w.device, dtype=delta_w.dtype)
        scale = self.scale.to(device=delta_w.device, dtype=delta_w.dtype)
        return (delta_w * scale) @ basis.T + mean


class ActionConditionedVectorField(nn.Module):
    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        *,
        hidden_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        if action_dim < 0:
            raise ValueError("action_dim must be non-negative")
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.use_film = bool(use_film)
        self.z_norm = nn.LayerNorm(z_dim)
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.context_proj = nn.Linear(z_dim, hidden_dim)
        self.tau_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_mlp = nn.Sequential(
            nn.LayerNorm(max(1, action_dim)),
            nn.Linear(max(1, action_dim), hidden_dim * (2 if use_film else 1)),
            nn.GELU(),
            nn.Linear(hidden_dim * (2 if use_film else 1), hidden_dim * (2 if use_film else 1)),
        )
        layers: list[nn.Module] = []
        for _ in range(max(1, int(depth))):
            layers.extend((nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, z_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        if action.shape[-1] != self.action_dim:
            if self.action_dim == 0 and action.numel() == 0:
                action = torch.zeros(z_tau.shape[0], 1, device=z_tau.device, dtype=z_tau.dtype)
            else:
                raise ValueError(f"expected action dim {self.action_dim}, got {action.shape[-1]}")
        if action.shape[-1] == 0:
            action = torch.zeros(z_tau.shape[0], 1, device=z_tau.device, dtype=z_tau.dtype)
        hidden = self.z_proj(self.z_norm(z_tau)) + self.tau_mlp(tau.to(dtype=z_tau.dtype))
        if context is not None:
            hidden = hidden + self.context_proj(context)
        action_params = self.action_mlp(action.to(dtype=z_tau.dtype))
        if self.use_film:
            scale, shift = action_params.chunk(2, dim=-1)
            hidden = hidden * (1.0 + torch.tanh(scale)) + shift
        else:
            hidden = hidden + action_params
        hidden = self.layers(hidden)
        return self.out(hidden)


class LowRankActionKoopman(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, *, rank: int) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.rank = max(1, int(rank))
        self.v = nn.Parameter(torch.empty(z_dim, self.rank))
        self.u = nn.Parameter(torch.empty(z_dim, self.rank))
        self.action_to_rank_gates = nn.Sequential(
            nn.LayerNorm(max(1, action_dim)),
            nn.Linear(max(1, action_dim), self.rank),
        )
        self.action_to_bias = nn.Sequential(
            nn.LayerNorm(max(1, action_dim)),
            nn.Linear(max(1, action_dim), z_dim),
        )
        nn.init.normal_(self.v, std=1.0 / max(1, z_dim) ** 0.5)
        nn.init.zeros_(self.u)
        nn.init.zeros_(self.action_to_rank_gates[-1].weight)
        nn.init.zeros_(self.action_to_rank_gates[-1].bias)
        nn.init.zeros_(self.action_to_bias[-1].weight)
        nn.init.zeros_(self.action_to_bias[-1].bias)

    def forward(self, z_source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] == 0:
            action = torch.zeros(z_source.shape[0], 1, device=z_source.device, dtype=z_source.dtype)
        gates = self.action_to_rank_gates(action)
        left = z_source @ self.v
        bilinear_delta = (left * gates) @ self.u.T
        return bilinear_delta + self.action_to_bias(action)


class FlowIntegrator(nn.Module):
    def __init__(
        self,
        vector_field: ActionConditionedVectorField,
        *,
        steps: int = 4,
        use_tangent_projection: bool = True,
        normalize_endpoint: bool = True,
    ) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.steps = max(1, int(steps))
        self.use_tangent_projection = bool(use_tangent_projection)
        self.normalize_endpoint = bool(normalize_endpoint)

    def forward(self, z0: torch.Tensor, action: torch.Tensor, *, context: torch.Tensor | None = None) -> torch.Tensor:
        z = z0
        for step in range(self.steps):
            tau = torch.full((z.shape[0],), step / self.steps, device=z.device, dtype=z.dtype)
            velocity = self.vector_field(z, tau, action, context=context)
            if self.use_tangent_projection:
                velocity = tangent_project(velocity, z)
            z = z + velocity / self.steps
            if self.normalize_endpoint:
                z = F.normalize(z, dim=-1)
        return z


def tangent_project(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return v - (v * z).sum(dim=-1, keepdim=True) * z


class BioFlowJEPA(nn.Module):
    """BioFlow-JEPA wrapper around BioTech encoders plus a controlled vector field."""

    def __init__(self, config: BioFlowJEPAConfig) -> None:
        super().__init__()
        if config.transition_mode not in {"vector_field", "low_rank_koopman", "hybrid"}:
            raise ValueError("transition_mode must be vector_field, low_rank_koopman, or hybrid")
        self.config = config
        self.base = BioTechJEPA(config.base_biotech_config)
        z_dim = config.base_biotech_config.bio_dim
        hidden = config.base_biotech_config.predictor_dim
        self.delta_whitening = DeltaWhitening(z_dim, rank=config.delta_whitening_rank)
        self.vector_field = ActionConditionedVectorField(
            z_dim,
            config.action_dim,
            hidden_dim=hidden,
            depth=max(1, config.base_biotech_config.predictor_depth + 1),
            dropout=config.base_biotech_config.dropout,
            use_film=config.use_action_conditioned_film,
        )
        self.integrator = FlowIntegrator(
            self.vector_field,
            steps=config.flow_steps,
            use_tangent_projection=config.use_tangent_projection,
            normalize_endpoint=True,
        )
        self.low_rank_operator = LowRankActionKoopman(
            z_dim,
            config.action_dim,
            rank=config.low_rank_operator_rank,
        )

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def trainable_transition_parameters(self):
        for module in (self.vector_field, self.low_rank_operator):
            yield from module.parameters()

    @torch.no_grad()
    def update_teachers(self, *, decay: float | None = None) -> None:
        self.base.update_teachers(decay=decay)

    @torch.no_grad()
    def fit_delta_whitening_from_batches(self, batches, *, device: torch.device | str = "cpu") -> None:
        deltas = []
        was_training = self.training
        self.eval()
        for batch in batches:
            batch = batch.to_device(device)
            source = self.base.encode_condition(
                gene_ids=batch.control_gene_ids,
                expression_values=batch.control_expression_values,
                images=batch.control_images,
                rna_bag_mask=batch.control_rna_bag_mask,
                image_bag_mask=batch.control_image_bag_mask,
                mode="target",
            )
            target = self.base.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="target",
            )
            deltas.append((target["joint_z_bio"] - source["joint_z_bio"]).detach())
        if deltas:
            self.delta_whitening.fit(torch.cat(deltas, dim=0))
        self.train(was_training)

    def forward_bioflow(self, batch: BioActionConditionBatch) -> dict[str, torch.Tensor]:
        action = self._action_descriptor(batch)
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
        source_online = control_context["joint_z_bio"]
        source_teacher = control_teacher["joint_z_bio"].detach()
        target_z = target_teacher["joint_z_bio"].detach()
        true_delta = target_z - source_teacher
        tau = torch.rand(source_online.shape[0], device=source_online.device, dtype=source_online.dtype)
        z_tau = (1.0 - tau[:, None]) * source_teacher + tau[:, None] * target_z
        velocity_pred = self.vector_field(z_tau, tau, action, context=source_online)
        if self.config.use_tangent_projection:
            velocity_pred = tangent_project(velocity_pred, z_tau)
        z_pred = self._transition(source_online, action)
        pred_delta = z_pred - source_online
        wrong_z = None
        if source_online.shape[0] > 1:
            wrong_action = action.roll(shifts=1, dims=0)
            wrong_z = self._transition(source_online, wrong_action)
        outputs: dict[str, torch.Tensor] = {
            "source_z_bio_online": source_online,
            "source_z_bio_teacher": source_teacher,
            "target_z_bio_teacher": target_z,
            "source_z_tech_teacher": control_teacher["joint_z_tech"].detach(),
            "target_z_tech_teacher": target_teacher["joint_z_tech"].detach(),
            "true_delta": true_delta.detach(),
            "velocity_pred": velocity_pred,
            "velocity_target": true_delta.detach(),
            "z_pred": z_pred,
            "pred_delta": pred_delta,
            "action_descriptor": action,
            "condition_key_exact_feature_present": torch.zeros((), device=source_online.device),
            "biological_key_onehot_present": torch.zeros((), device=source_online.device),
            "pls_raw_linear_used_as_main_path": torch.zeros((), device=source_online.device),
            "teacher_stop_gradient_verified": torch.ones((), device=source_online.device),
            "ema_target_present": torch.ones((), device=source_online.device),
            "transition_target_stop_gradient": torch.ones((), device=source_online.device),
            "separate_bio_and_tech_latents_present": torch.ones((), device=source_online.device),
            "z_bio_used_for_transition": torch.ones((), device=source_online.device),
            "z_tech_used_as_transition_shortcut": torch.zeros((), device=source_online.device),
        }
        if self.config.use_delta_whitening:
            outputs["velocity_pred_whitened"] = self.delta_whitening.whiten(velocity_pred)
            outputs["velocity_target_whitened"] = self.delta_whitening.whiten(true_delta.detach())
        if wrong_z is not None:
            outputs["wrong_action_z_pred"] = wrong_z
        for prefix, encoded in (("target", target_teacher), ("source", control_teacher)):
            for key in ("rna_z_bio", "image_z_bio", "joint_z_bio", "joint_z_tech"):
                if key in encoded:
                    outputs[f"{prefix}_{key}"] = encoded[key].detach()
        return outputs

    def _transition(self, source_online: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.config.transition_mode == "low_rank_koopman":
            return F.normalize(source_online + self.low_rank_operator(source_online, action), dim=-1)
        z_flow = self.integrator(source_online, action, context=source_online)
        if self.config.transition_mode == "hybrid":
            z_flow = F.normalize(z_flow + self.low_rank_operator(source_online, action), dim=-1)
        return z_flow

    def _action_descriptor(self, batch: BioActionConditionBatch) -> torch.Tensor:
        batch_size = int(batch.perturbation_id.shape[0])
        if self.config.action_dim == 0:
            return torch.zeros(batch_size, 0, device=batch.perturbation_id.device, dtype=batch.dose.dtype)
        if batch.descriptor is None:
            raise ValueError("BioFlow-JEPA requires an explicit non-leaky action descriptor")
        if batch.descriptor.shape[-1] != self.config.action_dim:
            raise ValueError(f"expected action descriptor dim {self.config.action_dim}, got {batch.descriptor.shape[-1]}")
        return batch.descriptor.to(dtype=batch.dose.dtype)
