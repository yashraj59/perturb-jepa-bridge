from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from perturb_jepa.models.bioaction_jepa import TargetQueryPredictor
from perturb_jepa.models.biotech_jepa import BioTechJEPA, BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch


@dataclass(frozen=True)
class BioMechanisticJEPAConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 64
    bio_dim: int = 48
    tech_dim: int = 16
    predictor_dim: int = 128
    target_query_dim: int = 48
    predictor_depth: int = 1
    predictor_heads: int = 4
    num_condition_prototypes: int = 4
    num_rna_program_targets: int = 8
    num_image_region_targets: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996
    count_decoder_aux: bool = False
    gene_program_assignment: tuple[int, ...] = ()
    enable_delta_jepa: bool = True
    enable_program_action_encoder: bool = False
    enable_population_transition: bool = False
    enable_cross_modal_repair: bool = False
    descriptor_gene_dim: int = 0
    descriptor_program_dim: int = 0
    action_dim: int = 48
    disable_perturbation_id_embedding_for_heldout_generalization: bool = False
    forbid_condition_key_features: bool = True


class ProgramActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        gene_dim: int,
        program_dim: int,
        num_types: int,
        num_cell_lines: int,
        action_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gene_dim = int(gene_dim)
        self.program_dim = int(program_dim)
        self.action_dim = int(action_dim)
        self.gene_embedding = nn.Parameter(torch.empty(max(1, self.gene_dim), action_dim))
        self.program_embedding = nn.Parameter(torch.empty(max(1, self.program_dim), action_dim))
        self.type_embedding = nn.Embedding(num_types, action_dim)
        self.cell_embedding = nn.Embedding(num_cell_lines, action_dim)
        self.numeric_mlp = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, action_dim), nn.GELU(), nn.Linear(action_dim, action_dim))
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(action_dim * 2 + 1),
            nn.Linear(action_dim * 2 + 1, action_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_dim, action_dim),
        )
        self.global_fuser = nn.Sequential(
            nn.LayerNorm(action_dim * 5),
            nn.Linear(action_dim * 5, action_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_dim, action_dim),
        )
        self.program_head = nn.Linear(action_dim, max(1, self.program_dim))
        nn.init.trunc_normal_(self.gene_embedding, std=0.02)
        nn.init.trunc_normal_(self.program_embedding, std=0.02)

    def forward(
        self,
        descriptor: torch.Tensor,
        perturbation_type_id: torch.Tensor,
        cell_line_id: torch.Tensor,
        dose: torch.Tensor,
        time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if descriptor.ndim != 2:
            raise ValueError("descriptor must have shape [batch, descriptor_dim]")
        batch = descriptor.shape[0]
        device = descriptor.device
        dtype = descriptor.dtype
        if self.gene_dim > 0:
            gene_values = descriptor[:, : self.gene_dim]
            gene_token = gene_values @ self.gene_embedding.to(dtype=dtype, device=device)
            gene_count = gene_values.sum(dim=-1, keepdim=True)
        else:
            gene_values = torch.zeros(batch, 0, device=device, dtype=dtype)
            gene_token = torch.zeros(batch, self.action_dim, device=device, dtype=dtype)
            gene_count = torch.zeros(batch, 1, device=device, dtype=dtype)
        if self.program_dim > 0:
            program_values = descriptor[:, self.gene_dim : self.gene_dim + self.program_dim]
            program_token = program_values @ self.program_embedding.to(dtype=dtype, device=device)
        else:
            program_token = torch.zeros(batch, self.action_dim, device=device, dtype=dtype)
        type_token = self.type_embedding(perturbation_type_id)
        cell_token = self.cell_embedding(cell_line_id)
        numeric_token = self.numeric_mlp(torch.stack((dose, time), dim=-1))
        pair_indicator = (gene_count > 1.0).to(dtype=dtype)
        interaction_token = self.interaction_mlp(torch.cat((gene_token, gene_token * pair_indicator, gene_count.clamp_max(2.0)), dim=-1))
        action_global = self.global_fuser(torch.cat((gene_token, program_token, interaction_token, type_token, cell_token + numeric_token), dim=-1))
        action_global = F.normalize(action_global, dim=-1)
        action_tokens = torch.stack(
            (
                F.normalize(gene_token, dim=-1),
                F.normalize(program_token, dim=-1),
                F.normalize(interaction_token, dim=-1),
                action_global,
            ),
            dim=1,
        )
        return {
            "action_global": action_global,
            "action_tokens": action_tokens,
            "action_program_logits": self.program_head(action_global),
            "action_interaction_token": interaction_token,
        }


class PopulationTransitionPredictor(nn.Module):
    def __init__(
        self,
        *,
        bio_dim: int,
        query_dim: int,
        predictor_dim: int,
        depth: int,
        heads: int,
        dropout: float,
        num_prototypes: int,
    ) -> None:
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.query_embeddings = nn.Embedding(max(1, num_prototypes), query_dim)
        self.predictor = TargetQueryPredictor(
            shared_dim=bio_dim,
            query_dim=query_dim,
            predictor_dim=predictor_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
        )

    def forward(self, source_prototypes: torch.Tensor, action_global: torch.Tensor) -> torch.Tensor:
        batch = source_prototypes.shape[0]
        ids = torch.arange(self.num_prototypes, device=source_prototypes.device)
        queries = self.query_embeddings(ids)[None, :, :].expand(batch, -1, -1)
        delta = self.predictor(source_prototypes, queries, action_tokens=action_global)
        return F.normalize(source_prototypes[:, : self.num_prototypes, :] + 0.25 * delta, dim=-1)


class BioMechanisticJEPA(nn.Module):
    def __init__(self, config: BioMechanisticJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = BioTechJEPA(
            BioTechJEPAConfig(
                rna=config.rna,
                image=config.image,
                perturbation=config.perturbation,
                shared_dim=config.shared_dim,
                bio_dim=config.bio_dim,
                tech_dim=config.tech_dim,
                predictor_dim=config.predictor_dim,
                target_query_dim=config.target_query_dim,
                predictor_depth=config.predictor_depth,
                predictor_heads=config.predictor_heads,
                num_condition_prototypes=config.num_condition_prototypes,
                num_rna_program_targets=config.num_rna_program_targets,
                num_image_region_targets=config.num_image_region_targets,
                dropout=config.dropout,
                ema_decay=config.ema_decay,
                count_decoder_aux=config.count_decoder_aux,
                gene_program_assignment=config.gene_program_assignment,
            )
        )
        self.program_action_encoder = (
            ProgramActionEncoder(
                gene_dim=config.descriptor_gene_dim,
                program_dim=config.descriptor_program_dim,
                num_types=config.perturbation.num_types,
                num_cell_lines=config.perturbation.num_cell_lines,
                action_dim=config.bio_dim,
                dropout=config.dropout,
            )
            if config.enable_program_action_encoder
            else None
        )
        self.delta_query = nn.Embedding(1, config.target_query_dim)
        self.delta_predictor = TargetQueryPredictor(
            shared_dim=config.bio_dim,
            query_dim=config.target_query_dim,
            predictor_dim=config.predictor_dim,
            depth=config.predictor_depth,
            heads=config.predictor_heads,
            dropout=config.dropout,
        )
        self.population_transition = (
            PopulationTransitionPredictor(
                bio_dim=config.bio_dim,
                query_dim=config.target_query_dim,
                predictor_dim=config.predictor_dim,
                depth=config.predictor_depth,
                heads=config.predictor_heads,
                dropout=config.dropout,
                num_prototypes=config.num_condition_prototypes,
            )
            if config.enable_population_transition
            else None
        )
        self.shared_target_normalizer = nn.LayerNorm(config.bio_dim, elementwise_affine=False)

    @torch.no_grad()
    def update_teachers(self, *, decay: float | None = None) -> None:
        self.backbone.update_teachers(decay=decay)

    def encode_condition(self, **kwargs) -> dict[str, torch.Tensor]:
        return self.backbone.encode_condition(**kwargs)

    def forward_jepa(self, batch: BioActionConditionBatch) -> dict[str, torch.Tensor]:
        base = self.backbone.forward_jepa(batch)
        control_teacher = self.backbone.encode_condition(
            gene_ids=batch.control_gene_ids,
            expression_values=batch.control_expression_values,
            images=batch.control_images,
            rna_bag_mask=batch.control_rna_bag_mask,
            image_bag_mask=batch.control_image_bag_mask,
            mode="target",
        )
        target_teacher = self.backbone.encode_condition(
            gene_ids=batch.target_gene_ids,
            expression_values=batch.target_expression_values,
            images=batch.target_images,
            rna_bag_mask=batch.target_rna_bag_mask,
            image_bag_mask=batch.target_image_bag_mask,
            mode="target",
        )
        action = self._action(batch, base)
        z_control_online = base["joint_z_bio"]
        z_control_teacher = self._norm(control_teacher["joint_z_bio"]).detach()
        z_target_teacher = self._norm(target_teacher["joint_z_bio"]).detach()
        delta_teacher = (z_target_teacher - z_control_teacher).detach()
        context_tokens = torch.cat((z_control_online[:, None, :], action["action_tokens"]), dim=1)
        query = self.delta_query.weight[None, :, :].expand(z_control_online.shape[0], -1, -1)
        delta_pred = self.delta_predictor(context_tokens, query, action_tokens=action["action_global"]).squeeze(1)
        control_mask = batch.perturbation_id.eq(0).to(dtype=delta_pred.dtype, device=delta_pred.device)[:, None]
        delta_pred = delta_pred * (1.0 - control_mask)
        z_target_pred = self._norm(z_control_online + delta_pred)

        base.update(
            {
                "action_global": action["action_global"],
                "action_tokens": action["action_tokens"],
                "action_program_logits": action["action_program_logits"],
                "action_interaction_token": action["action_interaction_token"],
                "heldout_action_descriptor_valid": action["descriptor_valid"],
                "perturbation_id_for_loss": batch.perturbation_id.detach(),
                "z_control_teacher_bio": z_control_teacher,
                "z_target_teacher_bio": z_target_teacher,
                "delta_teacher": delta_teacher,
                "delta_pred": delta_pred,
                "z_target_pred": z_target_pred,
                "delta_jepa_pred": delta_pred[:, None, :],
                "delta_jepa_target": delta_teacher[:, None, :],
                "delta_jepa_available": torch.ones((), dtype=torch.bool, device=delta_pred.device),
                "target_transition_jepa_pred": z_target_pred[:, None, :],
                "target_transition_jepa_target": z_target_teacher[:, None, :],
                "target_transition_jepa_available": torch.ones((), dtype=torch.bool, device=delta_pred.device),
                "delta_teacher_effective_rank": _effective_rank(delta_teacher),
                "delta_pred_effective_rank": _effective_rank(delta_pred),
            }
        )
        self._add_population_outputs(base, control_teacher, target_teacher, action)
        return base

    def _action(self, batch: BioActionConditionBatch, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.program_action_encoder is not None:
            if batch.descriptor is None:
                raise ValueError("program action encoder requires non-leaky action descriptors")
            encoded = self.program_action_encoder(
                batch.descriptor,
                batch.perturbation_type_id,
                batch.cell_line_id,
                batch.dose,
                batch.time,
            )
            descriptor_valid = batch.descriptor.detach().abs().sum(dim=-1).gt(0.0) | batch.perturbation_id.eq(0)
            encoded["descriptor_valid"] = descriptor_valid.float().mean()
            return encoded
        action_global = base["action_embedding"]
        return {
            "action_global": action_global,
            "action_tokens": action_global[:, None, :],
            "action_program_logits": torch.zeros(action_global.shape[0], 1, device=action_global.device, dtype=action_global.dtype),
            "action_interaction_token": action_global,
            "descriptor_valid": torch.ones((), device=action_global.device),
        }

    def _add_population_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        control_teacher: dict[str, torch.Tensor],
        target_teacher: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
    ) -> None:
        source = _prototype_tokens(control_teacher)
        target = _prototype_tokens(target_teacher)
        count = min(source.shape[1], target.shape[1], self.config.num_condition_prototypes)
        source = source[:, :count, :]
        target = target[:, :count, :].detach()
        outputs["source_bio_prototypes"] = source.detach()
        outputs["teacher_target_bio_prototypes"] = target
        if self.population_transition is None:
            outputs["predicted_target_bio_prototypes"] = source.detach()
            outputs["prototype_transition_jepa_available"] = torch.zeros((), dtype=torch.bool, device=source.device)
            return
        pred = self.population_transition(source, action["action_global"])
        outputs["predicted_target_bio_prototypes"] = pred
        outputs["prototype_transition_jepa_pred"] = pred
        outputs["prototype_transition_jepa_target"] = target
        outputs["prototype_transition_jepa_available"] = torch.ones((), dtype=torch.bool, device=source.device)

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.shared_target_normalizer(value), dim=-1)


def _prototype_tokens(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    tokens = []
    if "rna_condition_bio_prototypes" in encoded:
        tokens.append(encoded["rna_condition_bio_prototypes"])
    if "image_condition_bio_prototypes" in encoded:
        tokens.append(encoded["image_condition_bio_prototypes"])
    if not tokens:
        return encoded["joint_z_bio"][:, None, :]
    return torch.cat(tokens, dim=1)


def _effective_rank(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    centered = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered)
    probs = spectrum / spectrum.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return torch.exp(entropy)
