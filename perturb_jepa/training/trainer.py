from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from perturb_jepa.losses import BridgeLossWeights, bridge_loss
from perturb_jepa.models.bridge import PerturbJEPABridge
from perturb_jepa.models.image_encoder import patchify
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


def move_batch_to_device(batch: SyntheticBridgeBatch, device: torch.device | str) -> SyntheticBridgeBatch:
    device = torch.device(device)
    return SyntheticBridgeBatch(**{field.name: getattr(batch, field.name).to(device) for field in fields(batch)})


def loss_for_batch(
    model: PerturbJEPABridge,
    batch: SyntheticBridgeBatch,
    *,
    weights: BridgeLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = forward_batch(model, batch)
    image_patches = patchify(batch.images, model.config.image.patch_size)
    return bridge_loss(
        outputs,
        rna_values=batch.expression_values,
        rna_mask=batch.rna_token_mask,
        image_patches=image_patches,
        image_patch_mask=batch.image_patch_mask,
        perturbation_id=batch.perturbation_id,
        batch_id=batch.batch_id,
        weights=weights,
    )


def terms_to_float(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def train_step(
    model: PerturbJEPABridge,
    optimizer: torch.optim.Optimizer,
    batch: SyntheticBridgeBatch,
    *,
    weights: BridgeLossWeights | None = None,
    ema_decay: float = 0.996,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, terms = loss_for_batch(model, batch, weights=weights)
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
        ema_decay: float = 0.996,
        device: torch.device | str | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.weights = weights
        self.ema_decay = float(ema_decay)
        self.device = torch.device(device) if device is not None else None
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        if self.device is not None:
            self.model.to(self.device)

    def step(self, batch: SyntheticBridgeBatch) -> dict[str, float]:
        if self.device is not None:
            batch = move_batch_to_device(batch, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, terms = loss_for_batch(self.model, batch, weights=self.weights)
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
        return {"global_step": self.global_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))

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
