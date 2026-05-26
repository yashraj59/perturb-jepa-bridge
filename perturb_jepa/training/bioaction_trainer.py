from __future__ import annotations

from dataclasses import dataclass

import torch

from perturb_jepa.models.bioaction_jepa import BioActionJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.bioaction_losses import BioActionJEPALossWeights, bioaction_jepa_loss, collapse_diagnostics


@dataclass
class BioActionTrainStep:
    loss: float
    diagnostics: dict[str, float]


class BioActionJEPATrainer:
    def __init__(
        self,
        model: BioActionJEPA,
        optimizer: torch.optim.Optimizer,
        *,
        weights: BioActionJEPALossWeights | None = None,
        device: torch.device | str = "cpu",
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.weights = weights or BioActionJEPALossWeights()
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.step = 0

    def train_step(self, batch: BioActionConditionBatch) -> BioActionTrainStep:
        self.model.train()
        batch = batch.to_device(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model.forward_jepa(batch)
        loss, diagnostics = bioaction_jepa_loss(outputs, self.weights)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))
        self.optimizer.step()
        self.model.update_teachers()
        self.step += 1
        payload = {key: float(value.detach().cpu()) for key, value in diagnostics.items() if torch.is_tensor(value)}
        payload.update(collapse_diagnostics(outputs))
        payload["step"] = float(self.step)
        payload["teacher/ema_decay"] = float(self.model.config.ema_decay)
        payload["condition_key_exact_feature_present"] = float(outputs["condition_key_exact_feature_present"].detach().cpu())
        payload["pls_raw_linear_used_as_main_path"] = float(outputs["pls_raw_linear_used_as_main_path"].detach().cpu())
        return BioActionTrainStep(loss=float(loss.detach().cpu()), diagnostics=payload)

