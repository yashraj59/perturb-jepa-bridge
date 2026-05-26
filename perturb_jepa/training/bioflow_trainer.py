from __future__ import annotations

from dataclasses import dataclass

import torch

from perturb_jepa.models.bioflow_jepa import BioFlowJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.bioflow_losses import BioFlowJEPALossWeights, bioflow_jepa_loss


@dataclass
class BioFlowTrainStep:
    loss: float
    diagnostics: dict[str, float]


class BioFlowJEPATrainer:
    def __init__(
        self,
        model: BioFlowJEPA,
        optimizer: torch.optim.Optimizer,
        *,
        weights: BioFlowJEPALossWeights | None = None,
        device: torch.device | str = "cpu",
        grad_clip_norm: float | None = 1.0,
        update_teachers: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.weights = weights or BioFlowJEPALossWeights()
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.update_teachers = bool(update_teachers)
        self.step = 0

    def train_step(self, batch: BioActionConditionBatch) -> BioFlowTrainStep:
        self.model.train()
        batch = batch.to_device(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model.forward_bioflow(batch)
        loss, diagnostics = bioflow_jepa_loss(outputs, self.weights)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                float(self.grad_clip_norm),
            )
        self.optimizer.step()
        if self.update_teachers:
            self.model.update_teachers()
        self.step += 1
        payload = {key: float(value.detach().cpu()) for key, value in diagnostics.items() if torch.is_tensor(value)}
        payload["step"] = float(self.step)
        payload["condition_key_exact_feature_present"] = float(outputs["condition_key_exact_feature_present"].detach().cpu())
        payload["biological_key_onehot_present"] = float(outputs["biological_key_onehot_present"].detach().cpu())
        payload["pls_raw_linear_used_as_main_path"] = float(outputs["pls_raw_linear_used_as_main_path"].detach().cpu())
        payload["teacher_stop_gradient_verified"] = float(outputs["teacher_stop_gradient_verified"].detach().cpu())
        payload["transition_target_stop_gradient"] = float(outputs["transition_target_stop_gradient"].detach().cpu())
        return BioFlowTrainStep(loss=float(loss.detach().cpu()), diagnostics=payload)
