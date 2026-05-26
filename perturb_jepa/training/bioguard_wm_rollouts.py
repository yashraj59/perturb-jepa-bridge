from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BiologicalRolloutExample:
    source_key: str
    action_1: Any
    action_2: Any | None
    source_z: torch.Tensor
    target_1_z: torch.Tensor
    target_2_z: torch.Tensor | None
    action_1_features: torch.Tensor
    action_2_features: torch.Tensor | None
    metadata: dict


def build_single_step_examples(records: list[dict[str, Any]]) -> list[BiologicalRolloutExample]:
    examples = []
    for record in records:
        examples.append(
            BiologicalRolloutExample(
                source_key=str(record.get("source_key", "")),
                action_1=record.get("action_1"),
                action_2=None,
                source_z=record["source_z"],
                target_1_z=record["target_1_z"],
                target_2_z=None,
                action_1_features=record["action_1_features"],
                action_2_features=None,
                metadata=dict(record.get("metadata", {})),
            )
        )
    return examples


def build_two_step_genetic_examples(records: list[dict[str, Any]]) -> list[BiologicalRolloutExample]:
    return [example for example in build_single_step_examples(records) if example.action_2 is not None and example.target_2_z is not None]


def build_two_step_time_or_dose_examples(records: list[dict[str, Any]]) -> list[BiologicalRolloutExample]:
    return [example for example in build_single_step_examples(records) if example.metadata.get("ordered_time_or_dose", False)]


def validate_rollout_examples_no_leakage(examples: list[BiologicalRolloutExample]) -> None:
    forbidden = {"condition_key", "biological_key", "exact_target_key", "eval_target_mean"}
    for example in examples:
        bad = forbidden.intersection(example.metadata)
        if bad:
            raise ValueError(f"rollout leakage metadata present: {sorted(bad)}")


def validate_multistep_targets(step_predictions: list[torch.Tensor], step_targets: list[torch.Tensor]) -> None:
    if len(step_predictions) != len(step_targets):
        raise ValueError("rollout prediction/target length mismatch")
    for index, (prediction, target) in enumerate(zip(step_predictions, step_targets)):
        if prediction.shape != target.shape:
            raise ValueError(f"rollout step {index + 1} target shape mismatch")
