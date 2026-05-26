from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActionGroupedResidualSplitConfig:
    n_folds: int = 4
    seed: int = 0
    action_col: str = "perturbation_id"
    split_col: str = "split"
    train_split_value: str = "train"
    stratify_cols: tuple[str, ...] = ("cell_line_id", "batch_id")
    leave_action_out: bool = True


@dataclass(frozen=True)
class ResidualFold:
    fold_id: int
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    fit_actions: tuple[str, ...]
    calibration_actions: tuple[str, ...]
    fallback_reason: str
    stratification_counts: dict[str, dict[str, int]]

    def report_row(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "fit_rows": int(self.fit_indices.size),
            "calibration_rows": int(self.calibration_indices.size),
            "fit_actions": ",".join(self.fit_actions),
            "calibration_actions": ",".join(self.calibration_actions),
            "fallback_reason": self.fallback_reason,
            "stratification_counts": self.stratification_counts,
        }


class ActionGroupedResidualSplitter:
    """Train-only grouped folds for residual fitting/calibration."""

    def __init__(self, config: ActionGroupedResidualSplitConfig | None = None) -> None:
        self.config = config or ActionGroupedResidualSplitConfig()

    def split(self, metadata: pd.DataFrame) -> list[ResidualFold]:
        frame = metadata.reset_index(drop=True).copy()
        if self.config.split_col in frame.columns:
            non_train = frame[self.config.split_col].astype(str) != self.config.train_split_value
            if bool(non_train.any()):
                raise ValueError("ActionGroupedResidualSplitter received non-train rows")
        if self.config.action_col not in frame.columns:
            raise ValueError(f"missing action column: {self.config.action_col}")
        action_values = frame[self.config.action_col].astype(str).to_numpy()
        actions = np.unique(action_values)
        if actions.size < 2:
            raise ValueError("at least two action groups are required; refusing random-row fallback")
        n_folds = min(max(2, int(self.config.n_folds)), int(actions.size))
        fallback_reason = "" if n_folds == int(self.config.n_folds) else f"requested_{self.config.n_folds}_folds_but_only_{actions.size}_actions"
        rng = np.random.default_rng(self.config.seed)
        shuffled = np.array(actions, dtype=object)
        rng.shuffle(shuffled)
        action_groups = [tuple(str(item) for item in shuffled[index::n_folds]) for index in range(n_folds)]
        folds: list[ResidualFold] = []
        all_indices = np.arange(len(frame))
        for fold_id, calibration_actions in enumerate(action_groups):
            cal_mask = np.isin(action_values, np.asarray(calibration_actions, dtype=str))
            fit_indices = all_indices[~cal_mask]
            calibration_indices = all_indices[cal_mask]
            if fit_indices.size == 0 or calibration_indices.size == 0:
                raise ValueError("grouped split produced empty fit or calibration partition")
            fit_actions = tuple(sorted(np.unique(action_values[fit_indices]).astype(str).tolist()))
            cal_actions = tuple(sorted(np.unique(action_values[calibration_indices]).astype(str).tolist()))
            if self.config.leave_action_out and set(fit_actions).intersection(cal_actions):
                raise ValueError("action leakage detected between fit and calibration partitions")
            folds.append(
                ResidualFold(
                    fold_id=fold_id,
                    fit_indices=fit_indices,
                    calibration_indices=calibration_indices,
                    fit_actions=fit_actions,
                    calibration_actions=cal_actions,
                    fallback_reason=fallback_reason,
                    stratification_counts=self._stratification_counts(frame, fit_indices, calibration_indices),
                )
            )
        return folds

    def report(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([fold.report_row() for fold in self.split(metadata)])

    def _stratification_counts(self, frame: pd.DataFrame, fit_indices: Sequence[int], calibration_indices: Sequence[int]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for column in self.config.stratify_cols:
            if column not in frame.columns:
                continue
            fit = frame.iloc[list(fit_indices)][column].astype(str).value_counts().to_dict()
            cal = frame.iloc[list(calibration_indices)][column].astype(str).value_counts().to_dict()
            counts[column] = {f"fit/{key}": int(value) for key, value in fit.items()}
            counts[column].update({f"calibration/{key}": int(value) for key, value in cal.items()})
        return counts
