from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Phase7Status:
    decision: str
    no_residual_passed: bool
    later_experiments_not_run: bool
    floor_values_match: bool


def parse_phase7_status(root: Path = Path("outputs/autoresearch_bioguard_jepa_phase7")) -> Phase7Status:
    decision_path = root / "REOPENING_DECISION.md"
    final_path = root / "final_report.md"
    results_path = root / "results.tsv"
    if not decision_path.exists() or not final_path.exists() or not results_path.exists():
        raise FileNotFoundError("Phase 7 status artifacts are missing")
    decision = decision_path.read_text(encoding="utf-8").splitlines()[0].strip()
    final = final_path.read_text(encoding="utf-8")
    results = pd.read_csv(results_path, sep="\t")
    floor_row = results[results["experiment_id"].astype(str) == "BSG000"].iloc[0]
    floor_values_match = (
        abs(float(floor_row["transition_improvement"]) - 0.0057) <= 1.0e-4
        and abs(float(floor_row["delta_cosine"]) - 0.3980) <= 1.0e-4
        and abs(float(floor_row["recall_at_1"]) - 0.4815) <= 1.0e-4
        and abs(float(floor_row["delta_rank"]) - 10.2835) <= 1.0e-3
    )
    return Phase7Status(
        decision=decision,
        no_residual_passed="No residual candidate passed" in final or "no residual passed" in final.lower(),
        later_experiments_not_run="BSG005-BSG008: not run" in final,
        floor_values_match=floor_values_match,
    )
