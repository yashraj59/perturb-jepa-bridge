from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_biotech_batch_audit_writes_required_stage1_files(tmp_path: Path) -> None:
    output_root = tmp_path / "audit"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_biotech_batch_audit.py",
            "--datasets",
            "synth_micro",
            "--eval-splits",
            "test",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    for name in (
        "INVENTORY.md",
        "METHODS.md",
        "SPLIT_AND_CONFOUNDING_AUDIT.md",
        "RAW_SIGNAL_BATCH_AUDIT.md",
        "TEACHER_TARGET_AUDIT.md",
        "REPRESENTATION_AUDIT.md",
        "REOPENING_DECISION.md",
    ):
        assert (output_root / name).exists()
    decision = json.loads((output_root / "reopening_decision.json").read_text(encoding="utf-8"))
    assert decision["decision_label"] in {"PHASE2_AUDIT_COMPLETE_REOPEN", "PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN"}
    assert (output_root / "raw_signal_batch_probe.tsv").exists()
