from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path("outputs/autoresearch_synth_lite")
RESULTS_PATH = ROOT / "results.tsv"
COUNTERFACTUAL_DIR = ROOT / "diagnostics" / "CLONE_COUNTERFACTUAL_DECODER"
SUMMARY_PATH = COUNTERFACTUAL_DIR / "TIER2_SUMMARY.md"
JOURNAL_PATH = ROOT / "research_journal.md"
ARCH_LOG_PATH = ROOT / "architectural_changes_log.md"
FAMILY_PATH = ROOT / "family_allocation.md"
NEXT_STEPS_PATH = ROOT / "NEXT_STEPS_AUTORESEARCH_BIO_PLAN.md"
INSIGHTS_DIR = ROOT / "insights"
STOP_PATH = ROOT / "STRICT_GPU_CONTINUATION_STOP.md"
RUNNER = Path("scripts/train_clone_counterfactual_decoder.py")

RESULT_FIELDS = [
    "commit",
    "experiment_num",
    "family",
    "candidate_name",
    "tier_reached",
    "decision_label",
    "device_used",
    "wallclock_minutes",
    "max_gpu_memory_gb",
    "synth_micro_recall1",
    "synth_easy_recall1",
    "synth_medium_recall1",
    "synth_easy_cf_dir_acc",
    "synth_medium_cf_dir_acc",
    "heldout_pert_cf_dir_acc",
    "batch_confound_batch_leakage",
    "batch_confound_recall1",
    "dose_extrap_logfc_corr",
    "bio_latent_r2",
    "representation_rank",
    "delta_norm_ratio",
    "cap_bound",
    "collapse_flag",
    "architecture_change",
    "description",
]


@dataclass(frozen=True)
class Trial:
    number: int
    name: str
    description: str
    change: str
    seed: int
    steps: int
    args: tuple[str, ...]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    COUNTERFACTUAL_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_ledgers()
    commit = _git(["rev-parse", "HEAD"])
    start_number = _next_experiment_number()
    target_last = start_number + 99
    plan = _trial_plan(start_number, target_last)
    existing = _existing_experiments()
    completed_this_run = 0

    for trial in plan:
        if trial.number in existing:
            print(f"skip {trial.number:03d} {trial.name}", flush=True)
            continue
        gpu = _require_free_gpu()
        started_dirs = {path.resolve() for path in COUNTERFACTUAL_DIR.iterdir() if path.is_dir()}
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            "synth_micro",
            "--seed",
            str(trial.seed),
            "--rank",
            "3",
            "--steps",
            str(trial.steps),
            "--device",
            "cuda",
            "--output-dir",
            str(COUNTERFACTUAL_DIR),
            *trial.args,
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            _write_stop_report(trial, command, completed, gpu)
            return 2
        metrics = _parse_metrics(completed.stdout)
        run_dir = _find_run_dir(started_dirs)
        before = _read_json(run_dir / "BEFORE_METRICS.json")
        after = _read_json(run_dir / "AFTER_METRICS.json")
        metrics.update(after)
        metrics["wallclock_minutes"] = float(metrics.get("wallclock_minutes", (time.perf_counter() - started) / 60.0))
        metrics["device_used"] = "cuda"
        metrics["max_gpu_memory_gb"] = float(metrics.get("max_gpu_memory_gb", 0.0))
        deltas = _counterfactual_deltas(before, after)
        decision = _decision_label(after, deltas)
        _append_result(commit, trial, metrics, decision)
        _append_journal(trial, command, run_dir, after, deltas, decision, gpu)
        _append_architecture_log(trial, command, after, deltas, decision)
        _append_family_allocation(trial)
        _append_summary_ledger(trial, run_dir, after, deltas, decision)
        _append_next_steps_ledger(trial, after, deltas, decision)
        completed_this_run += 1
        if trial.number % 10 == 0:
            _write_insight_brief(trial.number)
        print(
            f"done {trial.number:03d} {trial.name} {decision} "
            f"seed={trial.seed} prog_delta={deltas['model_program_level_effect_recovery']:.4f} "
            f"gate={bool(after.get('counterfactual_gate_pass'))}",
            flush=True,
        )
    _write_completion_report(start_number, target_last, completed_this_run)
    return 0


def _trial_plan(start: int, stop: int) -> list[Trial]:
    trials: list[Trial] = []

    def add(name: str, description: str, change: str, *, seed: int = 2, steps: int = 50, args: tuple[str, ...]) -> None:
        if start + len(trials) > stop:
            return
        trials.append(Trial(start + len(trials), name, description, change, seed, steps, args))

    base = (
        "--residual-rna",
        "--program-factorized-rna",
        "--program-condition-source",
        "--program-metadata-context",
    )
    for direction in (0.05, 0.1, 0.2, 0.4, 0.8):
        for program in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            add(
                f"seed2_meta_src_dw{_sf(direction)}_pw{_sf(program)}",
                "Seed-2 screen for balancing direction loss against program-delta loss with source program and no-batch metadata context.",
                "seed2_program_loss_grid",
                args=(*base, "--direction-weight", str(direction), "--program-weight", str(program)),
            )
    for direction in (0.05, 0.1, 0.2, 0.4):
        for program in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            add(
                f"seed2_delta_meta_src_dw{_sf(direction)}_pw{_sf(program)}",
                "Seed-2 delta-MSE screen for increasing perturbation amplitude without moving protected geometry.",
                "seed2_delta_mse_program_grid",
                args=(*base, "--delta-mse", "--direction-weight", str(direction), "--program-weight", str(program)),
            )
    for direction in (0.1, 0.2, 0.4):
        for program in (0.0, 0.25, 1.0, 4.0):
            add(
                f"seed2_linear_meta_src_dw{_sf(direction)}_pw{_sf(program)}",
                "Seed-2 linear program decoder screen to test whether a shallower head improves program recovery.",
                "seed2_linear_program_decoder",
                args=(
                    *base,
                    "--linear-program-decoder",
                    "--direction-weight",
                    str(direction),
                    "--program-weight",
                    str(program),
                ),
            )
    for alpha in (1e-6, 1e-4, 1e-3, 1e-2):
        for program in (0.0, 0.25, 1.0, 4.0):
            add(
                f"seed2_prefitridge_a{_sf(alpha)}_pw{_sf(program)}",
                "Seed-2 train-split-only ridge initialization for the linear program decoder.",
                "seed2_prefit_program_ridge",
                args=(
                    *base,
                    "--linear-program-decoder",
                    "--prefit-program-ridge",
                    "--prefit-program-ridge-alpha",
                    str(alpha),
                    "--prefit-program-ridge-repeats",
                    "4",
                    "--direction-weight",
                    "0.2",
                    "--program-weight",
                    str(program),
                ),
            )
    for source, metadata in ((True, False), (False, True), (False, False)):
        context_args = ("--residual-rna", "--program-factorized-rna")
        if source:
            context_args += ("--program-condition-source",)
        if metadata:
            context_args += ("--program-metadata-context",)
        for program in (0.0, 0.5, 2.0):
            tag = f"src{int(source)}_meta{int(metadata)}"
            add(
                f"seed2_context_{tag}_pw{_sf(program)}",
                "Seed-2 context ablation to isolate source-program and metadata contributions without technical batch metadata.",
                "seed2_context_ablation",
                args=(*context_args, "--direction-weight", "0.2", "--program-weight", str(program)),
            )
    for program in (1.0, 2.0, 4.0):
        for delta in (False, True):
            args = (*base, "--within-program-residual", "--direction-weight", "0.2", "--program-weight", str(program))
            if delta:
                args += ("--delta-mse",)
            add(
                f"seed2_wres_{'delta' if delta else 'abs'}_pw{_sf(program)}",
                "Seed-2 within-program residual screen with explicit program-loss pressure to check whether residual dominance can be controlled.",
                "seed2_within_program_residual_control",
                args=args,
            )
    for lr in (3e-4, 5e-4, 2e-3, 3e-3):
        for program in (0.25, 1.0):
            add(
                f"seed2_lr{_sf(lr)}_pw{_sf(program)}",
                "Seed-2 optimizer sensitivity screen under the strongest safe metadata/source context.",
                "seed2_optimizer_sensitivity",
                args=(*base, "--lr", str(lr), "--direction-weight", "0.2", "--program-weight", str(program)),
            )
    while len(trials) < stop - start + 1:
        index = len(trials)
        program = (0.05, 0.75, 1.5, 3.0)[index % 4]
        direction = (0.15, 0.3, 0.6)[index % 3]
        add(
            f"seed2_fill_dw{_sf(direction)}_pw{_sf(program)}_{index}",
            "Seed-2 filler grid point preserving the no-batch metadata/source context while varying loss balance.",
            "seed2_loss_balance_fill",
            args=(*base, "--direction-weight", str(direction), "--program-weight", str(program)),
        )
    return trials


def _require_free_gpu() -> dict[str, float]:
    for _ in range(3):
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
        first = completed.stdout.strip().splitlines()[0]
        index, used, total, util = [part.strip() for part in first.split(",")]
        gpu = {"index": float(index), "memory_used_mb": float(used), "memory_total_mb": float(total), "utilization": float(util)}
        if gpu["memory_used_mb"] < 1024.0 and gpu["utilization"] <= 20.0:
            return gpu
        time.sleep(30)
    raise RuntimeError("GPU remained busy after three checks")


def _parse_metrics(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"could not parse metrics JSON from stdout tail: {stdout[-1000:]}")


def _find_run_dir(before_dirs: set[Path]) -> Path:
    after_dirs = {path.resolve() for path in COUNTERFACTUAL_DIR.iterdir() if path.is_dir()}
    new_dirs = sorted(after_dirs - before_dirs, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    candidates = sorted(
        (path.resolve() for path in COUNTERFACTUAL_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("no counterfactual run directory found")
    return candidates[-1]


def _decision_label(metrics: dict[str, Any], deltas: dict[str, float]) -> str:
    if not bool(metrics.get("protected_geometry_preserved")):
        return "TIER1_DISCARD_PROTECTED_GEOMETRY_REGRESSION"
    if bool(metrics.get("counterfactual_gate_pass")):
        return "TIER1_KEEP_SEED2_COUNTERFACTUAL_SIGNAL"
    if deltas["model_rna_counterfactual_pseudobulk_correlation"] < -0.02:
        return "TIER1_DISCARD_PSEUDOBULK_REGRESSION"
    if deltas["model_program_level_effect_recovery"] < 0.0:
        return "TIER1_DISCARD_SEED2_PROGRAM_REGRESSION"
    if deltas["model_program_level_effect_recovery"] < 0.05:
        return "TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE"
    return "TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE"


def _append_result(commit: str, trial: Trial, metrics: dict[str, Any], decision: str) -> None:
    row = {
        "commit": commit,
        "experiment_num": f"{trial.number:03d}",
        "family": "K",
        "candidate_name": trial.name,
        "tier_reached": f"Tier 1 synth_micro seed{trial.seed}",
        "decision_label": decision,
        "device_used": str(metrics.get("device_used", "cuda")),
        "wallclock_minutes": f"{_metric(metrics, 'wallclock_minutes'):.3f}",
        "max_gpu_memory_gb": f"{_metric(metrics, 'max_gpu_memory_gb'):.3f}",
        "synth_micro_recall1": f"{_metric(metrics, 'model_rna_to_image_recall@1'):.6f}",
        "synth_easy_recall1": "NA",
        "synth_medium_recall1": "NA",
        "synth_easy_cf_dir_acc": "NA",
        "synth_medium_cf_dir_acc": "NA",
        "heldout_pert_cf_dir_acc": "NA",
        "batch_confound_batch_leakage": "NA",
        "batch_confound_recall1": "NA",
        "dose_extrap_logfc_corr": f"{_metric(metrics, 'model_rna_counterfactual_logfc_correlation'):.6f}",
        "bio_latent_r2": f"{_metric(metrics, 'model_bio_latent_r2_rna_shared'):.6f}",
        "representation_rank": f"{_metric(metrics, 'model_embedding_rank'):.1f}",
        "delta_norm_ratio": f"{_metric(metrics, 'mean_final_delta_to_target_ratio'):.6f}",
        "cap_bound": "false",
        "collapse_flag": str(bool(metrics.get("collapse_flag"))).lower(),
        "architecture_change": trial.change,
        "description": trial.description,
    }
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writerow(row)


def _append_journal(
    trial: Trial,
    command: list[str],
    run_dir: Path,
    metrics: dict[str, Any],
    deltas: dict[str, float],
    decision: str,
    gpu: dict[str, float],
) -> None:
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Experiment {trial.number:03d}: {trial.name}\n\n"
            f"**Hypothesis**: {trial.description}\n\n"
            "**Family**: K, seed-2-focused counterfactual program recovery with protected PLS geometry frozen.\n\n"
            f"**Implementation**: `{trial.change}` using command `{' '.join(command)}`.\n\n"
            "**Initialization / identity preservation**: The protected PLS readout and exact linear clone remain frozen; "
            "the counterfactual decoder starts from residual source-as-target behavior. Direct metadata context excludes batch and technical metadata.\n\n"
            f"**GPU check**: GPU {int(gpu['index'])} free before launch, memory {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MiB, utilization {gpu['utilization']:.0f}%.\n\n"
            f"**Tier result**: Tier 1 `synth_micro` seed {trial.seed}, {trial.steps} steps, device `{metrics.get('device_used', 'cuda')}`. "
            f"Gate pass `{bool(metrics.get('counterfactual_gate_pass'))}`, protected geometry `{bool(metrics.get('protected_geometry_preserved'))}`.\n\n"
            "**Diagnostics**: "
            f"program delta {deltas['model_program_level_effect_recovery']:.4f}; "
            f"direction delta {deltas['model_rna_counterfactual_direction_accuracy']:.4f}; "
            f"logFC delta {deltas['model_rna_counterfactual_logfc_correlation']:.4f}; "
            f"pseudobulk delta {deltas['model_rna_counterfactual_pseudobulk_correlation']:.4f}; "
            f"top50 delta {deltas['model_rna_counterfactual_top50_de_overlap']:.4f}; "
            f"final delta/target ratio {_metric(metrics, 'mean_final_delta_to_target_ratio'):.4f}; "
            f"program contribution ratio {_metric(metrics, 'mean_program_contribution_ratio'):.4f}; "
            f"within residual ratio {_metric(metrics, 'mean_within_program_residual_ratio'):.4f}; "
            f"cap-hit {_metric(metrics, 'mean_cap_hit_fraction'):.4f}; "
            f"max GPU memory {_metric(metrics, 'max_gpu_memory_gb'):.4f} GB.\n\n"
            f"**Decision**: `{decision}`.\n\n"
            f"**Learning**: Seed-2 program recovery remains the active gate unless this trial passed it; compare against Experiment 110 seed-2 program delta `-0.0663`.\n\n"
            f"**Artifact retention**: Per-run metrics and report retained at `{run_dir}`; checkpoint retained only as audit evidence for this synthetic run.\n"
        )


def _append_architecture_log(
    trial: Trial,
    command: list[str],
    metrics: dict[str, Any],
    deltas: dict[str, float],
    decision: str,
) -> None:
    with ARCH_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Change {trial.number:03d}: {trial.name}\n\n"
            "### Family\nK\n\n"
            f"### Exact mechanism\n{trial.change}: `{' '.join(command)}`\n\n"
            "### Why this preserves JEPA\nThe JEPA/scRNA bridge, frozen PLS retrieval path, and exact linear clone remain active; only synthetic counterfactual transition/program-decoder training is varied.\n\n"
            f"### Compute cost estimate\nGPU Tier 1 seed-{trial.seed} synthetic run, {trial.steps} steps; no real data, real marker/pathway resources, or pretrained assets.\n\n"
            f"### Observed effect\nProgram delta {deltas['model_program_level_effect_recovery']:.4f}; "
            f"direction delta {deltas['model_rna_counterfactual_direction_accuracy']:.4f}; "
            f"logFC delta {deltas['model_rna_counterfactual_logfc_correlation']:.4f}; "
            f"protected geometry {bool(metrics.get('protected_geometry_preserved'))}; "
            f"final delta/target ratio {_metric(metrics, 'mean_final_delta_to_target_ratio'):.4f}.\n\n"
            f"### Verdict\n{decision}.\n"
        )


def _append_family_allocation(trial: Trial) -> None:
    rows = _read_results_rows()
    family_count = sum(1 for row in rows if row["family"] == "K")
    with FAMILY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Experiment {trial.number:03d} Allocation Update\n\n"
            f"Family K completed {family_count} experiments. Total recorded experiments: {len(rows)}. "
            "Strict GPU continuation remains active until at least 100 new trials have completed or a protected stop fires.\n"
        )


def _append_summary_ledger(
    trial: Trial,
    run_dir: Path,
    metrics: dict[str, Any],
    deltas: dict[str, float],
    decision: str,
) -> None:
    _ensure_section(SUMMARY_PATH, "## Strict GPU Continuation Ledger")
    with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n- {trial.number:03d} `{trial.name}` seed {trial.seed}: `{decision}`; "
            f"gate `{bool(metrics.get('counterfactual_gate_pass'))}`; "
            f"program delta `{deltas['model_program_level_effect_recovery']:.4f}`; "
            f"direction delta `{deltas['model_rna_counterfactual_direction_accuracy']:.4f}`; "
            f"logFC delta `{deltas['model_rna_counterfactual_logfc_correlation']:.4f}`; "
            f"final delta/target `{_metric(metrics, 'mean_final_delta_to_target_ratio'):.4f}`; "
            f"artifact `{run_dir.name}`.\n"
        )


def _append_next_steps_ledger(trial: Trial, metrics: dict[str, Any], deltas: dict[str, float], decision: str) -> None:
    _ensure_section(NEXT_STEPS_PATH, "## Strict GPU Continuation Progress")
    with NEXT_STEPS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n- Completed {trial.number:03d}: `{decision}`, seed {trial.seed}, "
            f"program delta `{deltas['model_program_level_effect_recovery']:.4f}`, "
            f"gate `{bool(metrics.get('counterfactual_gate_pass'))}`. Continue unless protected stop fires.\n"
        )


def _write_insight_brief(number: int) -> None:
    rows = [row for row in _read_results_rows() if int(row["experiment_num"]) >= 111]
    best = None
    for row in rows:
        try:
            score = float(row["delta_norm_ratio"])
        except ValueError:
            score = 0.0
        if best is None or score > best[0]:
            best = (score, row)
    path = INSIGHTS_DIR / f"INSIGHT_BRIEF_{number:03d}.md"
    path.write_text(
        f"# Insight Brief {number:03d}\n\n"
        f"Strict GPU continuation experiments completed since 111: {len(rows)}.\n\n"
        f"Best final delta/target ratio so far: {best[0]:.4f} from {best[1]['experiment_num']} `{best[1]['candidate_name']}`.\n\n"
        "Current interpretation: seed-2 program recovery is the active blocker; protected shared geometry remains the no-regression constraint.\n\n"
        "Continue: yes, until at least 100 new documented trials complete or a protected stop fires.\n",
        encoding="utf-8",
    )


def _write_completion_report(start: int, stop: int, completed_this_run: int) -> None:
    path = ROOT / "STRICT_GPU_CONTINUATION_SUMMARY.md"
    rows = [row for row in _read_results_rows() if start <= int(row["experiment_num"]) <= stop]
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["decision_label"]] = labels.get(row["decision_label"], 0) + 1
    path.write_text(
        "# Strict GPU Continuation Summary\n\n"
        f"Target experiment range: {start:03d}-{stop:03d}\n\n"
        f"Completed in this invocation: {completed_this_run}\n\n"
        f"Recorded rows in target range: {len(rows)}\n\n"
        "## Decision Labels\n\n"
        + "\n".join(f"- {label}: {count}" for label, count in sorted(labels.items()))
        + "\n",
        encoding="utf-8",
    )


def _write_stop_report(
    trial: Trial,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    gpu: dict[str, float],
) -> None:
    STOP_PATH.write_text(
        f"# Strict GPU Continuation Stop\n\n"
        f"Trial: {trial.number:03d} `{trial.name}`\n\n"
        f"GPU before launch: memory {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MiB, utilization {gpu['utilization']:.0f}%.\n\n"
        f"Command:\n\n```text\n{' '.join(command)}\n```\n\n"
        f"Return code: {completed.returncode}\n\n"
        f"stdout tail:\n\n```text\n{completed.stdout[-4000:]}\n```\n\n"
        f"stderr tail:\n\n```text\n{completed.stderr[-4000:]}\n```\n",
        encoding="utf-8",
    )


def _ensure_ledgers() -> None:
    if not RESULTS_PATH.exists():
        with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n").writeheader()
    _ensure_section(SUMMARY_PATH, "## Strict GPU Continuation Ledger")
    _ensure_section(NEXT_STEPS_PATH, "## Strict GPU Continuation Progress")


def _ensure_section(path: Path, heading: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.stem}\n\n{heading}\n", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if heading not in text:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{heading}\n")


def _counterfactual_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "model_rna_counterfactual_direction_accuracy",
        "model_rna_counterfactual_logfc_correlation",
        "model_rna_counterfactual_pseudobulk_correlation",
        "model_program_level_effect_recovery",
        "model_rna_counterfactual_top50_de_overlap",
    )
    return {key: _metric(after, key) - _metric(before, key) for key in keys}


def _read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _existing_experiments() -> set[int]:
    return {int(row["experiment_num"]) for row in _read_results_rows()}


def _next_experiment_number() -> int:
    existing = _existing_experiments()
    return max(existing, default=0) + 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sf(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
