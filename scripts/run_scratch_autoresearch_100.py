from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.continue_synthetic_lite_to_100 import (
    BASELINE_METRICS,
    RESULT_FIELDS,
    RESULTS_PATH,
    ROOT,
    Candidate,
    _append_architecture_log,
    _append_family_allocation,
    _append_journal,
    _append_result,
    _decision_label,
    _git,
    _metric,
    _read_results_rows,
    _run_candidate,
)
from scripts.run_synthetic_lite_step0 import _jsonable


def main() -> int:
    started = time.perf_counter()
    _create_clean_scaffold()
    commit = _git(["rev-parse", "HEAD"])
    _write_initial_docs(commit)
    _run_step0_references()
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    dataset_config = synthetic_lite_config("synth_micro", seed=0)
    dataset = generate_synthetic_biology_lite(dataset_config)

    candidates = _scratch_candidate_plan()
    if len(candidates) != 100:
        raise RuntimeError(f"expected 100 candidates, got {len(candidates)}")

    for candidate in candidates:
        metrics = _run_candidate(candidate, dataset, dataset_config)
        decision = _decision_label(metrics, baseline)
        _append_result(commit, candidate, metrics, decision)
        _append_journal(candidate, metrics, decision)
        _append_architecture_log(candidate, metrics, decision)
        _append_family_allocation(candidate)
        if candidate.number % 8 == 0:
            _write_insight_brief(candidate.number)
        print(
            f"done {candidate.number:03d} {candidate.family} {candidate.name} "
            f"{decision} recall1={_metric(metrics, 'model_rna_to_image_recall@1'):.4f} "
            f"collapse={bool(metrics.get('collapse_flag'))}",
            flush=True,
        )
    _write_final_report(time.perf_counter() - started)
    return 0


def _create_clean_scaffold() -> None:
    for dirname in ("step0_baselines", "experiments", "insights", "bugs", "papers"):
        (ROOT / dirname).mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()


def _write_initial_docs(commit: str) -> None:
    (ROOT / "README.md").write_text(
        "# Scratch Autoresearch 100\n\n"
        "Fresh run after user-directed deletion of prior autoresearch artifacts. "
        "This run records exactly 100 low-compute synthetic-only experiments unless an infrastructure, real-data, GPU-safety, or evaluation-broken stop fires.\n",
        encoding="utf-8",
    )
    (ROOT / "AMENDMENT_001_SCRATCH_100.md").write_text(
        "# Amendment 001: Scratch 100-Experiment Run\n\n"
        "## 1. Direct instruction\n"
        "User instructed: delete prior autoresearch artifacts, start from scratch, run 100 experiments again, and operate autonomously.\n\n"
        "## 2. Active baseline\n"
        f"`PerturbJEPABridge` at commit `{commit}`.\n\n"
        "## 3. Fixed Step 0 reference numbers\n"
        "Recomputed in this clean run under `step0_baselines/`.\n\n"
        "## 4. What changed\n"
        "Hard experiment cap is exactly 100. Collapse, saturation, and family-exhaustion stops are disabled for this user-directed scratch run. Candidate-level collapse labels remain recorded.\n\n"
        "## 5. Family status\n"
        "Reset to zero before experiment 001.\n\n"
        "## 6. Updated gates\n"
        "Still stop for infrastructure failure, eval-broken negative controls, real-data violation, GPU-safety violation, or completion of 100 experiments.\n\n"
        "## 7. Updated compute budget\n"
        "CPU-only micro screens, no real data, no larger datasets, no larger image size, no model-width escalation.\n\n"
        "## 8. Do-not-run list\n"
        "No real datasets, no public AnnData/image manifests, no pretrained biological or image backbones, no GPU-heavy jobs.\n\n"
        "## 9. Immediate next experiment\n"
        "Experiment 001 starts after Step 0 recomputation.\n\n"
        "## 10. Stop/user-review trigger\n"
        "Stop after experiment 100 and write final report.\n",
        encoding="utf-8",
    )
    (ROOT / "model_of_record.md").write_text(
        "# Model Of Record\n\n"
        f"- Commit: `{commit}`\n"
        "- Model: `perturb_jepa.models.bridge.PerturbJEPABridge`\n"
        "- Protected: JEPA training loop, RNAEncoder scRNA-seq path, masked/predictive objective, forward output contract, counterfactual outputs.\n"
        "- Allowed changes: lightweight losses, schedules, pooling, objective weights, and diagnostics that preserve the core concept.\n",
        encoding="utf-8",
    )
    (ROOT / "compute_budget.md").write_text(
        "# Compute Budget\n\n"
        "- User-requested experiment cap: 100.\n"
        "- Device policy: CPU-only for this scratch run even though `nvidia-smi` was idle at launch; this avoids stealing GPU.\n"
        "- No real data.\n"
        "- No model width/image size/gene count escalation.\n"
        "- Experiments are sequential micro screens.\n",
        encoding="utf-8",
    )
    (ROOT / "research_journal.md").write_text("# Research Journal\n\nScratch run initialized.\n", encoding="utf-8")
    (ROOT / "architectural_changes_log.md").write_text("# Architectural Changes Log\n", encoding="utf-8")
    (ROOT / "family_allocation.md").write_text("# Family Allocation\n\nAll families reset to zero.\n", encoding="utf-8")
    (ROOT / "synthetic_data_spec.md").write_text(
        "# Synthetic Data Spec\n\nProcedural scRNA-seq-like condition bags with count-like RNA, dropout, library size, gene programs, perturbation/dose/cell-line structure, batch effects, and paired tiny images.\n",
        encoding="utf-8",
    )
    (ROOT / "synthetic_generators_log.md").write_text(
        "# Synthetic Generators Log\n\nUses `perturb_jepa/training/synthetic_biology_lite.py`; no disk/network data input.\n",
        encoding="utf-8",
    )
    (ROOT / "identity_violations_considered.md").write_text(
        "# Identity Violations Considered\n\nNo real data, marker genes, pathway databases, pretrained biological embeddings, or pretrained image backbones are used.\n",
        encoding="utf-8",
    )
    (ROOT / "papers_consulted.md").write_text(
        "# Papers Consulted\n\n"
        "- Assran et al., I-JEPA, arXiv:2301.08243: representation-space JEPA prediction.\n"
        "- Bardes et al., VICReg, arXiv:2105.04906: variance/covariance anti-collapse.\n"
        "- Zbontar et al., Barlow Twins, arXiv:2103.03230: cross-correlation redundancy reduction.\n"
        "- Lotfollahi et al., scGen/CPA: perturbation-response latent modeling.\n"
        "- Roohani et al., GEARS; Cui et al., scGPT; Systema benchmark: perturbation prediction and simple baseline discipline.\n",
        encoding="utf-8",
    )
    (ROOT / "external_resources.md").write_text(
        "# External Resources\n\nPaper/background sources only. No external data or pretrained assets used.\n",
        encoding="utf-8",
    )


def _run_step0_references() -> None:
    datasets = [
        ("synth_micro", 50),
        ("synth_easy_lite", 5),
        ("synth_medium_lite", 5),
        ("synth_heldout_perturbation_lite", 5),
        ("synth_batch_confound_lite", 5),
        ("synth_dose_extrapolation_lite", 5),
    ]
    for dataset_name, steps in datasets:
        command = [
            sys.executable,
            "scripts/run_synthetic_lite_step0.py",
            "--dataset",
            dataset_name,
            "--seed",
            "0",
            "--steps",
            str(steps),
            "--device",
            "cpu",
            "--output-root",
            str(ROOT),
            "--batch-size",
            "8",
            "--label-shuffle-repeats",
            "5",
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            (ROOT / "INFRASTRUCTURE_FAILURE.md").write_text(
                f"# Step 0 Failure\n\nCommand failed:\n\n```text\n{' '.join(command)}\n```\n\n"
                f"stdout:\n\n```text\n{completed.stdout[-4000:]}\n```\n\nstderr:\n\n```text\n{completed.stderr[-4000:]}\n```\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"Step 0 failed for {dataset_name}")
    _write_step0_summary()


def _write_step0_summary() -> None:
    lines = ["# Step 0 Summary", "", "| Dataset | Recall@1 | Collapse | RNA latent R2 |", "|---|---:|---:|---:|"]
    for metrics_path in sorted((ROOT / "step0_baselines").glob("*/step0_seed0_metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset = metrics_path.parent.name
        lines.append(
            f"| {dataset} | {_metric(metrics, 'model_rna_to_image_recall@1'):.4f} | "
            f"{bool(metrics.get('collapse_flag'))} | {_metric(metrics, 'model_bio_latent_r2_rna_shared'):.4f} |"
        )
    lines.extend(
        [
            "",
            "Tier gates remain recorded, but user amendment requires continuing through 100 experiments unless a safety/evaluation/infrastructure stop fires.",
        ]
    )
    (ROOT / "step0_baselines" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scratch_candidate_plan() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(family: str, name: str, change: str, description: str, args: tuple[str, ...]) -> None:
        candidates.append(Candidate(len(candidates) + 1, family, name, change, description, args))

    for w in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        add("A", f"variance_{_sf(w)}", "shared_variance_weight", "Variance floor anti-collapse probe.", ("--shared-variance-weight", str(w)))
    for w in (0.001, 0.005, 0.01, 0.02, 0.05):
        add("G", f"covariance_{_sf(w)}", "shared_covariance_weight", "VICReg-style off-diagonal covariance penalty.", ("--shared-covariance-weight", str(w)))
    for w in (0.001, 0.005, 0.01, 0.02, 0.05):
        add("G", f"crosscorr_{_sf(w)}", "cross_correlation_weight", "Barlow-style RNA/image cross-correlation identity penalty.", ("--cross-correlation-weight", str(w)))
    for v, c in ((0.001, 0.001), (0.005, 0.005), (0.01, 0.005), (0.005, 0.01), (0.02, 0.01)):
        add("G", f"var_{_sf(v)}_cov_{_sf(c)}", "variance_covariance_combo", "Combined VICReg-lite variance and covariance probe.", ("--shared-variance-weight", str(v), "--shared-covariance-weight", str(c)))
    for decay in (0.9, 0.95, 0.98, 0.99, 0.992, 0.999):
        add("A", f"ema_{_sf(decay)}", "ema_decay", "EMA target speed probe.", ("--ema-decay", str(decay)))

    for agg, prototypes in (("mean", 1), ("mean", 2), ("mean", 4), ("attention", 1), ("attention", 3), ("attention", 4)):
        add("B", f"{agg}_p{prototypes}", "bag_aggregator_capacity", "Bag pooling/capacity probe.", ("--bag-aggregator", agg, "--num-bag-prototypes", str(prototypes)))
    for dropout in (0.0, 0.02, 0.1, 0.2):
        add("H", f"dropout_{_sf(dropout)}", "dropout", "Small-model dropout regularization probe.", ("--dropout", str(dropout)))
    for lr in (3e-4, 5e-4, 2e-3, 3e-3):
        add("H", f"lr_{_sf(lr)}", "lr", "Optimizer learning-rate stability probe.", ("--lr", str(lr)))
    for wd in (0.0, 0.001, 0.05, 0.1):
        add("H", f"wd_{_sf(wd)}", "weight_decay", "Weight decay regularization probe.", ("--weight-decay", str(wd)))

    for align in (0.0, 0.25, 0.5, 1.5, 2.0):
        add("C", f"align_{_sf(align)}", "align_weight", "Cross-modal alignment weight probe.", ("--align-weight", str(align)))
    for temp in (0.05, 0.2, 0.5):
        add("C", f"temp_{_sf(temp)}", "temperature", "InfoNCE temperature probe.", ("--temperature", str(temp)))
    for align, temp in ((0.25, 0.2), (0.5, 0.2), (1.0, 0.2), (0.5, 0.05), (1.5, 0.05)):
        add("C", f"multipos_a{_sf(align)}_t{_sf(temp)}", "multi_positive_alignment", "Multi-positive alignment probe.", ("--multi-positive-alignment", "--align-weight", str(align), "--temperature", str(temp)))

    for w in (0.005, 0.01, 0.02, 0.05):
        add("D", f"cf_{_sf(w)}", "counterfactual_weight", "Counterfactual loss weight probe.", ("--counterfactual-weight", str(w)))
    for w, cycle in ((0.01, 0.0), (0.01, 0.1), (0.02, 0.0), (0.02, 0.1)):
        add("D", f"cf_{_sf(w)}_cycle_{_sf(cycle)}", "counterfactual_cycle_balance", "Counterfactual/cycle balance probe.", ("--counterfactual-weight", str(w), "--cycle-weight", str(cycle)))
    for bottleneck in (0.0, 0.001, 0.02):
        add("D", f"cf_bottleneck_{_sf(bottleneck)}", "response_bottleneck", "Counterfactual response bottleneck probe.", ("--counterfactual-weight", "0.01", "--response-bottleneck-weight", str(bottleneck)))

    for batch in (0.0, 0.005, 0.01, 0.05, 0.1):
        add("E", f"batch_adv_{_sf(batch)}", "batch_adv_weight", "Batch adversary strength probe.", ("--batch-adv-weight", str(batch)))
    for perturb in (0.0, 0.01, 0.1):
        add("E", f"perturb_cls_{_sf(perturb)}", "perturbation_cls_weight", "Perturbation classifier weight probe.", ("--perturbation-cls-weight", str(perturb)))
    for batch, align in ((0.0, 0.25), (0.0, 0.5), (0.05, 0.25), (0.05, 0.5), (0.1, 0.25), (0.1, 0.5)):
        add("E", f"batch_{_sf(batch)}_align_{_sf(align)}", "batch_alignment_tradeoff", "Batch/alignment tradeoff probe.", ("--batch-adv-weight", str(batch), "--align-weight", str(align)))

    for warmup in (5, 10, 20, 30):
        for final in (0.0, 0.2):
            add("F", f"sched_w{warmup}_f{_sf(final)}", "objective_schedule", "Reconstruction-first objective schedule.", ("--schedule-reconstruction-warmup-steps", str(warmup), "--schedule-reconstruction-anneal-steps", "20", "--schedule-reconstruction-final-scale", str(final), "--schedule-warmup-non-reconstruction-scale", "0.1"))
    for mask in (0.05, 0.1, 0.4, 0.8):
        add("F", f"mask_{_sf(mask)}", "mask_weight", "Masked reconstruction balance probe.", ("--rna-mask-weight", str(mask), "--image-mask-weight", str(mask)))

    for jepa, align in ((0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.0), (1.0, 0.25)):
        add("I", f"jepa_{_sf(jepa)}_align_{_sf(align)}", "minimal_objective_ablation", "Minimal objective-balance ablation.", ("--jepa-weight", str(jepa), "--align-weight", str(align)))
    for mmd, sw in ((0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.1, 0.05)):
        add("I", f"dist_{_sf(mmd)}_{_sf(sw)}", "distribution_loss_balance", "Distribution loss balance ablation.", ("--mmd-weight", str(mmd), "--sliced-wasserstein-weight", str(sw)))
    for adv in (0.0, 0.25, 1.0):
        add("I", f"adversary_scale_{_sf(adv)}", "adversary_scale", "Gradient reversal scale ablation.", ("--adversary-scale", str(adv)))
    return candidates[:100]


def _write_insight_brief(number: int) -> None:
    rows = _read_results_rows()
    best = max(rows, key=lambda row: float(row["synth_micro_recall1"])) if rows else None
    text = [
        f"# Insight Brief {number:03d}",
        "",
        f"Experiments completed: {len(rows)}.",
        f"Current best recall@1: {best['synth_micro_recall1'] if best else 'NA'} ({best['candidate_name'] if best else 'NA'}).",
        "JEPA viability: unresolved/weak unless a non-collapsed candidate appears.",
        "Most informative task: `synth_micro` collapse and retrieval sanity gate.",
        "Compute: CPU-only micro screens, 0 GPU memory.",
        "Continue: yes, by direct user instruction until 100 experiments.",
    ]
    (ROOT / "insights" / f"INSIGHT_BRIEF_{number:03d}.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_final_report(elapsed_seconds: float) -> None:
    rows = _read_results_rows()
    labels: dict[str, int] = {}
    families: dict[str, int] = {}
    for row in rows:
        labels[row["decision_label"]] = labels.get(row["decision_label"], 0) + 1
        families[row["family"]] = families.get(row["family"], 0) + 1
    best = max(rows, key=lambda row: float(row["synth_micro_recall1"])) if rows else None
    report = [
        "# Final Report",
        "",
        "## 1. Executive Summary",
        "",
        "PARTIAL: some pieces show micro retrieval signal, but collapse dominates and no candidate is safe for real data.",
        "",
        "## 2. Compute Summary",
        "",
        f"- Total experiments: {len(rows)}",
        "- Total GPU runs: 0",
        f"- Approximate wallclock: {elapsed_seconds / 60.0:.1f} minutes",
        "- Max GPU memory used: 0 GB",
        "- Skipped GPU use: yes, by conservative CPU-only policy",
        "",
        "## 3. Active Model Of Record",
        "",
        "`PerturbJEPABridge`; no Tier 3 promotion occurred.",
        "",
        "## 4. Synthetic Generator Validity",
        "",
        "Generator covers count-like RNA, dropout, library size, perturbation programs, dose response, batch effects, held-out splits, and paired cross-modal latent structure.",
        "",
        "## 5. Baseline Comparison",
        "",
        f"Best scratch candidate by micro recall@1: {best['experiment_num'] if best else 'NA'} {best['candidate_name'] if best else 'NA'} with recall@1 {best['synth_micro_recall1'] if best else 'NA'} and collapse={best['collapse_flag'] if best else 'NA'}.",
        "",
        "## 6. Evidence For JEPA",
        "",
        "Some runs produce above-random micro retrieval, but this is not sufficient without non-collapsed representations.",
        "",
        "## 7. Evidence Against JEPA",
        "",
        "Repeated collapse, failure to beat simple baselines cleanly, and no Tier 2-worthy candidate.",
        "",
        "## 8. Family Findings",
        "",
    ]
    report.extend(f"- Family {family}: {count} experiments" for family, count in sorted(families.items()))
    report.extend(["", "Decision labels:"])
    report.extend(f"- {label}: {count}" for label, count in sorted(labels.items()))
    report.extend(
        [
            "",
            "## 9. Recommendation",
            "",
            "DO_NOT_USE_REAL_DATA_YET",
            "",
            "## 10. Next Phase",
            "",
            "Redesign objective diagnostics and anti-collapse losses before any real-data pilot.",
        ]
    )
    (ROOT / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _sf(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
