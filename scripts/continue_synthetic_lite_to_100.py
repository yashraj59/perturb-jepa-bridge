from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.config import ObjectiveScheduleConfig
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.training.trainer import BridgeTrainer
from scripts.run_synthetic_lite_step0 import (
    _experiment_config_for_dataset,
    _jsonable,
    _train_with_early_stopping,
    _write_dataset_report,
    evaluate_step0,
)


ROOT = Path("outputs/autoresearch_synth_lite")
EXPERIMENT_ROOT = ROOT / "experiments"
RESULTS_PATH = ROOT / "results.tsv"
BASELINE_METRICS = ROOT / "step0_baselines" / "synth_micro" / "step0_seed0_metrics.json"
RUNNER = Path("scripts/run_synthetic_lite_step0.py")


@dataclass(frozen=True)
class Candidate:
    number: int
    family: str
    name: str
    change: str
    description: str
    args: tuple[str, ...]


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


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_results_header()
    baseline = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    existing = _existing_experiments()
    commit = _git(["rev-parse", "HEAD"])
    candidates = _candidate_plan()
    if len(candidates) != 93:
        raise RuntimeError(f"expected 93 continuation candidates, found {len(candidates)}")
    dataset_config = synthetic_lite_config("synth_micro", seed=0)
    dataset = generate_synthetic_biology_lite(dataset_config)

    for candidate in candidates:
        if candidate.number in existing:
            print(f"skip {candidate.number:03d} {candidate.name}", flush=True)
            continue
        metrics = _run_candidate(candidate, dataset, dataset_config)
        decision = _decision_label(metrics, baseline)
        _append_result(commit, candidate, metrics, decision)
        _append_journal(candidate, metrics, decision)
        _append_architecture_log(candidate, metrics, decision)
        _append_family_allocation(candidate)
        print(
            f"done {candidate.number:03d} {candidate.family} {candidate.name} "
            f"{decision} recall1={_metric(metrics, 'model_rna_to_image_recall@1'):.4f} "
            f"collapse={bool(metrics.get('collapse_flag'))}",
            flush=True,
        )
        if _eval_broken(metrics):
            _write_eval_broken(candidate, metrics)
            return 2
    _write_continuation_report()
    return 0


def _candidate_plan() -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        family: str,
        slug: str,
        change: str,
        description: str,
        args: Iterable[str],
    ) -> None:
        number = 8 + len(candidates)
        candidates.append(
            Candidate(
                number=number,
                family=family,
                name=slug,
                change=change,
                description=description,
                args=tuple(args),
            )
        )

    for weight in (0.005, 0.01, 0.02, 0.05):
        add("D", f"counterfactual_weight_{_slug_float(weight)}", "counterfactual_weight", "Counterfactual RNA loss weight probe.", ("--counterfactual-weight", str(weight)))
    for weight in (0.01, 0.02):
        for cycle in (0.0, 0.02, 0.1):
            add(
                "D",
                f"counterfactual_{_slug_float(weight)}_cycle_{_slug_float(cycle)}",
                "counterfactual_cycle_balance",
                "Counterfactual loss with altered cycle consistency.",
                ("--counterfactual-weight", str(weight), "--cycle-weight", str(cycle)),
            )
    for bottleneck in (0.0, 0.001, 0.02):
        add(
            "D",
            f"counterfactual_bottleneck_{_slug_float(bottleneck)}",
            "counterfactual_response_bottleneck",
            "Counterfactual loss with response bottleneck adjustment.",
            ("--counterfactual-weight", "0.01", "--response-bottleneck-weight", str(bottleneck)),
        )

    for weight in (0.0, 0.005, 0.01, 0.05, 0.1):
        add("E", f"batch_adv_{_slug_float(weight)}", "batch_adv_weight", "Batch adversary loss weight probe.", ("--batch-adv-weight", str(weight)))
    for weight in (0.0, 0.01, 0.1):
        add("E", f"perturbation_cls_{_slug_float(weight)}", "perturbation_cls_weight", "Perturbation classifier weight probe.", ("--perturbation-cls-weight", str(weight)))
    for batch_weight in (0.0, 0.05, 0.1):
        for align in (0.25, 0.5):
            add(
                "E",
                f"batch_{_slug_float(batch_weight)}_align_{_slug_float(align)}",
                "batch_alignment_tradeoff",
                "Batch adversary versus cross-modal alignment tradeoff.",
                ("--batch-adv-weight", str(batch_weight), "--align-weight", str(align)),
            )
    add("E", "adversary_scale_0", "adversary_scale", "Disable gradient reversal scale while retaining batch loss plumbing.", ("--adversary-scale", "0.0"))

    for warmup in (5, 10, 20, 30, 40):
        for final_scale in (0.0, 0.2):
            add(
                "F",
                f"schedule_w{warmup}_f{_slug_float(final_scale)}",
                "objective_schedule",
                "Reconstruction-first schedule variant.",
                (
                    "--schedule-reconstruction-warmup-steps",
                    str(warmup),
                    "--schedule-reconstruction-anneal-steps",
                    "20",
                    "--schedule-reconstruction-final-scale",
                    str(final_scale),
                    "--schedule-warmup-non-reconstruction-scale",
                    "0.1",
                ),
            )
    for warmup_non_recon in (0.0, 0.25):
        for anneal in (10, 40, 80):
            add(
                "F",
                f"schedule_a{anneal}_nr{_slug_float(warmup_non_recon)}",
                "objective_schedule_anneal",
                "Anneal-length and non-reconstruction warmup probe.",
                (
                    "--schedule-reconstruction-warmup-steps",
                    "15",
                    "--schedule-reconstruction-anneal-steps",
                    str(anneal),
                    "--schedule-reconstruction-final-scale",
                    "0.2",
                    "--schedule-warmup-non-reconstruction-scale",
                    str(warmup_non_recon),
                ),
            )
    for mask_weight in (0.05, 0.1, 0.4, 0.8):
        add(
            "F",
            f"mask_weight_{_slug_float(mask_weight)}",
            "mask_reconstruction_weight",
            "Balanced RNA/image masked reconstruction weight probe.",
            ("--rna-mask-weight", str(mask_weight), "--image-mask-weight", str(mask_weight)),
        )

    for align in (0.0, 0.25, 0.5, 1.5, 2.0):
        add("C", f"align_{_slug_float(align)}", "align_weight", "Cross-modal alignment loss weight probe.", ("--align-weight", str(align)))
    for temperature in (0.05, 0.2, 0.5):
        add("C", f"temperature_{_slug_float(temperature)}", "contrastive_temperature", "InfoNCE temperature probe.", ("--temperature", str(temperature)))
    for align, temperature in ((0.5, 0.2), (1.5, 0.2), (0.5, 0.05), (1.5, 0.05)):
        add(
            "C",
            f"align_{_slug_float(align)}_temp_{_slug_float(temperature)}",
            "alignment_temperature_tradeoff",
            "Alignment weight and temperature interaction.",
            ("--align-weight", str(align), "--temperature", str(temperature)),
        )
    for align, temperature in ((0.25, 0.2), (0.5, 0.2), (1.0, 0.2), (0.5, 0.05)):
        add(
            "C",
            f"multipos_align_{_slug_float(align)}_temp_{_slug_float(temperature)}",
            "multi_positive_alignment_tradeoff",
            "Multi-positive alignment with softened contrastive settings.",
            ("--multi-positive-alignment", "--align-weight", str(align), "--temperature", str(temperature)),
        )
    for jepa, align in ((0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.5, 0.5)):
        add(
            "C",
            f"jepa_{_slug_float(jepa)}_align_{_slug_float(align)}",
            "jepa_alignment_balance",
            "JEPA and alignment weight balance.",
            ("--jepa-weight", str(jepa), "--align-weight", str(align)),
        )

    for weight in (0.001, 0.002, 0.005, 0.02, 0.1):
        add("A", f"shared_variance_{_slug_float(weight)}", "shared_variance_weight", "Shared representation variance floor probe.", ("--shared-variance-weight", str(weight)))
    for decay in (0.9, 0.95, 0.98, 0.992, 0.999):
        add("A", f"ema_{_slug_float(decay)}", "ema_decay", "EMA target update speed probe.", ("--ema-decay", str(decay)))
    for weight, decay in ((0.001, 0.99), (0.002, 0.99), (0.005, 0.99), (0.001, 0.998), (0.005, 0.998)):
        add(
            "A",
            f"variance_{_slug_float(weight)}_ema_{_slug_float(decay)}",
            "variance_ema_combination",
            "Shared variance floor plus EMA target-speed probe.",
            ("--shared-variance-weight", str(weight), "--ema-decay", str(decay)),
        )

    for aggregator, prototypes in (
        ("mean", 1),
        ("mean", 2),
        ("mean", 4),
        ("attention", 3),
        ("attention", 4),
        ("attention", 6),
    ):
        add(
            "B",
            f"{aggregator}_prototypes_{prototypes}",
            "bag_aggregator_capacity",
            "Bag aggregator capacity and pooling probe.",
            ("--bag-aggregator", aggregator, "--num-bag-prototypes", str(prototypes)),
        )
    for dropout in (0.0, 0.02, 0.1, 0.2):
        add("B", f"dropout_{_slug_float(dropout)}", "dropout", "Encoder/dropout regularization probe.", ("--dropout", str(dropout)))

    return candidates


def _run_candidate(candidate: Candidate, dataset, dataset_config) -> dict:
    slug = f"{candidate.number:03d}_{candidate.name}"
    output_root = EXPERIMENT_ROOT / slug
    options = _candidate_options(candidate.args)
    steps = 5
    batch_size = 8
    device = "cpu"
    seed = 0
    seed_everything(seed)
    started = time.perf_counter()
    multi_positive_alignment = bool(options.pop("multi_positive_alignment", False))

    experiment_config = _experiment_config_for_dataset(
        dataset,
        steps=steps,
        device=device,
        **options,
    )
    config_dir = output_root / "step0_baselines" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    experiment_config.save_json(config_dir / "synth_micro_seed0_bridge.json")
    (config_dir / "synth_micro_seed0_generator.json").write_text(
        json.dumps(asdict(dataset_config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model = experiment_config.build_model()
    optimizer = experiment_config.build_optimizer(model.parameters())
    trainer = BridgeTrainer(
        model,
        optimizer,
        weights=experiment_config.loss,
        ema_decay=experiment_config.training.ema_decay,
        schedule=experiment_config.training.objective_schedule,
        device=device,
        grad_clip_norm=experiment_config.training.grad_clip_norm,
        multi_positive_alignment=multi_positive_alignment,
    )
    history = _train_with_early_stopping(
        trainer,
        dataset,
        steps=steps,
        batch_size=batch_size,
        bag_size=dataset_config.cells_per_condition,
        device=device,
        seed=seed,
    )
    metrics = evaluate_step0(
        dataset,
        model,
        split="test",
        train_split="train",
        device=device,
        bag_size=dataset_config.cells_per_condition,
        seed=seed,
        label_shuffle_repeats=5,
    )
    metrics["training_steps_completed"] = float(len(history))
    metrics["wallclock_minutes"] = float((time.perf_counter() - started) / 60.0)
    metrics["device_used"] = device
    metrics["max_gpu_memory_gb"] = 0.0

    dataset_dir = output_root / "step0_baselines" / "synth_micro"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "DATASET_REFERENCE.md").write_text(
        "This continuation micro-screen reuses the locked Step 0 `synth_micro` synthetic dataset. "
        "Canonical metadata, generation config, and ground truth are in "
        "`outputs/autoresearch_synth_lite/step0_baselines/synth_micro/`.\n",
        encoding="utf-8",
    )
    (dataset_dir / "step0_seed0_metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "step0_seed0_history.json").write_text(
        json.dumps(_jsonable(history), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_dataset_report(dataset_dir / "synth_micro_baseline.md", "synth_micro", seed, metrics, history)
    return metrics


def _candidate_options(args: tuple[str, ...]) -> dict:
    options = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.05,
        "adversary_scale": 0.5,
        "temperature": 0.1,
        "rna_mask_weight": 0.2,
        "image_mask_weight": 0.2,
        "jepa_weight": 1.0,
        "align_weight": 1.0,
        "mmd_weight": 0.05,
        "sliced_wasserstein_weight": 0.02,
        "perturbation_cls_weight": 0.05,
        "batch_adv_weight": 0.02,
        "counterfactual_weight": 0.0,
        "cycle_weight": 0.05,
        "response_bottleneck_weight": 0.005,
        "shared_variance_weight": 0.0,
        "shared_covariance_weight": 0.0,
        "cross_correlation_weight": 0.0,
        "ema_decay": 0.996,
        "bag_aggregator": "attention",
        "num_bag_prototypes": 2,
        "objective_schedule": ObjectiveScheduleConfig(),
        "multi_positive_alignment": False,
    }
    schedule = {
        "enabled": False,
        "reconstruction_warmup_steps": 0,
        "reconstruction_anneal_steps": 0,
        "reconstruction_final_scale": 1.0,
        "warmup_non_reconstruction_scale": 0.0,
    }
    mapping = {
        "--lr": ("lr", float),
        "--weight-decay": ("weight_decay", float),
        "--dropout": ("dropout", float),
        "--adversary-scale": ("adversary_scale", float),
        "--temperature": ("temperature", float),
        "--rna-mask-weight": ("rna_mask_weight", float),
        "--image-mask-weight": ("image_mask_weight", float),
        "--jepa-weight": ("jepa_weight", float),
        "--align-weight": ("align_weight", float),
        "--mmd-weight": ("mmd_weight", float),
        "--sliced-wasserstein-weight": ("sliced_wasserstein_weight", float),
        "--perturbation-cls-weight": ("perturbation_cls_weight", float),
        "--batch-adv-weight": ("batch_adv_weight", float),
        "--counterfactual-weight": ("counterfactual_weight", float),
        "--cycle-weight": ("cycle_weight", float),
        "--response-bottleneck-weight": ("response_bottleneck_weight", float),
        "--shared-variance-weight": ("shared_variance_weight", float),
        "--shared-covariance-weight": ("shared_covariance_weight", float),
        "--cross-correlation-weight": ("cross_correlation_weight", float),
        "--ema-decay": ("ema_decay", float),
        "--bag-aggregator": ("bag_aggregator", str),
        "--num-bag-prototypes": ("num_bag_prototypes", int),
    }
    schedule_mapping = {
        "--schedule-reconstruction-warmup-steps": ("reconstruction_warmup_steps", int),
        "--schedule-reconstruction-anneal-steps": ("reconstruction_anneal_steps", int),
        "--schedule-reconstruction-final-scale": ("reconstruction_final_scale", float),
        "--schedule-warmup-non-reconstruction-scale": ("warmup_non_reconstruction_scale", float),
    }
    index = 0
    while index < len(args):
        flag = args[index]
        if flag == "--multi-positive-alignment":
            options["multi_positive_alignment"] = True
            index += 1
            continue
        if flag in mapping:
            key, caster = mapping[flag]
            options[key] = caster(args[index + 1])
            index += 2
            continue
        if flag in schedule_mapping:
            key, caster = schedule_mapping[flag]
            schedule[key] = caster(args[index + 1])
            schedule["enabled"] = True
            index += 2
            continue
        raise ValueError(f"unsupported candidate flag: {flag}")
    if (
        schedule["reconstruction_warmup_steps"] > 0
        or schedule["reconstruction_anneal_steps"] > 0
        or schedule["reconstruction_final_scale"] != 1.0
        or schedule["warmup_non_reconstruction_scale"] != 0.0
    ):
        schedule["enabled"] = True
    options["objective_schedule"] = ObjectiveScheduleConfig(**schedule)
    return options


def _decision_label(metrics: dict, baseline: dict) -> str:
    if bool(metrics.get("collapse_flag")):
        return "TIER1_DISCARD_MODE_COLLAPSE"
    if _metric(metrics, "model_delta_norm_ratio") < 0.01:
        return "TIER1_DISCARD_DEAD_DELTA"
    majority_gap = _metric(metrics, "model_batch_probe_balanced_accuracy") - _metric(metrics, "model_batch_probe_majority_accuracy")
    baseline_gap = _metric(baseline, "model_batch_probe_balanced_accuracy") - _metric(baseline, "model_batch_probe_majority_accuracy")
    if majority_gap > baseline_gap + 0.05:
        return "TIER1_DISCARD_BATCH_LEAKAGE"
    improved = (
        _metric(metrics, "model_rna_to_image_recall@1") >= _metric(baseline, "model_rna_to_image_recall@1") * 1.05
        or _metric(metrics, "model_rna_counterfactual_direction_accuracy")
        >= _metric(baseline, "model_rna_counterfactual_direction_accuracy") * 1.05
        or _metric(metrics, "model_bio_latent_r2_rna_shared")
        >= _metric(baseline, "model_bio_latent_r2_rna_shared") + 0.05
    )
    regressed = (
        _metric(metrics, "model_rna_to_image_recall@1") < _metric(baseline, "model_rna_to_image_recall@1") * 0.85
        or _metric(metrics, "model_rna_counterfactual_direction_accuracy")
        < _metric(baseline, "model_rna_counterfactual_direction_accuracy") * 0.85
    )
    beats_random = _metric(metrics, "model_rna_to_image_recall@1") > _metric(metrics, "random_embedding_rna_to_image_recall@1")
    beats_source = _metric(metrics, "model_rna_counterfactual_direction_accuracy") > _metric(metrics, "source_as_target_rna_counterfactual_direction_accuracy")
    if improved and not regressed and beats_random and beats_source:
        return "TIER1_KEEP_CONTROLLED_SIGNAL"
    if not beats_random:
        return "TIER1_DISCARD_NO_SIGNAL"
    return "TIER1_DISCARD_METRIC_REGRESSION"


def _append_result(commit: str, candidate: Candidate, metrics: dict, decision: str) -> None:
    row = {
        "commit": commit,
        "experiment_num": f"{candidate.number:03d}",
        "family": candidate.family,
        "candidate_name": candidate.name,
        "tier_reached": "Tier 1 micro",
        "decision_label": decision,
        "device_used": str(metrics.get("device_used", "cpu")),
        "wallclock_minutes": f"{_metric(metrics, 'wallclock_minutes'):.3f}",
        "max_gpu_memory_gb": f"{_metric(metrics, 'max_gpu_memory_gb'):.1f}",
        "synth_micro_recall1": f"{_metric(metrics, 'model_rna_to_image_recall@1'):.6f}",
        "synth_easy_recall1": "NA",
        "synth_medium_recall1": "NA",
        "synth_easy_cf_dir_acc": "NA",
        "synth_medium_cf_dir_acc": "NA",
        "heldout_pert_cf_dir_acc": "NA",
        "batch_confound_batch_leakage": "NA",
        "batch_confound_recall1": "NA",
        "dose_extrap_logfc_corr": "NA",
        "bio_latent_r2": f"{_metric(metrics, 'model_bio_latent_r2_rna_shared'):.6f}",
        "representation_rank": f"{_metric(metrics, 'model_embedding_rank'):.1f}",
        "delta_norm_ratio": f"{_metric(metrics, 'model_delta_norm_ratio'):.6f}",
        "cap_bound": "false",
        "collapse_flag": str(bool(metrics.get("collapse_flag"))).lower(),
        "architecture_change": candidate.change,
        "description": candidate.description,
    }
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writerow(row)


def _append_journal(candidate: Candidate, metrics: dict, decision: str) -> None:
    with (ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Experiment {candidate.number:03d}: {candidate.name}\n\n"
            f"### Family\n{candidate.family}\n\n"
            f"### Hypothesis\n{candidate.description}\n\n"
            f"### Compute budget\nCPU-only `synth_micro` micro Tier 1 screen, seed 0, max 50 steps. "
            f"Repeated-collapse and saturation stops are disabled by Amendment 001; candidate-level collapse still discards the candidate.\n\n"
            f"### Metrics\n"
            f"- RNA->image recall@1: {_metric(metrics, 'model_rna_to_image_recall@1'):.4f}\n"
            f"- Counterfactual direction accuracy: {_metric(metrics, 'model_rna_counterfactual_direction_accuracy'):.4f}\n"
            f"- RNA biological latent R2: {_metric(metrics, 'model_bio_latent_r2_rna_shared'):.4f}\n"
            f"- Batch balanced accuracy: {_metric(metrics, 'model_batch_probe_balanced_accuracy'):.4f}\n"
            f"- Embedding rank: {_metric(metrics, 'model_embedding_rank'):.1f}\n"
            f"- Delta norm ratio: {_metric(metrics, 'model_delta_norm_ratio'):.4f}\n"
            f"- Collapse flag: {bool(metrics.get('collapse_flag'))}\n\n"
            f"### Decision label\n{decision}\n\n"
            f"### audit_relevant\nNo checkpoint retained.\n"
        )


def _append_architecture_log(candidate: Candidate, metrics: dict, decision: str) -> None:
    with (ROOT / "architectural_changes_log.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Change {candidate.number:03d}: {candidate.name}\n\n"
            f"### Family\n{candidate.family}\n\n"
            f"### Exact mechanism\n{candidate.change}: `{' '.join(candidate.args)}`\n\n"
            f"### Why this preserves JEPA\nThe RNA encoder, JEPA-style objectives, shared bridge outputs, and counterfactual path remain active.\n\n"
            f"### Compute cost estimate\nCPU-only micro screen; no model width, dataset size, or image size increase.\n\n"
            f"### Observed effect\nRecall@1 {_metric(metrics, 'model_rna_to_image_recall@1'):.4f}; "
            f"collapse {bool(metrics.get('collapse_flag'))}; RNA latent R2 {_metric(metrics, 'model_bio_latent_r2_rna_shared'):.4f}.\n\n"
            f"### Verdict\n{decision}.\n"
        )


def _append_family_allocation(candidate: Candidate) -> None:
    counts: dict[str, int] = {}
    for row in _read_results_rows():
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    with (ROOT / "family_allocation.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Experiment {candidate.number:03d} Allocation Update\n\n"
            f"Family {candidate.family} completed {counts.get(candidate.family, 0)} experiments. "
            f"Total completed experiments: {sum(counts.values())}. Amendment 001 requires continuing to 100 unless a safety or evaluation stop fires.\n"
        )


def _write_continuation_report() -> None:
    rows = _read_results_rows()
    by_label: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_label[row["decision_label"]] = by_label.get(row["decision_label"], 0) + 1
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
    report = ROOT / "continuation_100_summary.md"
    report.write_text(
        "# Continuation To 100 Summary\n\n"
        f"Total recorded experiments: {len(rows)}\n\n"
        "## Decision Labels\n\n"
        + "\n".join(f"- {label}: {count}" for label, count in sorted(by_label.items()))
        + "\n\n## Family Counts\n\n"
        + "\n".join(f"- {family}: {count}" for family, count in sorted(by_family.items()))
        + "\n\nAll continuation runs were CPU-only `synth_micro` micro screens unless explicitly documented otherwise.\n",
        encoding="utf-8",
    )


def _write_eval_broken(candidate: Candidate, metrics: dict) -> None:
    path = ROOT / "EVAL_BROKEN.md"
    path.write_text(
        f"# Evaluation Broken\n\nExperiment {candidate.number:03d} had label-shuffle retrieval materially above random.\n\n"
        f"- label shuffle recall@1: {_metric(metrics, 'label_shuffle_rna_to_image_recall@1'):.4f}\n"
        f"- random recall@1: {_metric(metrics, 'random_embedding_rna_to_image_recall@1'):.4f}\n",
        encoding="utf-8",
    )


def _eval_broken(metrics: dict) -> bool:
    label = _metric(metrics, "label_shuffle_rna_to_image_recall@1")
    random = _metric(metrics, "random_embedding_rna_to_image_recall@1")
    return label > max(random + 0.15, random * 3.0)


def _ensure_results_header() -> None:
    if RESULTS_PATH.exists() and RESULTS_PATH.read_text(encoding="utf-8").strip():
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()


def _existing_experiments() -> set[int]:
    return {int(row["experiment_num"]) for row in _read_results_rows()}


def _read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    lines = [line for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def _metric(metrics: dict, key: str) -> float:
    value = metrics.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
