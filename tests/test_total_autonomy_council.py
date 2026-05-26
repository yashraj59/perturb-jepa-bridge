import numpy as np
import pandas as pd

from scripts.run_bioguard_wm_total_autonomy import (
    _f052_descriptor_near_tie_summary,
    _f053_select_descriptor_margin_gate,
    _f058_decision_from_summary,
    _f059_decision_from_summary,
    _f060_decision_from_summary,
    _f062_decision_from_summary,
    _f063_decision_from_summary,
    _f064_decision_from_summary,
    _f065_decision_from_summary,
    _f066_decision_from_summary,
    _f067_decision_from_summary,
    _f068_decision_from_summary,
    _f069_decision_from_summary,
    _f070_decision_from_summary,
    _f071_decision_from_summary,
    _f072_decision_from_summary,
    _f073_decision_from_summary,
    _f074_decision_from_summary,
    _f075_decision_from_summary,
    _f076_decision_from_summary,
    _f077_decision_from_summary,
    _f078_decision_from_summary,
    _f079_decision_from_summary,
    _f080_decision_from_summary,
    _f081_decision_from_summary,
    _f082_decision_from_summary,
    _f083_decision_from_summary,
    _f084_decision_from_summary,
    _f085_decision_from_summary,
    _f086_decision_from_summary,
    _f087_decision_from_summary,
    _f088_decision_from_summary,
    _f089_decision_from_summary,
    _f090_decision_from_summary,
    _f091_decision_from_summary,
    _f092_decision_from_summary,
    _create_status_write_reserve,
    _floor_direction_projected_head_delta,
    _hard_escalation_active,
    _release_status_write_reserve,
    _retrieval_endpoint_info,
    _residual_cone_head_delta,
    _rowwise_risk_scores,
    _summarize_retrieval_failure_rows,
    _unsafe_margin_mask,
    council_decision,
    default_council_proposals,
    ordinary_stop_requires_council,
)


def test_ordinary_stop_triggers_council_but_hard_escalation_does_not():
    assert ordinary_stop_requires_council("BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN")
    assert not ordinary_stop_requires_council("HARD_ESCALATE_FORBIDDEN_LEAKAGE_CANNOT_BE_FIXED")


def test_debate_council_selects_feasible_diagnostic():
    decision, selected = council_decision(default_council_proposals("BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN"))

    assert decision == "COUNCIL_EXECUTE"
    assert "bootstrap" in selected.name.lower()
    assert selected.minimum >= 0.40


def test_retrieval_endpoint_info_reports_rank_and_margin():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.95, 0.05]], dtype=float)
    gallery = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]], dtype=float)
    labels = ["a", "b", "a"]

    info = _retrieval_endpoint_info(query, gallery, labels)

    assert info["rank"].tolist() == [1.0, 1.0, 1.0]
    assert info["top1_correct"].tolist() == [True, True, True]
    assert np.all(info["margin"] > 0.0)


def test_retrieval_failure_summary_counts_broken_near_ties():
    frame = pd.DataFrame(
        [
            {
                "flip_type": "broken_by_residual",
                "floor_near_tie_0p05": True,
                "selected_near_tie_0p05": False,
                "floor_margin": 0.01,
                "selected_margin": -0.02,
                "margin_change_selected_minus_floor": -0.03,
                "transition_gain_change_selected_minus_floor": 0.01,
                "delta_cosine_change_selected_minus_floor": 0.02,
                "rank_change_selected_minus_floor": 1.0,
                "residual_norm": 0.2,
                "selected_rank": 2.0,
                "floor_rank": 1.0,
            },
            {
                "flip_type": "preserved_correct",
                "floor_near_tie_0p05": False,
                "selected_near_tie_0p05": False,
                "floor_margin": 0.2,
                "selected_margin": 0.21,
                "margin_change_selected_minus_floor": 0.01,
                "transition_gain_change_selected_minus_floor": 0.02,
                "delta_cosine_change_selected_minus_floor": 0.03,
                "rank_change_selected_minus_floor": 0.0,
                "residual_norm": 0.1,
                "selected_rank": 1.0,
                "floor_rank": 1.0,
            },
        ]
    )

    summary = _summarize_retrieval_failure_rows(frame)

    assert summary["broken_count"] == 1.0
    assert summary["preserved_correct_count"] == 1.0
    assert summary["broken_near_tie_fraction"] == 1.0


def test_f052_near_tie_audit_labels_local_margin_failure():
    metrics = {
        "active_gates_pass": True,
        "heldout_local_floor_preserved": False,
        "mean_selected_gap_transition": -0.00001,
        "mean_selected_gap_delta_cosine": 0.0003,
        "mean_selected_gap_recall": -0.01,
    }
    frame = pd.DataFrame(
        [
            {
                "flip_type": "broken_by_residual",
                "floor_top1_correct": True,
                "floor_near_tie_0p02": True,
                "selected_near_tie_0p02": True,
                "floor_near_tie_0p05": True,
                "selected_near_tie_0p05": True,
                "floor_margin": 0.001,
                "selected_margin": -0.001,
                "margin_change_selected_minus_floor": -0.002,
                "transition_gain_change_selected_minus_floor": 0.001,
                "delta_cosine_change_selected_minus_floor": 0.001,
                "residual_norm": 0.05,
            },
            {
                "flip_type": "preserved_correct",
                "floor_top1_correct": True,
                "floor_near_tie_0p02": False,
                "selected_near_tie_0p02": False,
                "floor_near_tie_0p05": False,
                "selected_near_tie_0p05": False,
                "floor_margin": 0.2,
                "selected_margin": 0.21,
                "margin_change_selected_minus_floor": 0.01,
                "transition_gain_change_selected_minus_floor": 0.01,
                "delta_cosine_change_selected_minus_floor": 0.01,
                "residual_norm": 0.05,
            },
        ]
    )

    summary = _f052_descriptor_near_tie_summary(metrics, frame)

    assert summary["diagnostic_label"] == "F052_DESCRIPTOR_FAILURE_IS_NEAR_TIE_RETRIEVAL_MARGIN"
    assert summary["broken_near_tie_0p05_fraction"] == 1.0
    assert summary["broken_continuous_help_fraction"] == 1.0


def test_f053_margin_gate_zero_fallback_without_lower_tail_certificate():
    grid = pd.DataFrame(
        [
            {
                "seed": 1,
                "scale": 0.05,
                "safe": True,
                "gap_transition": 0.001,
                "gap_delta_cosine": 0.001,
                "gap_recall": 0.0,
                "train_broken_count_diagnostic_only": 0.0,
                "train_mean_margin_change_diagnostic_only": 0.001,
            },
            {
                "seed": 2,
                "scale": 0.05,
                "safe": True,
                "gap_transition": 0.001,
                "gap_delta_cosine": 0.001,
                "gap_recall": 0.0,
                "train_broken_count_diagnostic_only": 0.0,
                "train_mean_margin_change_diagnostic_only": 0.001,
            },
        ]
    )

    selected = _f053_select_descriptor_margin_gate(grid)

    assert selected["selected_scale"] == 0.0
    assert selected["selection_label"] == "F053_ZERO_FALLBACK_INSUFFICIENT_TRAIN_MARGIN_CERTIFICATE"
    assert "margin_q10_change" in selected["missing_lower_tail_columns"]


def test_unsafe_margin_mask_flags_near_tie_erosion_and_breaks():
    frame = pd.DataFrame(
        [
            {
                "flip_type": "preserved_correct",
                "floor_near_tie_0p05": True,
                "floor_margin": 0.01,
                "selected_margin": 0.00995,
            },
            {
                "flip_type": "preserved_correct",
                "floor_near_tie_0p05": True,
                "floor_margin": 0.01,
                "selected_margin": 0.009,
            },
            {
                "flip_type": "broken_by_residual",
                "floor_near_tie_0p05": False,
                "floor_margin": 0.2,
                "selected_margin": -0.1,
            },
        ]
    )

    mask = _unsafe_margin_mask(frame)

    assert mask.tolist() == [False, True, True]


def test_rowwise_risk_scores_are_highest_near_unsafe_feature():
    features = np.asarray([[0.0, 0.0], [1.0, 1.0], [4.0, 4.0]], dtype=float)
    unsafe = np.asarray([[1.0, 1.0]], dtype=float)
    mean = np.asarray([[0.0, 0.0]], dtype=float)
    std = np.asarray([[1.0, 1.0]], dtype=float)

    scores = _rowwise_risk_scores(features, unsafe, mean, std)

    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_f058_decision_requires_stable_positive_nonzero_floor_preservation():
    summary = {
        "selected_nonzero_fraction": 1.0,
        "heldout_local_floor_preserved": True,
        "active_gates_pass": True,
        "mean_selected_gap_transition": 0.002,
        "std_selected_gap_transition": 0.001,
        "mean_selected_gap_delta_cosine": 0.003,
        "std_selected_gap_delta_cosine": 0.001,
        "min_selected_gap_transition": 0.0001,
        "min_selected_gap_delta_cosine": 0.0001,
        "min_selected_gap_recall": 0.0,
        "broken_count": 0.0,
    }

    assert _f058_decision_from_summary(summary) == "F058_ROWWISE_TIER2_PASS_NONPROMOTING"

    summary["std_selected_gap_transition"] = 0.004

    assert _f058_decision_from_summary(summary) == "F058_ROWWISE_TIER2_SAFE_BUT_WEAK"


def test_f059_decision_localizes_negative_transition_noise_when_delta_and_recall_hold():
    summary = {
        "f058_broken_count": 0.0,
        "f058_negative_transition_seed_count": 1.0,
        "f058_delta_positive_all": True,
        "f058_recall_preserved_all": True,
        "f058_mean_transition_gap": 0.0001,
        "f058_std_transition_gap": 0.0003,
    }

    assert _f059_decision_from_summary(summary) == "F059_TIER2_WEAKNESS_IS_TRANSITION_GENERALIZATION_NOISE"


def test_floor_direction_projection_keeps_only_positive_floor_aligned_residual():
    floor_delta = np.asarray([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]], dtype=float)
    head_delta = np.asarray([[3.0, 4.0], [0.0, 1.5], [1.0, 1.0]], dtype=float)

    projected = _floor_direction_projected_head_delta(None, floor_delta, head_delta)

    np.testing.assert_allclose(projected[0], [3.0, 0.0])
    np.testing.assert_allclose(projected[1], [0.0, 3.0])
    np.testing.assert_allclose(projected[2], [0.0, 0.0])


def test_residual_cone_interpolates_unprojected_and_projected_residuals():
    floor_delta = np.asarray([[2.0, 0.0]], dtype=float)
    head_delta = np.asarray([[3.0, 4.0]], dtype=float)

    unprojected = _residual_cone_head_delta(None, floor_delta, head_delta, cone_mix=0.0)
    projected = _residual_cone_head_delta(None, floor_delta, head_delta, cone_mix=1.0)
    halfway = _residual_cone_head_delta(None, floor_delta, head_delta, cone_mix=0.5)

    np.testing.assert_allclose(unprojected, head_delta)
    np.testing.assert_allclose(projected, [[3.0, 0.0]])
    np.testing.assert_allclose(halfway, [[3.0, 2.0]])


def test_f060_decision_requires_nonnegative_min_transition_gap():
    summary = {
        "selected_nonzero_fraction": 1.0,
        "heldout_local_floor_preserved": True,
        "active_gates_pass": True,
        "mean_selected_gap_transition": 0.001,
        "min_selected_gap_transition": 0.0,
        "mean_selected_gap_delta_cosine": 0.001,
        "min_selected_gap_recall": 0.0,
        "broken_count": 0.0,
    }

    assert _f060_decision_from_summary(summary) == "F060_FLOOR_DIRECTION_PROJECTION_STABILIZES_TRANSITION_DIAGNOSTIC"

    summary["min_selected_gap_transition"] = -0.0001

    assert _f060_decision_from_summary(summary) == "F060_FLOOR_DIRECTION_PROJECTION_SAFE_BUT_WEAK"


def test_f062_decision_marks_all_seed_oracle_cone_capacity():
    summary = {
        "oracle_nonzero_fraction": 1.0,
        "max_oracle_broken_count": 0.0,
        "mean_oracle_gap_transition": 0.001,
        "mean_oracle_gap_delta_cosine": 0.002,
        "min_oracle_gap_recall": 0.0,
    }

    assert _f062_decision_from_summary(summary) == "F062_ORACLE_CONE_CAPACITY_EXISTS_ALL_SEEDS"

    summary["oracle_nonzero_fraction"] = 0.0

    assert _f062_decision_from_summary(summary) == "F062_ORACLE_CONE_NO_SAFE_NONZERO_CAPACITY"


def test_f063_decision_requires_train_selected_nonzero_and_heldout_floor_preservation():
    summary = {
        "selected_nonzero_fraction": 1.0,
        "heldout_local_floor_preserved": True,
        "active_gates_pass": True,
        "mean_selected_gap_transition": 0.001,
        "mean_selected_gap_delta_cosine": 0.002,
        "min_selected_gap_transition": 0.0001,
        "min_selected_gap_recall": 0.0,
        "broken_count": 0.0,
    }

    assert _f063_decision_from_summary(summary) == "F063_TRAIN_CONE_SELECTOR_SAFE_NONZERO_DIAGNOSTIC"

    summary["min_selected_gap_transition"] = -0.0001

    assert _f063_decision_from_summary(summary) == "F063_TRAIN_CONE_SELECTOR_SAFE_BUT_WEAK"

    summary["selected_nonzero_fraction"] = 0.0
    summary["min_selected_gap_transition"] = 0.0

    assert _f063_decision_from_summary(summary) == "F063_TRAIN_CONE_SELECTOR_ZERO_FALLBACK"


def test_f064_decision_identifies_continuous_transition_overfit_without_retrieval_breaks():
    summary = {
        "train_safe_nonzero_fraction": 1.0,
        "heldout_local_floor_preserved": False,
        "broken_count": 0.0,
        "heldout_mean_gap_transition": -0.001,
    }

    assert _f064_decision_from_summary(summary) == "F064_TRAIN_SELECTOR_OVERFITS_CONTINUOUS_TRANSITION_GAINS"

    summary["broken_count"] = 2.0

    assert _f064_decision_from_summary(summary) == "F064_TRAIN_SELECTOR_BREAKS_RETRIEVAL_ROWS"

    summary["heldout_local_floor_preserved"] = True

    assert _f064_decision_from_summary(summary) == "F064_TRAIN_SELECTOR_MISMATCH_NOT_CONFIRMED"


def test_f065_decision_prefers_zero_fallback_when_inner_action_gate_rejects_residuals():
    summary = {
        "inner_gate_safe_nonzero_fraction": 0.0,
        "heldout_local_floor_preserved": True,
        "selected_nonzero_fraction": 0.0,
    }

    assert _f065_decision_from_summary(summary) == "F065_ACTION_HELDOUT_GATE_REJECTS_RESIDUAL_ZERO_FALLBACK"

    summary.update(
        {
            "inner_gate_safe_nonzero_fraction": 1.0,
            "selected_nonzero_fraction": 1.0,
            "active_gates_pass": True,
            "mean_selected_gap_transition": 0.001,
            "mean_selected_gap_delta_cosine": 0.001,
            "min_selected_gap_transition": 0.0,
            "min_selected_gap_recall": 0.0,
            "broken_count": 0.0,
        }
    )

    assert _f065_decision_from_summary(summary) == "F065_ACTION_HELDOUT_GATE_SAFE_NONZERO_DIAGNOSTIC"


def test_f066_decision_flags_train_action_holdout_optimism():
    summary = {
        "inner_gate_safe_nonzero_fraction": 1.0,
        "heldout_local_floor_preserved": False,
        "broken_count": 0.0,
        "train_action_positive_real_negative_transition_seed_count": 2.0,
        "real_direction_support_gap": 0.1,
    }

    assert _f066_decision_from_summary(summary) == "F066_TRAIN_ACTION_HELDOUT_STILL_OPTIMISTIC_FOR_REAL_HELDOUT"

    summary["train_action_positive_real_negative_transition_seed_count"] = 0.0
    summary["real_direction_support_gap"] = -0.2

    assert _f066_decision_from_summary(summary) == "F066_REAL_HELDOUT_ACTION_SUPPORT_IS_WEAKER_THAN_TRAIN_FOLDS"


def test_f067_decision_closes_residual_selector_family_after_oracle_capacity_and_train_only_failures():
    summary = {
        "oracle_capacity_seen": 1.0,
        "train_only_below_floor_count": 2.0,
        "calibration_mismatch_count": 3.0,
        "safe_nonzero_train_only_count": 2.0,
        "tier2_pass_count": 0.0,
    }

    assert _f067_decision_from_summary(summary) == "F067_CLOSE_RESIDUAL_SELECTOR_FAMILY_PIVOT_TO_DATA_CONTRACT"

    summary["oracle_capacity_seen"] = 0.0

    assert _f067_decision_from_summary(summary) == "F067_RESIDUAL_SELECTOR_SAFE_BUT_TOO_WEAK_FOR_PROMOTION"


def test_f068_decision_identifies_split_contract_with_lower_calibration_optimism():
    summary = {
        "current_abs_transition_optimism": 0.006,
        "best_alternative_abs_transition_optimism": 0.003,
        "best_alternative_policy": "stratified_program",
    }

    assert _f068_decision_from_summary(summary) == "F068_DATA_CONTRACT_REDIRECT_SPLIT_CANDIDATE_IDENTIFIED"

    summary["best_alternative_abs_transition_optimism"] = 0.0045
    summary["current_abs_transition_optimism"] = 0.0048

    assert _f068_decision_from_summary(summary) == "F068_TRAIN_REAL_MISMATCH_PERSISTS_ACROSS_SPLIT_CONTRACTS"


def test_f069_decision_flags_observed_representation_noise_amplification():
    summary = {
        "mean_observed_minus_true_abs_delta_optimism": 0.05,
        "mean_observed_to_true_abs_delta_optimism_ratio": 2.0,
        "mean_true_abs_transition_optimism": 0.01,
    }

    assert _f069_decision_from_summary(summary) == "F069_OBSERVED_RNA_REPRESENTATION_AMPLIFIES_CALIBRATION_MISMATCH"

    summary["mean_observed_minus_true_abs_delta_optimism"] = 0.0
    summary["mean_observed_to_true_abs_delta_optimism_ratio"] = 1.0
    summary["mean_true_abs_transition_optimism"] = 0.03

    assert _f069_decision_from_summary(summary) == "F069_TRUE_ZBIO_ALSO_MISMATCHED_DATA_CONTRACT_DOMINATES"


def test_f070_decision_prioritizes_train_only_denoising_before_oracle_targets():
    summary = {
        "observed_mean_abs_delta_optimism": 0.10,
        "observed_mean_abs_transition_optimism": 0.08,
        "best_candidate_mean_abs_delta_optimism": 0.06,
        "best_candidate_mean_abs_transition_optimism": 0.05,
        "clean_rna_mean_abs_delta_optimism": 0.04,
        "clean_rna_mean_abs_transition_optimism": 0.03,
        "true_z_bio_mean_abs_delta_optimism": 0.02,
        "true_z_bio_mean_abs_transition_optimism": 0.01,
    }

    assert _f070_decision_from_summary(summary) == "F070_TRAIN_ONLY_DENOISING_REDUCES_REPRESENTATION_MISMATCH"

    summary["best_candidate_mean_abs_delta_optimism"] = 0.09
    summary["best_candidate_mean_abs_transition_optimism"] = 0.07

    assert _f070_decision_from_summary(summary) == "F070_CLEAN_RNA_ORACLE_REDUCES_MISMATCH_NEEDS_LEARNED_DENOISING"

    summary["clean_rna_mean_abs_delta_optimism"] = 0.09
    summary["clean_rna_mean_abs_transition_optimism"] = 0.07

    assert _f070_decision_from_summary(summary) == "F070_TRUE_ZBIO_ONLY_REDUCES_MISMATCH_RNA_OBSERVATION_LIMIT"


def test_f071_decision_requires_recovery_and_mismatch_reduction():
    summary = {
        "observed_mean_abs_delta_optimism": 0.10,
        "observed_mean_abs_transition_optimism": 0.08,
        "best_recovered_mean_abs_delta_optimism": 0.06,
        "best_recovered_mean_abs_transition_optimism": 0.05,
        "best_recovered_eval_zbio_cosine": 0.80,
        "best_image_fusion_delta_optimism_gain": 0.01,
    }

    assert _f071_decision_from_summary(summary) == "F071_CROSS_MODAL_ZBIO_RECOVERY_REDUCES_MISMATCH"

    summary["best_image_fusion_delta_optimism_gain"] = 0.0

    assert _f071_decision_from_summary(summary) == "F071_RNA_ZBIO_RECOVERY_REDUCES_MISMATCH"

    summary["best_recovered_mean_abs_delta_optimism"] = 0.09
    summary["best_recovered_mean_abs_transition_optimism"] = 0.07

    assert _f071_decision_from_summary(summary) == "F071_ZBIO_RECOVERABLE_BUT_TRANSITION_MISMATCH_REMAINS"

    summary["best_recovered_eval_zbio_cosine"] = 0.4

    assert _f071_decision_from_summary(summary) == "F071_ZBIO_NOT_RECOVERABLE_FROM_CURRENT_OBSERVATIONS"


def test_f072_decision_separates_image_latent_and_rna_bridge_signal():
    summary = {
        "observed_mean_abs_delta_optimism": 0.10,
        "observed_mean_abs_transition_optimism": 0.08,
        "image_pca_mean_abs_delta_optimism": 0.05,
        "image_pca_mean_abs_transition_optimism": 0.05,
        "rna_to_image_mean_abs_delta_optimism": 0.055,
        "rna_to_image_mean_abs_transition_optimism": 0.055,
        "rna_to_image_eval_image_latent_cosine": 0.80,
    }

    assert _f072_decision_from_summary(summary) == "F072_RNA_TO_IMAGE_LATENT_PRESERVES_NONORACLE_IMAGE_SIGNAL"

    summary["rna_to_image_eval_image_latent_cosine"] = 0.4

    assert _f072_decision_from_summary(summary) == "F072_IMAGE_LATENT_NONORACLE_REDUCES_MISMATCH"

    summary["image_pca_mean_abs_transition_optimism"] = 0.07

    assert _f072_decision_from_summary(summary) == "F072_IMAGE_LATENT_HELPS_DELTA_BUT_NOT_TRANSITION"


def test_f073_decision_checks_image_teacher_preservation_and_underfit():
    summary = {
        "observed_mean_abs_delta_optimism": 0.10,
        "observed_mean_abs_transition_optimism": 0.08,
        "image_pca_mean_abs_delta_optimism": 0.05,
        "image_pca_mean_abs_transition_optimism": 0.02,
        "candidate_mean_abs_delta_optimism": 0.055,
        "candidate_mean_abs_transition_optimism": 0.025,
        "candidate_eval_image_latent_cosine": 0.75,
        "candidate_mean_effective_rank": 4.0,
    }

    assert _f073_decision_from_summary(summary) == "F073_RNA_STUDENT_PRESERVES_IMAGE_TEACHER_SIGNAL"

    summary["candidate_mean_abs_transition_optimism"] = 0.055

    assert _f073_decision_from_summary(summary) == "F073_RNA_STUDENT_REDUCES_MISMATCH_BUT_BELOW_IMAGE_TEACHER"

    summary["candidate_eval_image_latent_cosine"] = 0.30

    assert _f073_decision_from_summary(summary) == "F073_RNA_STUDENT_UNDERFITS_IMAGE_TEACHER"


def test_f074_decision_localizes_target_underfit_or_perturbation_loss():
    summary = {
        "candidate_mean_image_cosine": 0.30,
        "mean_train_prediction_target_cosine": 0.80,
        "candidate_minus_image_perturbation_probe_gap": -0.10,
        "candidate_minus_image_recall_gap": -0.10,
        "candidate_minus_image_delta_cosine_gap": -0.05,
    }

    assert _f074_decision_from_summary(summary) == "F074_FAILURE_IS_IMAGE_TARGET_UNDERFIT"

    summary["candidate_mean_image_cosine"] = 0.70
    summary["candidate_minus_image_perturbation_probe_gap"] = -0.30
    summary["candidate_minus_image_recall_gap"] = -0.30

    assert _f074_decision_from_summary(summary) == "F074_FAILURE_IS_PERTURBATION_STRUCTURE_LOSS"

    summary["candidate_minus_image_perturbation_probe_gap"] = -0.10
    summary["candidate_minus_image_delta_cosine_gap"] = -0.20
    summary["candidate_minus_image_recall_gap"] = -0.15

    assert _f074_decision_from_summary(summary) == "F074_FAILURE_IS_TRANSITION_GEOMETRY_DISTORTION"


def test_f075_decision_checks_relational_structure_repair():
    summary = {
        "candidate_eval_image_latent_cosine": 0.75,
        "candidate_mean_effective_rank": 4.0,
        "candidate_mean_abs_delta_optimism": 0.052,
        "candidate_mean_abs_transition_optimism": 0.025,
        "image_pca_mean_abs_delta_optimism": 0.05,
        "image_pca_mean_abs_transition_optimism": 0.02,
        "candidate_minus_image_recall_gap": -0.05,
        "candidate_minus_image_delta_cosine_gap": -0.04,
        "candidate_minus_image_perturbation_probe_gap": -0.05,
        "candidate_minus_f073_recall_gap": 0.12,
        "candidate_minus_f073_delta_cosine_gap": 0.11,
        "candidate_minus_f073_perturbation_probe_gap": 0.12,
    }

    assert _f075_decision_from_summary(summary) == "F075_RELATIONAL_RNA_STUDENT_PRESERVES_IMAGE_TEACHER_SIGNAL"

    summary["candidate_mean_abs_transition_optimism"] = 0.06
    summary["candidate_minus_image_recall_gap"] = -0.25

    assert _f075_decision_from_summary(summary) == "F075_RELATIONAL_OBJECTIVE_REPAIRS_STRUCTURE_BUT_WEAK"

    summary["candidate_eval_image_latent_cosine"] = 0.40

    assert _f075_decision_from_summary(summary) == "F075_RELATIONAL_RNA_STUDENT_UNDERFITS_IMAGE_TEACHER"


def test_f076_decision_checks_condition_centroid_teacher_repair():
    summary = {
        "candidate_eval_image_latent_cosine": 0.74,
        "candidate_mean_effective_rank": 5.0,
        "candidate_mean_abs_delta_optimism": 0.052,
        "candidate_mean_abs_transition_optimism": 0.025,
        "image_pca_mean_abs_delta_optimism": 0.05,
        "image_pca_mean_abs_transition_optimism": 0.02,
        "candidate_minus_image_recall_gap": -0.04,
        "candidate_minus_image_delta_cosine_gap": -0.05,
        "candidate_minus_image_perturbation_probe_gap": -0.05,
        "candidate_minus_f075_recall_gap": 0.12,
        "candidate_minus_f075_delta_cosine_gap": 0.11,
        "candidate_minus_f075_perturbation_probe_gap": 0.12,
    }

    assert _f076_decision_from_summary(summary) == "F076_CONDITION_CENTROID_TEACHER_PRESERVES_IMAGE_SIGNAL"

    summary["candidate_mean_abs_transition_optimism"] = 0.06
    summary["candidate_minus_image_recall_gap"] = -0.25

    assert _f076_decision_from_summary(summary) == "F076_CONDITION_CENTROID_TEACHER_REPAIRS_STRUCTURE_BUT_WEAK"

    summary["candidate_eval_image_latent_cosine"] = 0.40

    assert _f076_decision_from_summary(summary) == "F076_CONDITION_CENTROID_TEACHER_UNDERFITS_IMAGE_SIGNAL"


def test_f077_decision_checks_source_program_action_repair():
    summary = {
        "candidate_eval_image_latent_cosine": 0.68,
        "candidate_mean_effective_rank": 4.0,
        "candidate_mean_abs_delta_optimism": 0.060,
        "candidate_mean_abs_transition_optimism": 0.035,
        "image_pca_mean_abs_delta_optimism": 0.05,
        "image_pca_mean_abs_transition_optimism": 0.02,
        "observed_mean_abs_delta_optimism": 0.09,
        "candidate_minus_image_recall_gap": -0.05,
        "candidate_minus_image_delta_cosine_gap": -0.05,
        "candidate_minus_image_perturbation_probe_gap": -0.05,
        "candidate_minus_f076_recall_gap": 0.12,
        "candidate_minus_f076_delta_cosine_gap": 0.11,
        "candidate_minus_f076_perturbation_probe_gap": 0.12,
    }

    assert _f077_decision_from_summary(summary) == "F077_SOURCE_PROGRAM_ACTION_PRESERVES_IMAGE_SIGNAL"

    summary["candidate_mean_abs_transition_optimism"] = 0.07
    summary["candidate_minus_image_recall_gap"] = -0.25

    assert _f077_decision_from_summary(summary) == "F077_SOURCE_PROGRAM_ACTION_REPAIRS_STRUCTURE_BUT_WEAK"

    summary["candidate_eval_image_latent_cosine"] = 0.20

    assert _f077_decision_from_summary(summary) == "F077_SOURCE_PROGRAM_ACTION_UNDERFITS_IMAGE_SIGNAL"


def test_f078_decision_localizes_source_program_weak_pass():
    summary = {
        "candidate_minus_image_recall_gap": 0.0,
        "candidate_minus_image_transition_gap": -0.22,
        "candidate_minus_image_delta_cosine_gap": -0.08,
        "candidate_minus_image_perturbation_probe_gap": -0.44,
        "candidate_minus_image_effective_rank_gap": -4.0,
        "candidate_mean_real_transition": 0.01,
    }

    assert _f078_decision_from_summary(summary) == "F078_SOURCE_PROGRAM_RECALL_WITH_LOW_TRANSITION_AND_ACTION_SIGNAL"

    summary["candidate_minus_image_recall_gap"] = -0.20

    assert _f078_decision_from_summary(summary) == "F078_SOURCE_PROGRAM_LOW_RANK_PROTOTYPE_LIMIT"

    summary["candidate_minus_image_effective_rank_gap"] = -1.0
    summary["candidate_minus_image_delta_cosine_gap"] = -0.20

    assert _f078_decision_from_summary(summary) == "F078_SOURCE_PROGRAM_DELTA_DIRECTION_FAILURE"


def test_f079_decision_checks_delta_rank_repair_quality():
    summary = {
        "candidate_mean_real_transition": 0.07,
        "candidate_mean_real_delta_cosine": 0.91,
        "candidate_mean_real_recall_at_1": 1.0,
        "candidate_mean_effective_rank": 3.4,
        "candidate_mean_perturbation_probe": 0.25,
        "candidate_minus_f077_transition_gap": 0.04,
        "candidate_minus_f077_delta_cosine_gap": 0.02,
        "candidate_minus_f077_effective_rank_gap": 0.7,
        "candidate_minus_image_transition_gap": -0.10,
        "candidate_minus_image_perturbation_probe_gap": -0.20,
    }

    assert _f079_decision_from_summary(summary) == "F079_SOURCE_DELTA_RANK_REPAIR_READY_FOR_WRAPPER"

    summary["candidate_mean_real_transition"] = 0.045
    summary["candidate_minus_image_perturbation_probe_gap"] = -0.35

    assert _f079_decision_from_summary(summary) == "F079_SOURCE_DELTA_RANK_REPAIRS_TRANSITION_BUT_WEAK"

    summary["candidate_mean_effective_rank"] = 1.5

    assert _f079_decision_from_summary(summary) == "F079_SOURCE_DELTA_RANK_COLLAPSE"


def test_f080_decision_checks_full_jepa_wrapper_quality():
    summary = {
        "max_identity_violation": 0.0,
        "max_leakage_flag": 0.0,
        "mean_image_teacher_transition_improvement": 0.11,
        "mean_image_teacher_delta_cosine": 0.88,
        "mean_image_teacher_recall_at_1": 1.0,
        "mean_image_teacher_delta_rank": 4.0,
        "mean_image_teacher_minus_f079_transition_gap": -0.02,
        "mean_rna_to_image_recall_at_1": 0.30,
        "mean_image_to_rna_recall_at_1": 0.35,
        "mean_target_image_cosine": 0.42,
        "mean_transition_image_cosine": 0.45,
    }

    assert _f080_decision_from_summary(summary) == "F080_FULL_JEPA_WRAPPER_TIER1_PASS_NONPROMOTING"

    summary["mean_image_teacher_transition_improvement"] = 0.07
    summary["mean_rna_to_image_recall_at_1"] = 0.05

    assert _f080_decision_from_summary(summary) == "F080_FULL_JEPA_WRAPPER_PRESERVES_WEAK_TRANSITION_SIGNAL"

    summary["mean_image_teacher_transition_improvement"] = 0.01
    summary["mean_image_teacher_minus_f079_transition_gap"] = -0.10

    assert _f080_decision_from_summary(summary) == "F080_FULL_JEPA_WRAPPER_LOSES_F079_TRANSITION_SIGNAL"

    summary["max_identity_violation"] = 1.0

    assert _f080_decision_from_summary(summary) == "F080_FULL_JEPA_WRAPPER_IDENTITY_OR_LEAKAGE_FAIL"


def test_f081_decision_checks_delta_calibrator_quality():
    summary = {
        "max_identity_violation": 0.0,
        "max_leakage_flag": 0.0,
        "mean_calibrated_transition_improvement": 0.12,
        "mean_calibrated_delta_cosine": 0.87,
        "mean_calibrated_recall_at_1": 1.0,
        "mean_calibrated_delta_rank": 5.0,
        "mean_raw_image_teacher_transition_improvement": 0.13,
        "mean_raw_image_teacher_delta_cosine": 0.78,
        "mean_train_calibrated_delta_cosine": 0.90,
    }

    assert _f081_decision_from_summary(summary) == "F081_DELTA_CALIBRATED_JEPA_TIER1_PASS_NONPROMOTING"

    summary["mean_calibrated_delta_cosine"] = 0.84

    assert _f081_decision_from_summary(summary) == "F081_DELTA_CALIBRATOR_REPAIRS_DIRECTION_BUT_WEAK"

    summary["mean_calibrated_delta_cosine"] = 0.79

    assert _f081_decision_from_summary(summary) == "F081_DELTA_CALIBRATOR_DOES_NOT_REPAIR_DIRECTION"

    summary["mean_train_calibrated_delta_cosine"] = 1.0
    summary["mean_calibrated_delta_cosine"] = 0.75

    assert _f081_decision_from_summary(summary) == "F081_DELTA_CALIBRATOR_OVERFITS_TRAIN_DIRECTION"


def test_f082_decision_checks_fresh_seed_stability():
    summary = {
        "max_identity_violation": 0.0,
        "max_leakage_flag": 0.0,
        "mean_calibrated_transition_improvement": 0.16,
        "mean_calibrated_delta_cosine": 0.90,
        "mean_calibrated_recall_at_1": 0.97,
        "min_calibrated_transition_improvement": 0.02,
        "min_calibrated_delta_cosine": 0.82,
        "min_calibrated_recall_at_1": 0.93,
        "std_calibrated_transition_improvement": 0.04,
    }

    assert _f082_decision_from_summary(summary) == "F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN"

    summary["min_calibrated_delta_cosine"] = 0.68

    assert _f082_decision_from_summary(summary) == "F082_DELTA_CALIBRATED_TIER2_SEED_INSTABILITY"

    summary["min_calibrated_delta_cosine"] = 0.78
    summary["min_calibrated_recall_at_1"] = 0.80

    assert _f082_decision_from_summary(summary) == "F082_DELTA_CALIBRATED_TIER2_MEAN_PASS_BUT_NO_REGRESSION_GAP"


def test_f083_decision_checks_tier3_preflight_gates():
    summary = {
        "max_identity_violation": 0.0,
        "max_leakage_flag": 0.0,
        "transition_floor_pass": True,
        "delta_floor_pass": True,
        "recall_floor_pass": True,
        "rank_floor_pass": True,
        "magnitude_floor_pass": True,
        "cross_modal_gate_pass": True,
        "paired_external_validator_available": True,
    }

    assert _f083_decision_from_summary(summary) == "F083_TIER3_PREFLIGHT_APPROVED_FOR_LOCKED_TIER3_RUN"

    summary["rank_floor_pass"] = False
    summary["paired_external_validator_available"] = False

    assert _f083_decision_from_summary(summary) == "F083_TIER3_PREFLIGHT_BLOCKED_BY_RANK_AND_VALIDATOR_GAPS"

    summary["rank_floor_pass"] = True

    assert _f083_decision_from_summary(summary) == "F083_TIER3_PREFLIGHT_NEEDS_PAIRED_EXTERNAL_VALIDATOR"

    summary["max_leakage_flag"] = 1.0

    assert _f083_decision_from_summary(summary) == "F083_TIER3_PREFLIGHT_IDENTITY_OR_LEAKAGE_FAIL"


def test_f084_decision_checks_rank_floor_comparability():
    summary = {
        "locked_delta_rank_floor": 10.0,
        "mean_target_image_effective_rank": 7.0,
        "mean_image_teacher_delta_rank": 8.0,
        "mean_calibrated_delta_rank": 7.5,
        "mean_raw_direct_delta_rank": 8.5,
    }

    assert _f084_decision_from_summary(summary) == "F084_LOCKED_RANK_FLOOR_EXCEEDS_IMAGE_TEACHER_CEILING"

    summary["mean_target_image_effective_rank"] = 12.0
    summary["mean_image_teacher_delta_rank"] = 11.0

    assert _f084_decision_from_summary(summary) == "F084_RANK_REPAIR_READY_IN_CURRENT_TEACHER_SPACE"

    summary["mean_raw_direct_delta_rank"] = 10.5
    summary["mean_image_teacher_delta_rank"] = 9.0

    assert _f084_decision_from_summary(summary) == "F084_DELTA_CALIBRATOR_LOSES_REPAIRABLE_RANK"

    summary["mean_calibrated_delta_rank"] = 10.1

    assert _f084_decision_from_summary(summary) == "F084_RANK_FLOOR_ALREADY_PRESERVED"


def test_f085_decision_checks_current_latent_floor_registry():
    summary = {
        "candidate_preserves_current_floor": True,
        "current_registry_rank_supported": True,
        "locked_rank_floor_incomparable": True,
    }

    assert _f085_decision_from_summary(summary) == "F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE"

    summary["current_registry_rank_supported"] = False

    assert _f085_decision_from_summary(summary) == "F085_CURRENT_LATENT_FLOOR_REVEALS_REMAINING_RANK_GAP"

    summary["candidate_preserves_current_floor"] = False

    assert _f085_decision_from_summary(summary) == "F085_CANDIDATE_BELOW_CURRENT_IMAGE_TEACHER_FLOOR"


def test_f086_decision_requires_local_validator_for_locked_tier3_run():
    summary = {
        "identity_or_leakage_fail": False,
        "synthetic_current_registry_gates_pass": True,
        "local_paired_validator_available": False,
        "future_paired_validator_candidate_found": True,
    }

    assert _f086_decision_from_summary(summary) == "F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED"

    summary["local_paired_validator_available"] = True

    assert _f086_decision_from_summary(summary) == "F086_CURRENT_REGISTRY_TIER3_PREFLIGHT_APPROVED_FOR_LOCKED_RUN"

    summary["synthetic_current_registry_gates_pass"] = False

    assert _f086_decision_from_summary(summary) == "F086_CURRENT_REGISTRY_TIER3_SYNTHETIC_GATES_FAIL"

    summary["identity_or_leakage_fail"] = True

    assert _f086_decision_from_summary(summary) == "F086_CURRENT_REGISTRY_TIER3_IDENTITY_OR_LEAKAGE_FAIL"


def test_f087_decision_tracks_scgenescope_manifest_readiness():
    summary = {
        "root_exists": False,
        "manifest_valid": False,
        "paired_condition_count": 0,
    }

    assert _f087_decision_from_summary(summary) == "F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL"

    summary["root_exists"] = True

    assert _f087_decision_from_summary(summary) == "F087_SCGENESCOPE_ROOT_PRESENT_MANIFEST_NEEDED"

    summary["manifest_valid"] = True
    summary["paired_condition_count"] = 2

    assert _f087_decision_from_summary(summary) == "F087_SCGENESCOPE_LOCAL_MANIFEST_READY_FOR_TIER3_DRY_RUN"


def test_f088_decision_prioritizes_light_metadata_then_large_feature_payloads():
    summary = {
        "remote_api_accessible": True,
        "light_manifest_candidate_found": False,
        "paired_feature_payloads_found": True,
    }

    assert _f088_decision_from_summary(summary) == "F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE"

    summary["light_manifest_candidate_found"] = True

    assert _f088_decision_from_summary(summary) == "F088_SCGENESCOPE_LIGHT_METADATA_FOUND_READY_FOR_MANIFEST_BUILD"

    summary["remote_api_accessible"] = False

    assert _f088_decision_from_summary(summary) == "F088_SCGENESCOPE_REMOTE_API_UNAVAILABLE"


def test_f089_decision_requires_croissant_and_split_contract():
    summary = {
        "supplement_downloaded": True,
        "croissant_found": True,
        "replicate_split_contract_found": True,
        "code_readme_found": True,
    }

    assert _f089_decision_from_summary(summary) == "F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED"

    summary["replicate_split_contract_found"] = False

    assert _f089_decision_from_summary(summary) == "F089_SCGENESCOPE_SUPPLEMENT_CODE_FOUND_METADATA_INCOMPLETE"

    summary["supplement_downloaded"] = False

    assert _f089_decision_from_summary(summary) == "F089_SCGENESCOPE_SUPPLEMENT_UNAVAILABLE"


def test_f090_decision_requires_valid_contract_ready_pairs_and_no_payload_download():
    summary = {
        "adapter_contract_valid": True,
        "all_dry_run_condition_pairs_ready": True,
        "payload_download_attempted": False,
    }

    assert _f090_decision_from_summary(summary) == "F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE"

    summary["all_dry_run_condition_pairs_ready"] = False

    assert _f090_decision_from_summary(summary) == "F090_SCGENESCOPE_DRY_RUN_PAIRING_FAIL"

    summary["adapter_contract_valid"] = False
    summary["all_dry_run_condition_pairs_ready"] = True

    assert _f090_decision_from_summary(summary) == "F090_SCGENESCOPE_CROISSANT_CONTRACT_INVALID"

    summary["adapter_contract_valid"] = True
    summary["payload_download_attempted"] = True

    assert _f090_decision_from_summary(summary) == "F090_SCGENESCOPE_CONTRACT_VIOLATED_PAYLOAD_DOWNLOAD"


def test_f091_decision_requires_storage_backed_io_and_low_compute_gates():
    summary = {
        "f090_contract_valid": True,
        "feature_pair_candidate_count": 1,
        "payload_download_attempted": False,
        "storage_gate_pass": True,
        "backed_io_gate_pass": True,
        "low_compute_payload_gate_pass": True,
    }

    assert _f091_decision_from_summary(summary) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN"

    blocked = dict(summary)
    blocked["f090_contract_valid"] = False
    assert _f091_decision_from_summary(blocked) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_BLOCKED_CONTRACT_INVALID"

    blocked = dict(summary)
    blocked["feature_pair_candidate_count"] = 0
    assert _f091_decision_from_summary(blocked) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_NO_PAIRED_FEATURES"

    blocked = dict(summary)
    blocked["storage_gate_pass"] = False
    assert _f091_decision_from_summary(blocked) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_STORAGE_GATE_FAIL"

    blocked = dict(summary)
    blocked["backed_io_gate_pass"] = False
    assert _f091_decision_from_summary(blocked) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_BACKED_IO_GATE_FAIL"

    blocked = dict(summary)
    blocked["low_compute_payload_gate_pass"] = False
    assert _f091_decision_from_summary(blocked) == "F091_SCGENESCOPE_FEATURE_PREFLIGHT_TOO_LARGE_FOR_LOW_COMPUTE"


def test_f092_decision_requires_download_backed_open_and_obs_contract():
    summary = {
        "f091_preflight_approved": True,
        "payload_cap_violation": False,
        "download_error": False,
        "all_expected_files_present": True,
        "backed_open_error": False,
        "obs_contract_pass": True,
    }

    assert _f092_decision_from_summary(summary) == "F092_SCGENESCOPE_OBS_ONLY_BACKED_DRY_RUN_PASS"

    blocked = dict(summary)
    blocked["f091_preflight_approved"] = False
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_BLOCKED_PREFLIGHT_NOT_APPROVED"

    blocked = dict(summary)
    blocked["payload_cap_violation"] = True
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_VIOLATED_PAYLOAD_CAP"

    blocked = dict(summary)
    blocked["download_error"] = True
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_DOWNLOAD_OR_ACCESS_FAIL"

    blocked = dict(summary)
    blocked["all_expected_files_present"] = False
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_INCOMPLETE_LOCAL_FILES"

    blocked = dict(summary)
    blocked["backed_open_error"] = True
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_BACKED_OPEN_FAIL"

    blocked = dict(summary)
    blocked["obs_contract_pass"] = False
    assert _f092_decision_from_summary(blocked) == "F092_SCGENESCOPE_OBS_DRY_RUN_OBS_CONTRACT_INCOMPLETE"


def test_hard_escalation_active_and_status_reserve_helpers(tmp_path):
    assert not _hard_escalation_active(tmp_path)
    (tmp_path / "HARD_ESCALATION_REPORT.md").write_text("- label: `HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED`\n")
    assert _hard_escalation_active(tmp_path)

    reserve_path = tmp_path / "reserve.bin"
    ok, error = _create_status_write_reserve(reserve_path, size_bytes=1024)

    assert ok
    assert error == ""
    assert reserve_path.exists()
    assert reserve_path.stat().st_size == 1024

    _release_status_write_reserve(reserve_path)

    assert not reserve_path.exists()
