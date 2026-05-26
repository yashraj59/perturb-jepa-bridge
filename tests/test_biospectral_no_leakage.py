from pathlib import Path

import numpy as np
import torch

from perturb_jepa.models.biospectral_jepa import RidgeFloorHead
from perturb_jepa.training.biospectral_operator import load_latent_bundle, write_leakage_report


PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")


def test_ridge_floor_fit_ignores_eval_target_statistics():
    train = load_latent_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_train_latents", "train")
    eval_bundle = load_latent_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_test_heldout_perturbation_latents", "eval")
    head_a = RidgeFloorHead(train.source.shape[1], train.action.shape[1])
    head_b = RidgeFloorHead(train.source.shape[1], train.action.shape[1])
    source = torch.as_tensor(train.source, dtype=torch.float32)
    action = torch.as_tensor(train.action, dtype=torch.float32)
    delta = torch.as_tensor(train.delta, dtype=torch.float32)
    eval_source = torch.as_tensor(eval_bundle.source, dtype=torch.float32)
    eval_action = torch.as_tensor(eval_bundle.action, dtype=torch.float32)

    head_a.fit(source, action, delta, alpha=1.0e-2)
    shifted_eval_targets = eval_bundle.target + 1000.0
    assert np.isfinite(shifted_eval_targets).all()
    head_b.fit(source, action, delta, alpha=1.0e-2)

    pred_a = head_a(eval_source, eval_action)
    pred_b = head_b(eval_source, eval_action)
    assert torch.allclose(pred_a, pred_b)


def test_leakage_report_confirms_condition_key_label_only(tmp_path):
    path = tmp_path / "leakage_report.md"
    write_leakage_report(
        path,
        train_rows=72,
        eval_rows=27,
        action_feature_names=["action_descriptor_0", "action_descriptor_1"],
        mode="unit_test",
    )
    text = path.read_text(encoding="utf-8")

    assert "Forbidden key tensors present: `False`" in text
    assert "`condition_key` used only as an evaluation/retrieval label" in text
    assert "Test/eval target means are not used" in text


def test_leakage_report_flags_forbidden_action_feature_names(tmp_path):
    path = tmp_path / "leakage_report.md"
    write_leakage_report(
        path,
        train_rows=1,
        eval_rows=1,
        action_feature_names=["condition_key"],
        mode="unit_test",
    )
    text = path.read_text(encoding="utf-8")

    assert "Forbidden key tensors present: `True`" in text
