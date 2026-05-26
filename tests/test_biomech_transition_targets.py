import torch

from test_biomech_jepa_model import _model_and_batch


def test_delta_teacher_is_stop_gradient_target_minus_control_teacher():
    model, batch = _model_and_batch(enable_population=False)
    outputs = model.forward_jepa(batch)

    expected = outputs["z_target_teacher_bio"] - outputs["z_control_teacher_bio"]

    assert not outputs["delta_teacher"].requires_grad
    assert torch.allclose(outputs["delta_teacher"], expected)
    assert outputs["delta_teacher_effective_rank"].ndim == 0
