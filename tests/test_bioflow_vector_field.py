import torch

from perturb_jepa.models.bioflow_jepa import ActionConditionedVectorField, FlowIntegrator, tangent_project


def test_vector_field_output_shape():
    vf = ActionConditionedVectorField(z_dim=6, action_dim=4, hidden_dim=8)
    z = torch.randn(3, 6)
    tau = torch.rand(3)
    action = torch.randn(3, 4)

    out = vf(z, tau, action, context=z)

    assert out.shape == z.shape


def test_zero_velocity_integration_returns_normalized_source():
    vf = ActionConditionedVectorField(z_dim=6, action_dim=4, hidden_dim=8)
    for parameter in vf.parameters():
        parameter.data.zero_()
    integrator = FlowIntegrator(vf, steps=3, use_tangent_projection=True, normalize_endpoint=True)
    source = torch.nn.functional.normalize(torch.randn(3, 6), dim=-1)
    action = torch.randn(3, 4)

    out = integrator(source, action, context=source)

    assert torch.allclose(out, source, atol=1.0e-6)


def test_tangent_projection_is_orthogonal_to_source():
    source = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
    velocity = torch.randn(5, 7)

    projected = tangent_project(velocity, source)

    assert torch.allclose((projected * source).sum(dim=-1), torch.zeros(5), atol=1.0e-6)
