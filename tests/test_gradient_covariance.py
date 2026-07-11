import pytest
import torch
import torch.nn as nn

from src.experiments.gradient_covariance import compute_gradient_covariance, flatten_gradients


def test_flatten_gradients_concatenates_in_parameter_order():
    model = nn.Linear(2, 1, bias=True)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()

    flat = flatten_gradients(model)

    expected_numel = model.weight.numel() + model.bias.numel()
    assert flat.shape == (expected_numel,)
    assert torch.equal(flat[: model.weight.numel()], model.weight.grad.view(-1))


def test_identical_samples_have_cosine_similarity_one():
    torch.manual_seed(0)
    model = nn.Linear(3, 1, bias=False)
    loss_fn = nn.MSELoss()

    x = torch.randn(1, 3)
    y = torch.randn(1, 1)
    inputs = torch.cat([x, x], dim=0)
    targets = torch.cat([y, y], dim=0)

    covariance = compute_gradient_covariance(model, loss_fn, inputs, targets)

    assert covariance.shape == (2, 2)
    assert covariance[0, 1].item() == pytest.approx(1.0, abs=1e-4)
    assert covariance.diag()[0].item() == pytest.approx(1.0, abs=1e-4)
