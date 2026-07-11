import pytest
import torch
import torch.nn as nn

from src.experiments.plasticity import (
    PlasticityProbeConfig,
    default_probe_optimizer,
    make_random_function_targets,
    measure_plasticity_loss,
    train_probe_task,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_make_random_function_targets_matches_input_batch_shape():
    model = _tiny_model()
    random_model = _tiny_model()
    inputs = torch.randn(6, 4)

    targets = make_random_function_targets(model, random_model, inputs)

    assert targets.shape == (6, 4)


def test_train_probe_task_reduces_loss_on_a_fixed_target():
    torch.manual_seed(0)
    model = _tiny_model()
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 4)
    config = PlasticityProbeConfig(steps=50, batch_size=None)

    result = train_probe_task(model, inputs, targets, default_probe_optimizer(lr=1e-2), config)

    assert result.final_loss < result.initial_loss


def test_measure_plasticity_loss_computes_baseline_when_given_a_baseline_model():
    torch.manual_seed(0)
    model = _tiny_model()
    inputs = torch.randn(8, 4)
    config = PlasticityProbeConfig(steps=5, num_tasks=2, batch_size=None)

    result = measure_plasticity_loss(
        model,
        _tiny_model,
        inputs,
        default_probe_optimizer(lr=1e-3),
        config,
        baseline_model=model,
    )

    assert result.baseline_probe_loss is not None
    assert result.plasticity_loss == pytest.approx(
        result.probe_loss - result.baseline_probe_loss
    )


def test_measure_plasticity_loss_skips_baseline_when_not_provided():
    torch.manual_seed(0)
    model = _tiny_model()
    inputs = torch.randn(8, 4)
    config = PlasticityProbeConfig(steps=5, num_tasks=1, batch_size=None)

    result = measure_plasticity_loss(
        model, _tiny_model, inputs, default_probe_optimizer(lr=1e-3), config
    )

    assert result.baseline_probe_loss is None
    assert result.plasticity_loss is None
