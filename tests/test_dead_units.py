import torch
import torch.nn as nn

from src.experiments.dead_units import compute_dead_units, dead_unit_statistics


def test_dead_unit_statistics_counts_units_never_positive():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    with torch.no_grad():
        # Unit 0 always fires (bias +10), unit 1 always dead (bias -10),
        # unit 2 depends on the input sign.
        model[0].weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
        model[0].bias.copy_(torch.tensor([10.0, -10.0, 0.0]))

    probe_batch = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]])

    stats = dead_unit_statistics(model, probe_batch)

    assert stats["total_units"] == 3.0
    assert stats["dead_units"] == 1.0
    assert stats["dead_fraction"] == 1.0 / 3.0


def test_compute_dead_units_return_fraction_flag():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
        model[0].bias.copy_(torch.tensor([-10.0, -10.0]))  # both units dead

    probe_batch = torch.randn(5, 2)

    assert compute_dead_units(model, probe_batch, return_fraction=False) == 2.0
    assert compute_dead_units(model, probe_batch, return_fraction=True) == 1.0


def test_dead_unit_statistics_handles_no_relu_modules():
    model = nn.Linear(2, 2)
    stats = dead_unit_statistics(model, torch.randn(4, 2))

    assert stats == {"dead_units": 0.0, "total_units": 0.0, "dead_fraction": 0.0}
