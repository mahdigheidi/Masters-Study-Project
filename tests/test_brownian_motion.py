import copy

import pytest
import torch
import torch.nn as nn

from src.experiments.brownian_motion import brownian_update, compute_update_norm


def test_brownian_update_matches_requested_global_step_norm():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    before = copy.deepcopy(model)

    brownian_update(model, step_norm=0.5)

    assert compute_update_norm(before, model).item() == pytest.approx(0.5, rel=1e-4)


def test_brownian_update_is_noop_for_nonpositive_step_norm():
    model = nn.Linear(3, 3)
    before = copy.deepcopy(model)

    brownian_update(model, step_norm=0.0)

    assert compute_update_norm(before, model).item() == 0.0


def test_compute_update_norm_matches_manual_l2_distance():
    model_before = nn.Linear(2, 2, bias=False)
    model_after = copy.deepcopy(model_before)
    with torch.no_grad():
        model_after.weight.add_(torch.tensor([[3.0, 0.0], [4.0, 0.0]]))

    assert compute_update_norm(model_before, model_after).item() == pytest.approx(5.0, rel=1e-4)
