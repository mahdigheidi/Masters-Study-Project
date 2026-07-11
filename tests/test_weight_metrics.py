import pytest
import torch
import torch.nn as nn

from src.experiments.weight_norm import compute_weight_norm
from src.experiments.weight_rank import compute_weight_rank, matrix_rank, weight_rank_statistics


def test_compute_weight_norm_matches_manual_l2_norm():
    model = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        model.bias.copy_(torch.tensor([0.0, 0.0]))

    # weight contributes sqrt(3^2 + 4^2) = 5; bias is all zero.
    assert compute_weight_norm(model, include_bias=True) == pytest.approx(5.0, abs=1e-5)
    assert compute_weight_norm(model, include_bias=False) == pytest.approx(5.0, abs=1e-5)


def test_compute_weight_norm_excludes_bias_when_requested():
    model = nn.Linear(1, 1, bias=True)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(10.0)

    assert compute_weight_norm(model, include_bias=False) == pytest.approx(0.0, abs=1e-5)
    assert compute_weight_norm(model, include_bias=True) == pytest.approx(10.0, abs=1e-5)


def test_matrix_rank_recovers_known_low_rank_matrix():
    # Rank-2 matrix built as the sum of two independent outer products.
    u1 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u2 = torch.tensor([0.0, 1.0, 0.0])
    v2 = torch.tensor([4.0, 3.0, 2.0, 1.0])
    matrix = torch.outer(u1, v1) + torch.outer(u2, v2)

    assert matrix_rank(matrix, threshold=1e-5) == 2


def test_weight_rank_statistics_reports_per_layer_and_aggregate():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    with torch.no_grad():
        # Force the first layer to rank 1.
        model[0].weight.copy_(torch.outer(torch.ones(4), torch.arange(1.0, 5.0)))

    stats = weight_rank_statistics(model, threshold=1e-5)

    assert stats["0"] == 1.0
    assert stats["min_weight_rank"] <= stats["mean_weight_rank"] <= stats["max_weight_rank"]
    assert compute_weight_rank(model, threshold=1e-5) == stats["mean_weight_rank"]

