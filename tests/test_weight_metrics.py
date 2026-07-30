import pytest
import torch
import torch.nn as nn

from src.experiments.weight_norm import compute_weight_norm
from src.experiments.weight_rank import (
    compute_weight_rank,
    compute_weight_srank,
    compute_weight_stable_rank,
    matrix_rank,
    matrix_srank,
    matrix_stable_rank,
    weight_rank_statistics,
)
from src.models.mlp import MLP


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


def _diag(values):
    # A diagonal matrix's singular values are the absolute diagonal entries,
    # so the whole spectrum is known exactly.
    return torch.diag(torch.tensor(values))


def test_matrix_stable_rank_equals_count_for_equal_singular_values():
    # Singular values [2, 2, 2]: sum(sigma^2)/max^2 = 12/4 = 3.
    assert matrix_stable_rank(_diag([2.0, 2.0, 2.0])) == pytest.approx(3.0, abs=1e-5)


def test_matrix_stable_rank_collapses_when_one_direction_dominates():
    # Singular values [10, 1, 1]: (100+1+1)/100 = 1.02 -- full numerical rank 3,
    # but stable rank near 1 because one direction carries almost all the energy.
    matrix = _diag([10.0, 1.0, 1.0])
    assert matrix_rank(matrix) == 3
    assert matrix_stable_rank(matrix) == pytest.approx(1.02, abs=1e-4)


def test_matrix_srank_thresholds_on_singular_value_mass():
    # Singular values [100, 1]: top one is 100/101 = 99.01% >= 99%, so srank = 1,
    # while the numerical rank counts both.
    matrix = _diag([100.0, 1.0])
    assert matrix_srank(matrix, delta=0.01) == 1
    assert matrix_rank(matrix) == 2
    # [1, 1]: neither reaches 99% alone.
    assert matrix_srank(_diag([1.0, 1.0]), delta=0.01) == 2


def test_matrix_srank_rejects_delta_outside_unit_interval():
    with pytest.raises(ValueError, match="delta"):
        matrix_srank(_diag([1.0, 1.0]), delta=1.0)


def test_compute_weight_rank_aggregators_exclude_output_head_by_default():
    model = MLP(input_shape=(1, 8, 8), num_actions=10, hidden_dim=16)
    # The output head (10x16) is present with all layers, absent when excluded,
    # so including it must change the mean.
    assert compute_weight_srank(model, exclude_output=False) != compute_weight_srank(
        model, exclude_output=True
    )
    assert compute_weight_stable_rank(model, exclude_output=False) != compute_weight_stable_rank(
        model, exclude_output=True
    )


def test_stable_rank_never_exceeds_numerical_rank():
    torch.manual_seed(0)
    weight = torch.randn(20, 12)
    assert matrix_stable_rank(weight) <= matrix_rank(weight) + 1e-6
