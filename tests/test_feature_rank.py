import pytest
import torch
import torch.nn as nn

from src.experiments.feature_rank import (
    collect_features,
    compute_feature_rank,
    compute_feature_srank,
    compute_model_feature_rank,
    compute_model_feature_srank,
)
from src.models.mlp import MLP


def _orthogonal_columns(scales: list[float]) -> torch.Tensor:
    """Zero-mean orthogonal columns, so the singular values are the column norms.

    The base columns are mutually orthogonal and each has norm 2, hence scaling
    column ``i`` by ``scales[i]`` yields singular values ``2 * scales[i]``.
    """
    base = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    return base[:, : len(scales)] * torch.tensor(scales)


def test_compute_feature_rank_detects_low_rank_features():
    basis_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    basis_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    coefficients = torch.randn(20, 2)
    # Every row is a linear combination of just two basis vectors -> rank 2.
    features = coefficients @ torch.stack([basis_a, basis_b])

    assert compute_feature_rank(features, threshold=1e-5) == 2


def test_compute_feature_rank_is_centered_before_svd():
    # A constant feature column has zero rank once the mean is removed.
    features = torch.ones(10, 3) * 5.0

    assert compute_feature_rank(features, threshold=1e-5) == 0


def test_compute_model_feature_rank_uses_forward_features():
    model = MLP(input_shape=(1, 4, 4), num_actions=10, hidden_dim=8)
    data = torch.randn(16, 1, 4, 4)

    rank = compute_model_feature_rank(model, data, threshold=1e-5)

    assert 0 <= rank <= model.feature_dim


def test_compute_model_feature_rank_requires_forward_features():
    model = nn.Linear(4, 4)  # no forward_features method

    with pytest.raises(ValueError, match="forward_features"):
        collect_features(model, torch.randn(2, 4))


def test_srank_discounts_directions_that_carry_negligible_energy():
    # Singular values [200, 2]: the first carries 200/202 = 99.01% of the mass,
    # so srank stops at 1 while the absolute threshold still counts both.
    features = _orthogonal_columns([100.0, 1.0])

    assert compute_feature_srank(features, delta=0.01) == 1
    assert compute_feature_rank(features, threshold=1e-5) == 2


def test_srank_counts_directions_with_comparable_energy():
    # Singular values [2, 2]: neither direction alone reaches 99% of the mass.
    features = _orthogonal_columns([1.0, 1.0])

    assert compute_feature_srank(features, delta=0.01) == 2


def test_srank_shrinks_as_delta_grows():
    # Singular values [200, 20, 2] -> cumulative mass [0.901, 0.991, 1.0].
    features = _orthogonal_columns([100.0, 10.0, 1.0])

    assert compute_feature_srank(features, delta=0.01) == 2
    assert compute_feature_srank(features, delta=0.20) == 1


def test_srank_is_centered_before_svd():
    features = torch.ones(10, 3) * 5.0

    assert compute_feature_srank(features) == 0


def test_srank_without_centering_keeps_the_mean_direction():
    # The same constant features carry all their mass in the mean direction,
    # which centering removes but Kumar et al.'s raw definition retains.
    features = torch.ones(10, 3) * 5.0

    assert compute_feature_srank(features, center=False) == 1


def test_srank_rejects_delta_outside_unit_interval():
    features = _orthogonal_columns([1.0, 1.0])

    with pytest.raises(ValueError, match="delta"):
        compute_feature_srank(features, delta=1.0)


def test_compute_model_feature_srank_uses_forward_features():
    model = MLP(input_shape=(1, 4, 4), num_actions=10, hidden_dim=8)
    data = torch.randn(16, 1, 4, 4)

    srank = compute_model_feature_srank(model, data, delta=0.01)

    assert 0 <= srank <= model.feature_dim
