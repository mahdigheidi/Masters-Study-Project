import pytest
import torch
import torch.nn as nn

from src.experiments.feature_rank import (
    collect_features,
    compute_feature_rank,
    compute_model_feature_rank,
)
from src.models.mlp import MLP


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
