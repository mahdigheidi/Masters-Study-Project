"""Feature-rank measurements for Section 5.

Feature rank is computed on the representation immediately before the output
head.  Models in ``src.models`` expose that representation through
``forward_features``; this file also keeps a tensor-only path for notebooks
that already have features cached.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _as_batches(data) -> Iterable[torch.Tensor]:
    if isinstance(data, torch.Tensor):
        yield data
        return
    for batch in data:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def collect_features(
    model: nn.Module,
    data,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not hasattr(model, "forward_features"):
        raise ValueError("Model must expose forward_features to measure feature rank.")
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    features = []
    for batch in _as_batches(data):
        features.append(model.forward_features(batch.to(device)).detach().cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def feature_singular_values(features: torch.Tensor, center: bool = True) -> torch.Tensor:
    """Singular values of the feature matrix, descending.

    Every rank definition in this module is a different way of reading this one
    spectrum, so it is exposed to let callers inspect it directly.
    """
    if features.dim() > 2:
        features = features.flatten(start_dim=1)
    if center:
        features = features - features.mean(dim=0, keepdim=True)
    # svdvals is not implemented for MPS tensors, so measure on the CPU copy.
    return torch.linalg.svdvals(features.detach().cpu())


@torch.no_grad()
def compute_feature_rank(features: torch.Tensor, threshold: float = 1e-5) -> int:
    singular_values = feature_singular_values(features)
    return int((singular_values > threshold).sum().item())


@torch.no_grad()
def compute_feature_srank(
    features: torch.Tensor,
    delta: float = 0.01,
    center: bool = True,
) -> int:
    """Effective rank of Kumar et al. (2020), cited by Lyle et al. for Figure 3.

    Returns the smallest ``k`` whose top-``k`` singular values carry at least
    ``1 - delta`` of the total singular-value mass.  Where ``compute_feature_rank``
    applies an absolute threshold and therefore counts every direction that is
    merely non-zero, this measures where the representation's energy is actually
    concentrated.

    Kumar et al. define srank on the raw feature matrix; ``center`` stays
    configurable because ReLU features are non-negative and so carry a large mean
    component that would otherwise dominate the spectrum.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must lie in [0, 1), got {delta}.")

    singular_values = feature_singular_values(features, center=center)
    total = float(singular_values.sum().item())
    if total <= 0.0:
        return 0

    cumulative = torch.cumsum(singular_values, dim=0) / total
    below_threshold = int((cumulative < (1.0 - delta)).sum().item())
    return min(below_threshold + 1, int(singular_values.numel()))


@torch.no_grad()
def compute_model_feature_rank(
    model: nn.Module,
    data,
    threshold: float = 1e-5,
    device: torch.device | str | None = None,
) -> int:
    features = collect_features(model, data, device=device)
    return compute_feature_rank(features, threshold=threshold)


@torch.no_grad()
def compute_model_feature_srank(
    model: nn.Module,
    data,
    delta: float = 0.01,
    center: bool = True,
    device: torch.device | str | None = None,
) -> int:
    features = collect_features(model, data, device=device)
    return compute_feature_srank(features, delta=delta, center=center)
