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
def compute_feature_rank(features: torch.Tensor, threshold: float = 1e-5) -> int:
    if features.dim() > 2:
        features = features.flatten(start_dim=1)
    features = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(features)
    return int((singular_values > threshold).sum().item())


@torch.no_grad()
def compute_model_feature_rank(
    model: nn.Module,
    data,
    threshold: float = 1e-5,
    device: torch.device | str | None = None,
) -> int:
    features = collect_features(model, data, device=device)
    return compute_feature_rank(features, threshold=threshold)
