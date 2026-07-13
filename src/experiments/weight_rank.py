"""Weight-rank measurements for Section 5.

The paper tests whether low-rank weights explain plasticity loss.  For a full
model we flatten each linear or convolutional kernel to a two-dimensional
matrix, compute its numerical rank, and report both per-layer and averaged
statistics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def matrix_rank(weight_matrix: torch.Tensor, threshold: float = 1e-5) -> int:
    # SVD on CPU: torch.linalg.svdvals is not implemented for MPS, and these
    # matrices are small enough that GPU offers no benefit anyway.
    matrix = weight_matrix.detach().cpu()
    if matrix.dim() > 2:
        matrix = matrix.reshape(matrix.size(0), -1)
    singular_values = torch.linalg.svdvals(matrix)
    return int((singular_values > threshold).sum().item())


@torch.no_grad()
def weight_rank_statistics(
    model: nn.Module,
    threshold: float = 1e-5,
) -> Dict[str, float]:
    ranks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            ranks[name] = float(matrix_rank(module.weight, threshold=threshold))

    values = list(ranks.values())
    return {
        **ranks,
        "mean_weight_rank": float(np.mean(values)) if values else 0.0,
        "min_weight_rank": float(np.min(values)) if values else 0.0,
        "max_weight_rank": float(np.max(values)) if values else 0.0,
    }


@torch.no_grad()
def compute_weight_rank(obj, threshold: float = 1e-5) -> float:
    if isinstance(obj, torch.Tensor):
        return float(matrix_rank(obj, threshold=threshold))
    return weight_rank_statistics(obj, threshold=threshold)["mean_weight_rank"]
