"""Weight-rank measurements for Section 5.

The paper tests whether low-rank weights explain plasticity loss.  Three notions
of "rank" appear here, and they behave very differently on trained weights:

* **Numerical rank** (:func:`matrix_rank`) -- count of singular values above an
  absolute threshold.  A dense weight matrix is generically full rank, and
  training does not drive it rank-deficient at a ``1e-5`` threshold, so this
  saturates at ``min(rows, cols)`` and never moves.  Averaged over an MLP's
  layers it is a constant fixed purely by the architecture dimensions -- e.g.
  ``(512 + 512 + 10) / 3 = 344.7`` for a width-512 MLP, identical at every
  training step and every seed.  It carries no training signal.
* **Effective rank** / srank (:func:`matrix_srank`) -- Kumar et al. (2020), the
  measure Lyle et al. cite for Figure 3: the smallest ``k`` whose top-``k``
  singular values carry ``1 - delta`` of the singular-value mass.  This matches
  the paper's Figure 3 weight-rank axis (~350-500), but it too varies only
  weakly, because the weight spectrum's *mass* stays spread across many
  directions even as training progresses.
* **Stable rank** (:func:`matrix_stable_rank`) -- ``||W||_F^2 / ||W||_2^2``.
  Continuous, always in ``[1, rank]``, and the one that actually tracks
  training-induced collapse: it falls sharply as energy concentrates into the
  top singular direction (implicit under-parameterization).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def _weight_singular_values(weight_matrix: torch.Tensor) -> torch.Tensor:
    # SVD on CPU: torch.linalg.svdvals is not implemented for MPS, and these
    # matrices are small enough that GPU offers no benefit anyway.
    matrix = weight_matrix.detach().cpu().float()
    if matrix.dim() > 2:
        matrix = matrix.reshape(matrix.size(0), -1)
    return torch.linalg.svdvals(matrix)


@torch.no_grad()
def matrix_rank(weight_matrix: torch.Tensor, threshold: float = 1e-5) -> int:
    singular_values = _weight_singular_values(weight_matrix)
    return int((singular_values > threshold).sum().item())


@torch.no_grad()
def matrix_srank(weight_matrix: torch.Tensor, delta: float = 0.01) -> int:
    """Effective rank: smallest ``k`` capturing ``>= 1 - delta`` of the mass."""
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must lie in [0, 1), got {delta}.")
    singular_values = _weight_singular_values(weight_matrix)
    total = float(singular_values.sum().item())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(singular_values, dim=0) / total
    below_threshold = int((cumulative < (1.0 - delta)).sum().item())
    return min(below_threshold + 1, int(singular_values.numel()))


@torch.no_grad()
def matrix_stable_rank(weight_matrix: torch.Tensor) -> float:
    """Stable rank ``||W||_F^2 / ||W||_2^2 = sum(sigma^2) / max(sigma)^2``."""
    singular_values = _weight_singular_values(weight_matrix)
    if singular_values.numel() == 0:
        return 0.0
    top = float(singular_values[0].item())
    if top <= 0.0:
        return 0.0
    return float((singular_values.pow(2).sum() / (top * top)).item())


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


def _weight_matrices(model: nn.Module, exclude_output: bool) -> List[torch.Tensor]:
    """Linear/Conv weight tensors, optionally dropping the final output head.

    The output head is ``num_actions`` wide (10 here), so its rank is capped at
    10 by the action count rather than by anything the network learns.  Averaging
    it in only dilutes a representational rank measure, so the rank aggregators
    below exclude it by default.
    """
    output_weight = None
    if exclude_output:
        last_layer = getattr(model, "last_layer", None)
        output_weight = getattr(last_layer, "weight", None)

    matrices = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if output_weight is not None and module.weight is output_weight:
                continue
            matrices.append(module.weight)
    return matrices


@torch.no_grad()
def compute_weight_srank(
    model: nn.Module,
    delta: float = 0.01,
    exclude_output: bool = True,
) -> float:
    """Mean effective rank over weight matrices -- Figure 3's paper-faithful axis."""
    matrices = _weight_matrices(model, exclude_output)
    if not matrices:
        return 0.0
    return float(np.mean([matrix_srank(weight, delta=delta) for weight in matrices]))


@torch.no_grad()
def compute_weight_stable_rank(
    model: nn.Module,
    exclude_output: bool = True,
) -> float:
    """Mean stable rank over weight matrices -- the measure that tracks training."""
    matrices = _weight_matrices(model, exclude_output)
    if not matrices:
        return 0.0
    return float(np.mean([matrix_stable_rank(weight) for weight in matrices]))
