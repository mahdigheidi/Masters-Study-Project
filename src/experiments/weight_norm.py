"""Weight-norm measurements for Section 5.

The falsification experiments log the global L2 norm of trainable parameters
and compare it with plasticity loss.  Biases are included by default to match
the model's full parameter vector, while a weights-only option is available
for diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def compute_weight_norm(
    model: nn.Module,
    include_bias: bool = True,
) -> float:
    total_norm_sq = 0.0
    for name, parameter in model.named_parameters():
        if not include_bias and name.endswith("bias"):
            continue
        if parameter.requires_grad:
            total_norm_sq += float(parameter.detach().pow(2).sum().cpu().item())
    return float(total_norm_sq ** 0.5)
