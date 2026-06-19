"""Two-hidden-layer MLP used throughout the plasticity experiments.

The paper uses fully connected ReLU networks with two hidden layers and a
width that changes by experiment: width 512 for most toy-RL evaluations,
width 1024 for the optimizer-instability case study, and a sweep over widths
for Figure 5.  This module keeps that width configurable and exposes the
penultimate representation via ``forward_features`` so rank and plasticity
probes can reuse the same model without duplicating architecture code.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _prod(values: Iterable[int]) -> int:
    return int(reduce(mul, values, 1))


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        input_shape: Sequence[int] = (1, 28, 28),
        num_actions=10,
        hidden_dim=512,
        width: Optional[int] = None,
        output_dim: Optional[int] = None,
        use_layernorm=False,
        spectral_norm=False,
    ):
        super().__init__()

        if isinstance(input_dim, (tuple, list)):
            input_shape = input_dim
            input_dim = None

        if input_dim is None:
            input_dim = _prod(input_shape)

        if width is not None:
            hidden_dim = int(width)

        if output_dim is not None:
            num_actions = int(output_dim)

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.feature_dim = self.hidden_dim

        self.use_layernorm = use_layernorm

        self.fc1 = _maybe_spectral_norm(
            nn.Linear(self.input_dim, self.hidden_dim),
            spectral_norm,
        )
        self.fc2 = _maybe_spectral_norm(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            spectral_norm,
        )
        self.output = nn.Linear(self.hidden_dim, self.num_actions)

        self.ln1 = nn.LayerNorm(self.hidden_dim) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(self.hidden_dim) if use_layernorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    @property
    def last_layer(self) -> nn.Linear:
        return self.output

    def reset_last_layer(self) -> None:
        self.output.reset_parameters()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.output(x)
