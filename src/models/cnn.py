"""CNN architecture from Appendix A.2 of the plasticity paper.

The toy-RL CNN has two convolutional layers with 64 channels and kernels
5x5 then 3x3, followed by two fully connected layers of width 256.  No pooling
is used in the paper description.  The implementation computes the flattened
convolutional size from ``input_shape`` so the same class works for MNIST and
CIFAR-10, and exposes ``forward_features`` for rank and plasticity probes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class CNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_shape: Optional[Sequence[int]] = None,
        image_size: int = 28,
        num_actions=10,
        output_dim: Optional[int] = None,
        conv_channels: int = 64,
        fc_dim: int = 256,
        use_layernorm=False,
        spectral_norm=False,
    ):
        super().__init__()

        if input_shape is None:
            input_shape = (input_channels, image_size, image_size)
        else:
            input_channels = int(input_shape[0])

        if output_dim is not None:
            num_actions = int(output_dim)

        self.input_shape = tuple(int(v) for v in input_shape)
        self.num_actions = int(num_actions)
        self.conv_channels = int(conv_channels)
        self.fc_dim = int(fc_dim)
        self.feature_dim = self.fc_dim

        self.use_layernorm = use_layernorm

        self.conv1 = _maybe_spectral_norm(
            nn.Conv2d(input_channels, self.conv_channels, kernel_size=5),
            spectral_norm,
        )
        self.conv2 = _maybe_spectral_norm(
            nn.Conv2d(self.conv_channels, self.conv_channels, kernel_size=3),
            spectral_norm,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            conv1_shape = self.conv1(dummy).shape[1:]
            conv2_shape = self.conv2(self.relu1(self.conv1(dummy))).shape[1:]
            conv_dim = int(torch.numel(torch.zeros(conv2_shape)))

        self.ln1 = nn.LayerNorm(conv1_shape) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(conv2_shape) if use_layernorm else nn.Identity()

        self.fc1 = _maybe_spectral_norm(nn.Linear(conv_dim, self.fc_dim), spectral_norm)
        self.fc2 = _maybe_spectral_norm(nn.Linear(self.fc_dim, self.fc_dim), spectral_norm)
        self.output = nn.Linear(self.fc_dim, self.num_actions)

        self.ln3 = nn.LayerNorm(self.fc_dim) if use_layernorm else nn.Identity()
        self.ln4 = nn.LayerNorm(self.fc_dim) if use_layernorm else nn.Identity()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    @property
    def last_layer(self) -> nn.Linear:
        return self.output

    def reset_last_layer(self) -> None:
        self.output.reset_parameters()

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.relu2(x)

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.ln3(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.ln4(x)
        x = self.relu4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.output(x)
