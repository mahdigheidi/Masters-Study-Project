"""Shared fixtures for the plasticity test suite.

Real experiments load MNIST/CIFAR-10 through torchvision. Tests avoid that
network/disk dependency by using a tiny in-memory dataset with the same
``__len__``/``__getitem__``/``targets`` interface the environments expect.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ToyImageDataset(Dataset):
    """Ten balanced classes of 1x4x4 images, labelled 0-9."""

    def __init__(self, num_states: int = 10, samples_per_state: int = 8):
        self.num_states = num_states
        self.targets = [state for state in range(num_states) for _ in range(samples_per_state)]
        self.data = [
            torch.full((1, 4, 4), fill_value=float(label))
            for label in self.targets
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]


def make_toy_dataset(num_states: int = 10, samples_per_state: int = 8) -> ToyImageDataset:
    return ToyImageDataset(num_states=num_states, samples_per_state=samples_per_state)
