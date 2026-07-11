"""Shared classification-MDP mechanics from Section 3.2.

The paper turns MNIST or CIFAR-10 classification into a ten-state, ten-action
block MDP.  A latent state selects a class-conditional image observation, the
agent predicts an action in ``{0, ..., 9}``, and the environment returns a
reward according to one of the paper's three variants.  This shared base keeps
dataset indexing, random-label generation, and transition sampling in one
place so the easy, hard, and sparse files only define their variant-specific
reward and transition rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ClassificationMDPSpec:
    name: str
    num_states: int = 10
    num_actions: int = 10


class ClassificationMDP:
    def __init__(
        self,
        dataset,
        spec: ClassificationMDPSpec,
        labels: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.spec = spec
        self.rng = random.Random(seed)
        self.labels = labels if labels is not None else self._true_labels(dataset)
        self.class_indices = self._build_class_indices(self.labels)
        self.state = self.rng.randrange(self.spec.num_states)

    @staticmethod
    def _true_labels(dataset) -> List[int]:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            return [int(dataset[idx][1]) for idx in range(len(dataset))]
        if isinstance(targets, torch.Tensor):
            return [int(value) for value in targets.tolist()]
        return [int(value) for value in targets]

    def _build_class_indices(self, labels: List[int]) -> Dict[int, List[int]]:
        class_indices: Dict[int, List[int]] = {
            state: [] for state in range(self.spec.num_states)
        }
        for idx, raw_label in enumerate(labels):
            label = int(raw_label)
            if 0 <= label < self.spec.num_states:
                class_indices[label].append(idx)

        empty = [state for state, indices in class_indices.items() if not indices]
        if empty:
            raise ValueError(
                "Every MDP state must have at least one image. "
                f"Missing states: {empty}"
            )
        return class_indices

    def reset(self, state: Optional[int] = None) -> torch.Tensor:
        self.state = self.rng.randrange(self.spec.num_states) if state is None else int(state)
        return self.sample_observation(self.state)

    def sample_observation(self, state: Optional[int] = None) -> torch.Tensor:
        state = self.state if state is None else int(state)
        idx = self.rng.choice(self.class_indices[state])
        image, _ = self.dataset[idx]
        return image

    def sample_observation_with_state(
        self,
        state: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        state = self.state if state is None else int(state)
        return self.sample_observation(state), state

    def transition(self, action: int) -> Tuple[float, int]:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[torch.Tensor, float, int]:
        reward, next_state = self.transition(int(action))
        self.state = int(next_state)
        return self.sample_observation(self.state), float(reward), self.state


def make_random_labels(
    dataset,
    num_states: int = 10,
    seed: Optional[int] = None,
) -> List[int]:
    rng = random.Random(seed)
    labels = [rng.randrange(num_states) for _ in range(len(dataset))]

    return labels
