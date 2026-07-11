"""Random-label classification MDP from Section 3.2.

This environment has the same reward and transition dynamics as the true-label
MDP, but the image-to-state mapping is made deliberately hard by assigning a
random label in ``{0, ..., 9}`` to every image before building the state
conditional observation buckets.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .classification_mdp import (
    ClassificationMDP,
    ClassificationMDPSpec,
    make_random_labels,
)


class HardMDP(ClassificationMDP):
    def __init__(self, dataset, seed: Optional[int] = None):
        labels = make_random_labels(dataset, seed=seed)
        super().__init__(
            dataset=dataset,
            spec=ClassificationMDPSpec(name="hard"),
            labels=labels,
            seed=seed,
        )

    def transition(self, action: int) -> Tuple[float, int]:
        reward = float(int(action) == int(self.state))
        self.state = self.rng.randrange(self.spec.num_states)
        return reward, self.state


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    import matplotlib.pyplot as plt

    dataset = MNIST(root="../data", train=True, download=True)
    # plot a few images of the dataset

    env = HardMDP(dataset)

    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        current_state = env.state
        image = env.sample_observation()

        action = env.rng.randrange(env.spec.num_actions)
        reward, next_state = env.transition(action)

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"state={current_state}\n"
            f"action={action}, reward={reward:.0f}\n"
            f"next_state={next_state}"
        )
        ax.axis("off")

        # env.state = next_state

    plt.tight_layout()
    plt.show()
