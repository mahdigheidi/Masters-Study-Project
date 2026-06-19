"""Sparse-reward classification MDP from Section 3.2.

Observations use the true dataset label like the easy environment.  The reward
is sparse: the agent receives one only when the current state is 9 and the
chosen action is also 9.  If the action matches the current state, the MDP
moves to ``s + 1`` modulo ten; otherwise it jumps to a uniformly random state.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .classification_mdp import ClassificationMDP, ClassificationMDPSpec


class SparseMDP(ClassificationMDP):
    def __init__(self, dataset, seed: Optional[int] = None):
        super().__init__(
            dataset=dataset,
            spec=ClassificationMDPSpec(name="sparse"),
            seed=seed,
        )

    def transition(self, action: int) -> Tuple[float, int]:
        action_matches_state = int(action) == int(self.state)
        reward = float(action_matches_state and int(self.state) == 9)
        if action_matches_state:
            next_state = (int(self.state) + 1) % self.spec.num_states
        else:
            next_state = self.rng.randrange(self.spec.num_states)
        self.state = next_state
        return reward, next_state


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    import matplotlib.pyplot as plt

    dataset = MNIST(root="data", train=True, download=True)
    # plot a few images of the dataset

    env = SparseMDP(dataset)

    fig, axes = plt.subplots(5, 10, figsize=(12, 8))
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
