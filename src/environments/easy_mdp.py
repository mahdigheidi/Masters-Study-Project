"""True-label classification MDP from Section 3.2.

Each latent state ``s`` emits images whose dataset label is ``s``.  The reward
is one when the agent predicts the correct class/action and zero otherwise.
After every action the next state is sampled uniformly at random, which keeps
the state visitation distribution independent of the policy.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .classification_mdp import ClassificationMDP, ClassificationMDPSpec


class EasyMDP(ClassificationMDP):
    def __init__(self, dataset, seed: Optional[int] = None):
        super().__init__(
            dataset=dataset,
            spec=ClassificationMDPSpec(name="easy"),
            seed=seed,
        )

    def transition(self, action: int) -> Tuple[float, int]:
        reward = float(int(action) == int(self.state))
        self.state = self.rng.randrange(self.spec.num_states)
        return reward, self.state


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.datasets import MNIST

    dataset = MNIST(root="data", train=True, download=True)
    # plot a few images of the dataset

    env = EasyMDP(dataset)

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
