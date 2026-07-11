import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(
        self,
        image,
        action,
        reward,
        next_image,
    ):
        self.buffer.append(
            (
                image,
                action,
                reward,
                next_image,
            )
        )

    def sample(self, batch_size, device=None):
        device = self.device if device is None else device
        batch = random.sample(self.buffer, batch_size)

        images, actions, rewards, next_images = zip(*batch)

        images = torch.tensor(
            np.array(images),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )

        next_images = torch.tensor(
            np.array(next_images),
            dtype=torch.float32,
            device=device,
        )

        return (
            images,
            actions,
            rewards,
            next_images,
        )

    def sample_states(self, batch_size, device=None):
        device = self.device if device is None else device
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        images = [item[0] for item in batch]
        return torch.tensor(np.array(images), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.buffer)
