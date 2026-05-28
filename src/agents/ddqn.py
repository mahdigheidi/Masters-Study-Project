import copy
import random

import numpy as np
import torch
import torch.nn.functional as F


class DoubleDQNAgent:
    def __init__(
        self,
        q_network,
        optimizer,
        replay_buffer,
        num_actions,
        gamma=0.99,
        batch_size=64,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=10000,
        device="cuda",
    ):
        self.device = device

        self.q_network = q_network.to(device)

        self.target_network = copy.deepcopy(q_network).to(device)

        self.optimizer = optimizer

        self.replay_buffer = replay_buffer

        self.num_actions = num_actions

        self.gamma = gamma
        self.batch_size = batch_size

        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.step_count = 0

    def epsilon(self):
        return self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * np.exp(-1.0 * self.step_count / self.epsilon_decay)

    @torch.no_grad()
    def act(self, state):
        if random.random() < self.epsilon():
            return random.randint(0, self.num_actions - 1)

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.q_network(state)

        return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        q_values = self.q_network(states)

        q_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_online_q = self.q_network(next_states)

            next_actions = next_online_q.argmax(dim=1)

            next_target_q = self.target_network(next_states)

            next_q = next_target_q.gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)

            td_target = rewards + (
                1.0 - dones
            ) * self.gamma * next_q

        loss = F.mse_loss(q_values, td_target)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        self.step_count += 1

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon(),
        }