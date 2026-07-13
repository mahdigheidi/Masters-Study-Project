"""Reusable DQN loop for the paper's classification MDP experiments.

Sections 5 and 6 repeatedly train DQN agents on the same toy RL testbed and
then probe checkpoints for plasticity.  This module centralizes dataset
loading, environment construction, model construction, replay sampling, target
network updates, and epsilon-greedy interaction so each figure implementation
can focus on its specific metric or intervention.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.agents.replay_buffer import ReplayBuffer
from src.environments.easy_mdp import EasyMDP
from src.environments.hard_mdp import HardMDP
from src.environments.sparse_mdp import SparseMDP
from src.models.cnn import CNN
from src.models.mlp import MLP
from src.models.vit import VisionTransformer


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple-silicon GPU: ~3-4x faster than CPU for this project's batch-512
    # MLP/CNN training and probe workloads (measured).
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _default_device()


@dataclass
class ClassificationDQNConfig:
    seed: int = 0
    data_root: str = "./data"
    download: bool = False
    observation_space: str = "mnist"
    environment: str = "easy"
    architecture: str = "mlp"
    hidden_dim: int = 512
    cnn_channels: int = 64
    cnn_fc_dim: int = 256
    gamma: float = 0.99
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    batch_size: int = 512
    replay_capacity: int = 50_000
    warmup_steps: int = 2_000
    train_steps: int = 20_000
    target_update_period: int = 1_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay: int = 10_000
    use_layernorm: bool = False
    spectral_norm: bool = False
    shrink_perturb_every: Optional[int] = None
    shrink: float = 0.4
    perturb: float = 0.1
    reset_last_layer_every: Optional[int] = None


Transition = Tuple[torch.Tensor, int, float, torch.Tensor]
MetricCallback = Callable[[int, nn.Module, ReplayBuffer], Dict[str, object]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(config: ClassificationDQNConfig):
    data_root = Path(config.data_root)
    if config.observation_space == "mnist":
        dataset = datasets.MNIST(
            root=str(data_root),
            train=True,
            download=config.download,
            transform=transforms.ToTensor(),
        )
        input_shape = (1, 28, 28)
    elif config.observation_space == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=config.download,
            transform=transforms.ToTensor(),
        )
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unknown observation space: {config.observation_space}")
    return dataset, input_shape


def build_environment(config: ClassificationDQNConfig, dataset):
    if config.environment == "easy":
        return EasyMDP(dataset, seed=config.seed)
    if config.environment == "hard":
        return HardMDP(dataset, seed=config.seed)
    if config.environment == "sparse":
        return SparseMDP(dataset, seed=config.seed)
    raise ValueError(f"Unknown environment: {config.environment}")


def build_model_factory(
    config: ClassificationDQNConfig,
    input_shape: Sequence[int],
) -> Callable[[], nn.Module]:
    def factory() -> nn.Module:
        if config.architecture == "mlp":
            return MLP(
                input_shape=input_shape,
                num_actions=10,
                hidden_dim=config.hidden_dim,
                use_layernorm=config.use_layernorm,
                spectral_norm=config.spectral_norm,
            )
        if config.architecture == "cnn":
            return CNN(
                input_shape=input_shape,
                num_actions=10,
                conv_channels=config.cnn_channels,
                fc_dim=config.cnn_fc_dim,
                use_layernorm=config.use_layernorm,
                spectral_norm=config.spectral_norm,
            )
        if config.architecture == "vit":
            return VisionTransformer(
                input_shape=input_shape,
                num_actions=10,
                patch_size=3,
                dim=256,
                depth=1,
                mlp_dim=1024,
                dropout=0.1,
            )
        raise ValueError(f"Unknown architecture: {config.architecture}")

    return factory


def build_model(
    config: ClassificationDQNConfig,
    input_shape: Sequence[int],
    device: torch.device | str = DEVICE,
) -> nn.Module:
    return build_model_factory(config, input_shape)().to(device)


def build_optimizer(
    config: ClassificationDQNConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def optimizer_factory_from_config(
    config: ClassificationDQNConfig,
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    def factory(model: nn.Module) -> torch.optim.Optimizer:
        return build_optimizer(config, model)

    return factory


def epsilon_at_step(config: ClassificationDQNConfig, step: int) -> float:
    return config.epsilon_final + (
        config.epsilon_start - config.epsilon_final
    ) * np.exp(-float(step) / float(config.epsilon_decay))


@torch.no_grad()
def select_action(
    model: nn.Module,
    observation: torch.Tensor,
    epsilon: float,
    device: torch.device | str = DEVICE,
) -> int:
    if random.random() < epsilon:
        return random.randrange(10)
    q_values = model(observation.unsqueeze(0).to(device))
    return int(q_values.argmax(dim=1).item())


def collect_transition(
    env,
    model: nn.Module,
    replay: ReplayBuffer,
    epsilon: float,
    device: torch.device | str = DEVICE,
) -> None:
    observation = env.sample_observation()
    action = select_action(model, observation, epsilon, device=device)
    next_observation, reward, _ = env.step(action)
    replay.push(observation, action, reward, next_observation)


def dqn_loss(
    model: nn.Module,
    target_model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
) -> torch.Tensor:
    states, actions, rewards, next_states = batch
    q_sa = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).max(dim=1).values
        target = rewards + gamma * next_q
    return F.mse_loss(q_sa, target)


def train_dqn_step(
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: ClassificationDQNConfig,
    device: torch.device | str = DEVICE,
) -> float:
    batch = replay.sample(config.batch_size, device=device)
    loss = dqn_loss(model, target_model, batch, config.gamma)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def shrink_and_perturb(
    model: nn.Module,
    fresh_model: nn.Module,
    shrink: float,
    perturb: float,
) -> None:
    fresh_state = fresh_model.state_dict()
    current_state = model.state_dict()
    for name, value in current_state.items():
        if torch.is_floating_point(value):
            value.mul_(shrink).add_(fresh_state[name].to(value.device), alpha=perturb)


def maybe_apply_periodic_interventions(
    model: nn.Module,
    model_factory: Callable[[], nn.Module],
    config: ClassificationDQNConfig,
    step: int,
    device: torch.device | str = DEVICE,
) -> None:
    if config.reset_last_layer_every and step % config.reset_last_layer_every == 0:
        if hasattr(model, "reset_last_layer"):
            model.reset_last_layer()

    if config.shrink_perturb_every and step % config.shrink_perturb_every == 0:
        fresh_model = model_factory().to(device)
        shrink_and_perturb(model, fresh_model, config.shrink, config.perturb)


def clone_for_checkpoint(model: nn.Module, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = copy.deepcopy(model).to(device)
    checkpoint.eval()
    return checkpoint


def run_dqn_training(
    config: ClassificationDQNConfig,
    checkpoint_steps: Sequence[int] = (),
    metric_callback: Optional[MetricCallback] = None,
    metric_steps: Optional[Sequence[int]] = None,
    device: torch.device | str = DEVICE,
) -> Dict[str, object]:
    set_seed(config.seed)
    dataset, input_shape = load_dataset(config)
    env = build_environment(config, dataset)
    model_factory = build_model_factory(config, input_shape)
    model = model_factory().to(device)
    initial_model = clone_for_checkpoint(model, device=device)
    target_model = copy.deepcopy(model).to(device)
    optimizer = build_optimizer(config, model)
    replay = ReplayBuffer(config.replay_capacity)

    for _ in range(config.warmup_steps):
        collect_transition(env, model, replay, epsilon=1.0, device=device)

    checkpoint_set = set(int(step) for step in checkpoint_steps)
    metric_step_set = None if metric_steps is None else set(int(step) for step in metric_steps)
    checkpoints: Dict[int, nn.Module] = {}
    metric_rows: List[Dict[str, object]] = []

    if 0 in checkpoint_set:
        checkpoints[0] = clone_for_checkpoint(model, device="cpu")
    if metric_callback is not None and (metric_step_set is None or 0 in metric_step_set):
        metric_rows.append(metric_callback(0, model, replay))

    last_loss = float("nan")
    for step in range(1, config.train_steps + 1):
        epsilon = epsilon_at_step(config, step)
        collect_transition(env, model, replay, epsilon=epsilon, device=device)
        last_loss = train_dqn_step(
            model,
            target_model,
            optimizer,
            replay,
            config,
            device=device,
        )

        maybe_apply_periodic_interventions(
            model,
            model_factory,
            config,
            step,
            device=device,
        )

        if step % config.target_update_period == 0:
            target_model.load_state_dict(model.state_dict())

        if step in checkpoint_set:
            checkpoints[step] = clone_for_checkpoint(model, device="cpu")
        if metric_callback is not None and (
            metric_step_set is None or step in metric_step_set
        ):
            row = metric_callback(step, model, replay)
            if row:
                metric_rows.append(row)

    return {
        "config": config,
        "dataset": dataset,
        "input_shape": input_shape,
        "env": env,
        "model": model,
        "initial_model": initial_model,
        "target_model": target_model,
        "model_factory": model_factory,
        "optimizer": optimizer,
        "replay": replay,
        "checkpoints": checkpoints,
        "metrics": metric_rows,
        "last_loss": last_loss,
    }
