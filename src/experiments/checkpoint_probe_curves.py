"""Section 5.3 checkpoint probe-learning curves.

The paper follows the Figure 3 plasticity protocol but plots the whole probe
optimization trajectory for checkpoints from different times in DQN training.
If later checkpoints fit the same style of random target functions more slowly
or to a worse final loss, the loss curves make the plasticity loss visible
without relying on a single scalar summary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from src.experiments.dqn_classification import (
    DEVICE,
    ClassificationDQNConfig,
    optimizer_factory_from_config,
    run_dqn_training,
)
from src.experiments.plasticity import PlasticityProbeConfig, probe_learning_curves


@dataclass
class CheckpointCurveConfig(ClassificationDQNConfig):
    checkpoint_steps: Sequence[int] = (0, 5_000, 20_000)
    probe_steps: int = 2_000
    num_probe_tasks: int = 10
    probe_batch_size: int = 512
    curve_log_every: int = 50


def make_smoke_config(data_root: str = "./data", download: bool = False) -> CheckpointCurveConfig:
    return CheckpointCurveConfig(
        data_root=data_root,
        download=download,
        observation_space="mnist",
        environment="easy",
        architecture="mlp",
        hidden_dim=128,
        batch_size=128,
        replay_capacity=5_000,
        warmup_steps=512,
        train_steps=1_000,
        target_update_period=250,
        checkpoint_steps=(0, 500, 1_000),
        probe_steps=100,
        num_probe_tasks=2,
        probe_batch_size=128,
        curve_log_every=10,
    )


def make_paper_like_config(
    data_root: str = "./data",
    download: bool = False,
    architecture: str = "mlp",
    environment: str = "easy",
) -> CheckpointCurveConfig:
    return CheckpointCurveConfig(
        data_root=data_root,
        download=download,
        observation_space="mnist",
        environment=environment,
        architecture=architecture,
        hidden_dim=512,
        train_steps=100_000,
        target_update_period=1_000,
        checkpoint_steps=(0, 5_000, 20_000, 100_000),
        probe_steps=2_000,
        num_probe_tasks=10,
        probe_batch_size=512,
        curve_log_every=50,
    )


def run_checkpoint_curve_experiment(
    config: CheckpointCurveConfig,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    training = run_dqn_training(
        config,
        checkpoint_steps=config.checkpoint_steps,
        device=DEVICE,
    )
    replay = training["replay"]
    model_factory = training["model_factory"]
    checkpoints: Dict[str, object] = {
        f"step_{step}": model.to(DEVICE)
        for step, model in training["checkpoints"].items()
    }
    inputs = replay.sample_states(config.probe_batch_size, device=DEVICE)
    probe_config = PlasticityProbeConfig(
        steps=config.probe_steps,
        num_tasks=config.num_probe_tasks,
        batch_size=config.probe_batch_size,
        log_every=config.curve_log_every,
    )
    rows = probe_learning_curves(
        checkpoints,
        model_factory,
        inputs,
        optimizer_factory_from_config(config),
        probe_config,
    )
    df = pd.DataFrame(rows)
    df.insert(0, "architecture", config.architecture)
    df.insert(0, "environment", config.environment)
    df.insert(0, "observation_space", config.observation_space)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved checkpoint probe curves to {save_path}")

    return df


def plot_checkpoint_learning_curves(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    grouped = (
        df.groupby(["checkpoint", "step"], as_index=False)["loss"]
        .mean()
        .sort_values(["checkpoint", "step"])
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)

    for checkpoint, group in grouped.groupby("checkpoint"):
        ax.plot(group["step"], group["loss"], linewidth=2.0, label=checkpoint)

    ax.set_title("Probe learning curves by DQN checkpoint")
    ax.set_xlabel("Probe optimizer step")
    ax.set_ylabel("MSE to random target")
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.legend(title="Checkpoint")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved checkpoint curve figure to {save_path}")

    return fig


def with_checkpoint_steps(
    config: CheckpointCurveConfig,
    checkpoint_steps: Sequence[int],
) -> CheckpointCurveConfig:
    return replace(config, checkpoint_steps=tuple(checkpoint_steps))
