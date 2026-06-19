"""Figure 5 foundation: model width versus plasticity loss.

This experiment tests whether simply increasing capacity removes plasticity
loss.  For each requested MLP or CNN width, we train a DQN agent on a
classification MDP, measure random-target probe loss at the final checkpoint,
and subtract the same probe estimate from the matching initial checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.experiments.dqn_classification import (
    DEVICE,
    ClassificationDQNConfig,
    optimizer_factory_from_config,
    run_dqn_training,
)
from src.experiments.plasticity import PlasticityProbeConfig, measure_plasticity_loss


@dataclass
class WidthSweepConfig(ClassificationDQNConfig):
    width: int = 512
    probe_steps: int = 2_000
    num_probe_tasks: int = 10
    probe_batch_size: int = 512


def config_for_width(
    base: WidthSweepConfig,
    architecture: str,
    width: int,
) -> WidthSweepConfig:
    if architecture == "mlp":
        return replace(
            base,
            architecture="mlp",
            width=int(width),
            hidden_dim=int(width),
        )
    if architecture == "cnn":
        return replace(
            base,
            architecture="cnn",
            width=int(width),
            cnn_channels=int(width),
            cnn_fc_dim=int(4 * width),
        )
    raise ValueError(f"Unsupported width-sweep architecture: {architecture}")


def make_width_sweep_configs(
    data_root: str = "./data",
    download: bool = False,
    architectures: Sequence[str] = ("mlp", "cnn"),
    environments: Sequence[str] = ("easy", "hard", "sparse"),
    observation_space: str = "mnist",
    seeds: Sequence[int] = (0,),
    base_width: int = 16,
    factors: Sequence[int] = (1, 2, 4, 8, 12, 16),
) -> List[WidthSweepConfig]:
    base = WidthSweepConfig(
        data_root=data_root,
        download=download,
        observation_space=observation_space,
        batch_size=512,
        replay_capacity=50_000,
        warmup_steps=2_000,
        train_steps=20_000,
        target_update_period=1_000,
        probe_steps=2_000,
        num_probe_tasks=10,
        probe_batch_size=512,
    )
    configs: List[WidthSweepConfig] = []
    for seed in seeds:
        for environment in environments:
            for architecture in architectures:
                for factor in factors:
                    configs.append(
                        config_for_width(
                            replace(base, seed=seed, environment=environment),
                            architecture,
                            base_width * factor,
                        )
                    )
    return configs


def make_smoke_configs(data_root: str = "./data", download: bool = False) -> List[WidthSweepConfig]:
    base = WidthSweepConfig(
        data_root=data_root,
        download=download,
        observation_space="mnist",
        environment="easy",
        batch_size=128,
        replay_capacity=5_000,
        warmup_steps=512,
        train_steps=1_000,
        target_update_period=250,
        probe_steps=50,
        num_probe_tasks=2,
        probe_batch_size=128,
    )
    return [
        config_for_width(base, "mlp", 32),
        config_for_width(base, "mlp", 64),
        config_for_width(base, "cnn", 16),
        config_for_width(base, "cnn", 32),
    ]


def run_width_config(config: WidthSweepConfig) -> dict:
    print(f"Running width config: {asdict(config)}")
    training = run_dqn_training(config, checkpoint_steps=(), device=DEVICE)
    replay = training["replay"]
    inputs = replay.sample_states(config.probe_batch_size, device=DEVICE)
    probe_config = PlasticityProbeConfig(
        steps=config.probe_steps,
        num_tasks=config.num_probe_tasks,
        batch_size=config.probe_batch_size,
    )
    result = measure_plasticity_loss(
        training["model"],
        training["model_factory"],
        inputs,
        optimizer_factory_from_config(config),
        probe_config,
        baseline_model=training["initial_model"],
    )
    return {
        "seed": config.seed,
        "observation_space": config.observation_space,
        "environment": config.environment,
        "architecture": config.architecture,
        "width": config.width,
        "hidden_dim": config.hidden_dim,
        "cnn_channels": config.cnn_channels,
        "cnn_fc_dim": config.cnn_fc_dim,
        "probe_loss": result.probe_loss,
        "initial_probe_loss": result.baseline_probe_loss,
        "plasticity_loss": result.plasticity_loss,
    }


def run_width_sweep(
    configs: Sequence[WidthSweepConfig],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    rows = [run_width_config(config) for config in configs]
    df = pd.DataFrame(rows)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved width sweep table to {save_path}")
    return df


def plot_width_sweep(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for (architecture, environment), group in df.groupby(["architecture", "environment"]):
        group = group.sort_values("width")
        ax.plot(
            group["width"],
            group["plasticity_loss"],
            marker="o",
            linewidth=2.0,
            label=f"{architecture}/{environment}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width")
    ax.set_ylabel("Plasticity loss")
    ax.set_title("Plasticity loss across network widths")
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved width sweep figure to {save_path}")

    return fig
