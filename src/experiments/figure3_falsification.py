"""Figure 3 falsification experiment.

Figure 3 asks whether simple statistics such as weight norm, weight rank,
dead units, or feature rank consistently explain plasticity loss.  This file
keeps the experiment reproducible by training DQN agents on the Section 3.2
classification MDPs, probing plasticity every fixed number of optimizer steps,
and logging the shared metric utilities from ``experiments.section5``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.experiments.dead_units import compute_dead_units
from src.experiments.dqn_classification import (
    DEVICE,
    ClassificationDQNConfig,
    ReplayBuffer,
    build_environment,
    build_model_factory,
    build_optimizer,
    collect_transition,
    epsilon_at_step,
    load_dataset,
    optimizer_factory_from_config,
    set_seed,
    train_dqn_step,
)
from src.experiments.feature_rank import compute_model_feature_rank
from src.experiments.plasticity import PlasticityProbeConfig, estimate_probe_loss
from src.experiments.weight_norm import compute_weight_norm
from src.experiments.weight_rank import compute_weight_rank


FIGURE3_SCHEMA = [
    "run_id",
    "seed",
    "observation_space",
    "environment",
    "architecture",
    "optimizer",
    "step",
    "probe_loss",
    "initial_probe_loss",
    "plasticity_loss",
    "weight_norm",
    "weight_rank",
    "dead_units",
    "feature_rank",
]


@dataclass
class Figure3RunConfig(ClassificationDQNConfig):
    probe_every: int = 5_000
    probe_steps: int = 2_000
    num_probe_tasks: int = 10
    probe_batch_size: int = 512
    metric_batch_size: int = 512


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    title: str
    x_column: str
    x_label: str
    condition_column: str
    conditions: Tuple[str, ...]


PANEL_SPECS: Tuple[PanelSpec, ...] = (
    PanelSpec(
        panel="weight_norm",
        title="Varying observation space",
        x_column="weight_norm",
        x_label="Weight norm",
        condition_column="observation_space",
        conditions=("cifar10", "mnist"),
    ),
    PanelSpec(
        panel="weight_rank",
        title="Varying observation space",
        x_column="weight_rank",
        x_label="Weight rank",
        condition_column="observation_space",
        conditions=("cifar10", "mnist"),
    ),
    PanelSpec(
        panel="dead_units",
        title="Varying architecture",
        x_column="dead_units",
        x_label="Dead units",
        condition_column="architecture",
        conditions=("cnn", "mlp"),
    ),
    PanelSpec(
        panel="feature_rank",
        title="Varying reward function",
        x_column="feature_rank",
        x_label="Feature rank",
        condition_column="environment",
        conditions=("easy", "hard", "sparse"),
    ),
)


COLORS = {
    "cifar10": "#23c4b6",
    "mnist": "#6d3df2",
    "cnn": "#23c4b6",
    "mlp": "#6d3df2",
    "easy": "#23c4b6",
    "hard": "#6d3df2",
    "sparse": "#e83f76",
}


def _has_regression_support(x: np.ndarray, y: np.ndarray) -> bool:
    return len(x) >= 2 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12


def _mean_probe_loss(results) -> float:
    return float(np.mean([result.final_loss for result in results]))


def run_id_for(config: Figure3RunConfig) -> str:
    return (
        f"{config.observation_space}_{config.environment}_"
        f"{config.architecture}_{config.optimizer}_seed{config.seed}"
    )


def compute_probe_loss(
    model,
    model_factory,
    optimizer_factory,
    replay: ReplayBuffer,
    config: Figure3RunConfig,
) -> float:
    states = replay.sample_states(config.probe_batch_size, device=DEVICE)
    probe_config = PlasticityProbeConfig(
        steps=config.probe_steps,
        num_tasks=config.num_probe_tasks,
        batch_size=config.probe_batch_size,
    )
    return _mean_probe_loss(
        estimate_probe_loss(
            model,
            model_factory,
            states,
            optimizer_factory,
            probe_config,
        )
    )


def metric_row(
    run_id: str,
    config: Figure3RunConfig,
    step: int,
    model,
    model_factory,
    optimizer_factory,
    replay: ReplayBuffer,
    initial_probe_loss: float,
    probe_loss_override: Optional[float] = None,
) -> Dict[str, object]:
    metric_states = replay.sample_states(config.metric_batch_size, device=DEVICE)
    probe_loss = (
        compute_probe_loss(
            model,
            model_factory,
            optimizer_factory,
            replay,
            config,
        )
        if probe_loss_override is None
        else probe_loss_override
    )

    return {
        "run_id": run_id,
        "seed": config.seed,
        "observation_space": config.observation_space,
        "environment": config.environment,
        "architecture": config.architecture,
        "optimizer": config.optimizer,
        "step": step,
        "probe_loss": probe_loss,
        "initial_probe_loss": initial_probe_loss,
        "plasticity_loss": probe_loss - initial_probe_loss,
        "weight_norm": compute_weight_norm(model),
        "weight_rank": compute_weight_rank(model),
        "dead_units": compute_dead_units(model, metric_states),
        "feature_rank": compute_model_feature_rank(model, metric_states),
    }


def run_training_config(config: Figure3RunConfig) -> pd.DataFrame:
    set_seed(config.seed)
    dataset, input_shape = load_dataset(config)
    env = build_environment(config, dataset)
    model_factory = build_model_factory(config, input_shape)
    model = model_factory().to(DEVICE)
    target_model = copy.deepcopy(model).to(DEVICE)
    optimizer = build_optimizer(config, model)
    optimizer_factory = optimizer_factory_from_config(config)
    replay = ReplayBuffer(config.replay_capacity)

    for _ in range(config.warmup_steps):
        collect_transition(env, model, replay, epsilon=1.0, device=DEVICE)

    run_id = run_id_for(config)
    initial_probe_loss = compute_probe_loss(
        model,
        model_factory,
        optimizer_factory,
        replay,
        config,
    )
    rows = [
        metric_row(
            run_id,
            config,
            0,
            model,
            model_factory,
            optimizer_factory,
            replay,
            initial_probe_loss,
            probe_loss_override=initial_probe_loss,
        )
    ]

    last_loss = float("nan")
    for step in range(1, config.train_steps + 1):
        epsilon = epsilon_at_step(config, step)
        collect_transition(env, model, replay, epsilon=epsilon, device=DEVICE)
        last_loss = train_dqn_step(
            model,
            target_model,
            optimizer,
            replay,
            config,
            device=DEVICE,
        )

        if step % config.target_update_period == 0:
            target_model.load_state_dict(model.state_dict())

        if step % config.probe_every == 0:
            print(
                f"{run_id}: step={step}, loss={last_loss:.4f}, "
                f"epsilon={epsilon:.3f}"
            )
            rows.append(
                metric_row(
                    run_id,
                    config,
                    step,
                    model,
                    model_factory,
                    optimizer_factory,
                    replay,
                    initial_probe_loss,
                )
            )

    return pd.DataFrame(rows, columns=FIGURE3_SCHEMA)


def make_smoke_configs(data_root: str = "./data", download: bool = False) -> List[Figure3RunConfig]:
    base = Figure3RunConfig(
        data_root=data_root,
        download=download,
        hidden_dim=128,
        batch_size=128,
        replay_capacity=5_000,
        warmup_steps=512,
        train_steps=1_000,
        target_update_period=250,
        probe_every=500,
        probe_steps=25,
        num_probe_tasks=2,
        probe_batch_size=128,
        metric_batch_size=128,
    )
    return [
        replace(base, seed=0, observation_space="mnist", environment="easy", architecture="mlp"),
        replace(base, seed=1, observation_space="mnist", environment="hard", architecture="mlp"),
        replace(base, seed=2, observation_space="mnist", environment="sparse", architecture="mlp"),
        replace(base, seed=3, observation_space="mnist", environment="easy", architecture="cnn"),
    ]


def make_paper_configs(
    data_root: str = "./data",
    download: bool = False,
    seeds: Sequence[int] = tuple(range(4)),
) -> List[Figure3RunConfig]:
    configs: List[Figure3RunConfig] = []
    for seed in seeds:
        for observation_space in ["mnist", "cifar10"]:
            for environment in ["easy", "hard", "sparse"]:
                for architecture in ["mlp", "cnn"]:
                    configs.append(
                        Figure3RunConfig(
                            seed=seed,
                            data_root=data_root,
                            download=download,
                            observation_space=observation_space,
                            environment=environment,
                            architecture=architecture,
                            hidden_dim=512,
                            train_steps=100_000,
                            target_update_period=1_000,
                            probe_every=5_000,
                            probe_steps=2_000,
                            num_probe_tasks=10,
                            probe_batch_size=512,
                            metric_batch_size=512,
                        )
                    )
    return configs


def run_sweep(
    configs: Sequence[Figure3RunConfig],
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    frames = []
    for idx, config in enumerate(configs, start=1):
        print(f"Running config {idx}/{len(configs)}: {asdict(config)}")
        frames.append(run_training_config(config))
    df = pd.concat(frames, ignore_index=True)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved summary rows to {save_path}")
    return df


def validate_summary_table(df: pd.DataFrame) -> None:
    missing = [column for column in FIGURE3_SCHEMA if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Figure 3 columns: {missing}")


def load_summary_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    validate_summary_table(df)
    return df


def figure3_long_table(df: pd.DataFrame) -> pd.DataFrame:
    validate_summary_table(df)
    rows = []
    for _, row in df.iterrows():
        for panel, column in [
            ("weight_norm", "weight_norm"),
            ("weight_rank", "weight_rank"),
            ("dead_units", "dead_units"),
            ("feature_rank", "feature_rank"),
        ]:
            rows.append(
                {
                    **row.to_dict(),
                    "panel": panel,
                    "statistic_name": column,
                    "statistic_value": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    long_df = figure3_long_table(df)
    rows = []
    for spec in PANEL_SPECS:
        panel_df = long_df[long_df["panel"] == spec.panel]
        for condition in spec.conditions:
            group = panel_df[panel_df[spec.condition_column] == condition]
            x = group["statistic_value"].to_numpy()
            y = group["plasticity_loss"].to_numpy()
            if not _has_regression_support(x, y):
                corr = np.nan
                slope = np.nan
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                slope = float(np.polyfit(x, y, 1)[0])
            rows.append(
                {
                    "panel": spec.panel,
                    "condition": condition,
                    "n": len(group),
                    "pearson_r": corr,
                    "linear_slope": slope,
                }
            )
    return pd.DataFrame(rows)


def plot_figure3(df: pd.DataFrame, save_path: Optional[str | Path] = None):
    import matplotlib.pyplot as plt

    long_df = figure3_long_table(df)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    fig.suptitle("Falsification of explanations of plasticity", fontsize=13, y=1.04)

    for ax, spec in zip(axes, PANEL_SPECS):
        panel_df = long_df[long_df["panel"] == spec.panel]
        for condition in spec.conditions:
            group = panel_df[panel_df[spec.condition_column] == condition]
            if group.empty:
                continue
            color = COLORS[condition]
            ax.scatter(
                group["statistic_value"],
                group["plasticity_loss"],
                s=24,
                alpha=0.62,
                color=color,
                edgecolor="none",
                label=condition,
            )
            x = group["statistic_value"].to_numpy()
            y = group["plasticity_loss"].to_numpy()
            if _has_regression_support(x, y):
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(float(x.min()), float(x.max()), 100)
                ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.6)

        ax.set_title(spec.title, fontsize=11)
        ax.set_xlabel(spec.x_label)
        ax.grid(True, linewidth=0.6, alpha=0.55)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("Plasticity loss")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--summary-path", default="outputs/figure3_real_summary.csv")
    parser.add_argument("--figure-path", default="outputs/figure3_real_reproduction.png")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    import matplotlib.pyplot as plt

    args = make_arg_parser().parse_args()
    configs = (
        make_paper_configs(args.data_root, args.download)
        if args.paper
        else make_smoke_configs(args.data_root, args.download)
    )
    df = run_sweep(configs, save_path=args.summary_path)
    print(correlation_summary(df).to_string(index=False))
    plot_figure3(df, save_path=args.figure_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
