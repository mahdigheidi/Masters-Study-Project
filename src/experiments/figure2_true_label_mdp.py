"""
Figure 2 reproduction helpers for Lyle et al. 2023.

This implements the Brownian-motion case study from
"Understanding Plasticity in Neural Networks" on the true-label MNIST
classification MDP:

* state space is {0, ..., 9}
* observations for state s are MNIST images with label s
* reward is 1[action == s]
* next state is uniformly random

The experiment couples two trajectories from the same initialization:

* gradient: SGD on a Q-learning objective with a target network
* brownian: Gaussian parameter perturbations with the same norm as each SGD
  update

Geometry is measured on a probe regression objective

    || f_theta(X) - stop_gradient(f_theta(X) + eps) ||^2

using gradient covariance and stochastic Lanczos Hessian ESD.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from tqdm import tqdm

from src.environments.easy_mdp import EasyMDP
from src.models.mlp import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Figure2Config:
    seed: int = 0
    data_root: str = "./data"
    download: bool = False
    hidden_dim: int = 512
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 512
    replay_capacity: int = 50_000
    prefill_steps: int = 50_000
    train_steps: int = 100_000
    target_update_period: int = 5_000
    epsilon: float = 0.1
    online_collection_per_step: int = 100
    probe_batch_size: int = 512
    cov_batch_size: int = 512
    lanczos_iter: int = 100
    lanczos_vectors: int = 3
    esd_points: int = 600
    esd_sigma: Optional[float] = None
    snapshot_target_updates: Tuple[int, ...] = (1, 20)
    esd_xlim: Tuple[float, float] = (0.0, 50.0)
    esd_ylim: Tuple[float, float] = (0.0, 0.4)
    esd_xtick_step: float = 10.0
    esd_ytick_step: float = 0.1
    covariance_vmin: float = -1.0
    covariance_vmax: float = 1.0
    compute_hessian: bool = True
    output_path: str = "outputs/figure2_true_label_mdp.png"


PAPER_SCALE_OVERRIDES = {
    "hidden_dim": 1024,
    "prefill_steps": 50_000,
    "train_steps": 100_000,
    "probe_batch_size": 512,
    "cov_batch_size": 512,
    "lanczos_iter": 100,
    "lanczos_vectors": 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QNetwork(MLP):
    def __init__(self, hidden_dim: int = 512):
        super().__init__(input_shape=(1, 28, 28), num_actions=10, hidden_dim=hidden_dim)



# what is what here? torch.Tensor, int, float?
Transition = Tuple[torch.Tensor, int, float, torch.Tensor, int, int]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, labels, next_labels = zip(*batch)
        # where is even the observations???
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(next_labels, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def load_true_label_mnist(data_root: str, download: bool) -> datasets.MNIST:
    transform = transforms.ToTensor()
    return datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )


@torch.no_grad()
def select_action(model: QNetwork, obs: torch.Tensor, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(10)

    q_values = model(obs.unsqueeze(0).to(DEVICE))
    return int(q_values.argmax(dim=1).item())


def collect_transition(
    env: EasyMDP,
    model: QNetwork,
    replay: ReplayBuffer,
    epsilon: float,
) -> None:
    state = int(env.state)
    obs = env.sample_observation(state)
    action = select_action(model, obs, epsilon)
    next_obs, reward, next_state = env.step(action)
    replay.add((obs.cpu(), action, reward, next_obs.cpu(), state, int(next_state)))


def populate_replay(
    env: EasyMDP,
    model: QNetwork,
    replay: ReplayBuffer,
    steps: int,
    epsilon: float,
) -> None:
    for _ in range(steps):
        collect_transition(env, model, replay, epsilon)


def q_learning_loss(
    model: QNetwork,
    target_model: QNetwork,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
) -> torch.Tensor:
    states, actions, rewards, next_states, _, _ = batch
    states = states.to(DEVICE)
    actions = actions.to(DEVICE)
    rewards = rewards.to(DEVICE)
    next_states = next_states.to(DEVICE)

    q_sa = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_states).max(dim=1).values
        td_target = rewards + gamma * next_q

    return F.mse_loss(q_sa, td_target)


def sgd_step_and_update_norm(
    model: QNetwork,
    target_model: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
    lr: float,
) -> Tuple[float, float]:
    model.train()
    loss = q_learning_loss(model, target_model, batch, gamma)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    update_norm_sq = torch.zeros((), device=DEVICE)
    for p in model.parameters():
        if p.grad is not None:
            update_norm_sq += (lr * p.grad).pow(2).sum()

    optimizer.step()
    return float(loss.detach().cpu().item()), float(update_norm_sq.sqrt().cpu().item())


@torch.no_grad()
def brownian_step(model: QNetwork, step_norm: float) -> None:
    if step_norm <= 0:
        return

    noise = [torch.randn_like(p) for p in model.parameters()]
    total_norm = torch.sqrt(sum((n * n).sum() for n in noise))
    scale = step_norm / (float(total_norm.item()) + 1e-12)

    for p, n in zip(model.parameters(), noise):
        p.add_(n * scale)


@torch.no_grad()
def evaluate_policy(
    model: QNetwork,
    env: EasyMDP,
    num_samples: int = 2048,
) -> float:
    model.eval()
    correct = 0

    for _ in range(num_samples):
        state = random.randrange(10)
        obs = env.sample_observation(state)
        pred = model(obs.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
        correct += int(pred == state)

    return correct / float(num_samples)


def make_probe_targets(
    model: QNetwork,
    inputs: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        if noise is None:
            noise = torch.randn_like(outputs)
        return (outputs + noise).detach(), noise.detach()


def probe_loss_from_targets(
    model: QNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(model(inputs.to(DEVICE)), targets.to(DEVICE))


def flat_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flat_grad_from_loss(
    model: QNetwork,
    loss: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        flat_parameters(model),
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return torch.cat([g.reshape(-1) for g in grads])


def sample_probe_gradient(
    model: QNetwork,
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = probe_loss_from_targets(
        model,
        x.unsqueeze(0),
        target.unsqueeze(0),
    )
    grad = flat_grad_from_loss(model, loss, create_graph=False).detach().cpu()
    model.zero_grad(set_to_none=True)
    return grad


def gradient_covariance(
    model: QNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> np.ndarray:
    model.eval()
    k = min(k, inputs.size(0))

    grads = []
    for i in range(k):
        grad = sample_probe_gradient(model, inputs[i], targets[i])
        grad = grad / grad.norm().clamp(min=1e-12)
        grads.append(grad)

    grad_matrix = torch.stack(grads)
    cov = grad_matrix @ grad_matrix.T
    return cov.numpy()


def hessian_vector_product(
    model: QNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    params = flat_parameters(model)
    model.zero_grad(set_to_none=True)
    loss = probe_loss_from_targets(model, inputs, targets)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    grad_vector_product = torch.dot(flat_grad, vector.detach())
    hvp = torch.autograd.grad(grad_vector_product, params, retain_graph=False)
    model.zero_grad(set_to_none=True)
    return torch.cat([h.detach().reshape(-1) for h in hvp])


def hessian_esd(
    model: QNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lanczos_iter: int,
    lanczos_vectors: int,
    initial_vectors: Optional[Sequence[torch.Tensor]] = None,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    num_params = sum(p.numel() for p in flat_parameters(model))
    eigenvalues: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    if initial_vectors is None:
        initial_vectors = [
            make_rademacher_vector(num_params)
            for _ in range(lanczos_vectors)
        ]

    for initial_v in initial_vectors:
        v = initial_v.to(DEVICE).clone()
        v = v / v.norm()

        v_prev = torch.zeros_like(v)
        beta_prev = 0.0
        alphas: List[float] = []
        betas: List[float] = []

        for _ in range(lanczos_iter):
            w = hessian_vector_product(model, inputs, targets, v)
            if alphas:
                w = w - beta_prev * v_prev

            alpha = torch.dot(v, w).item()
            w = w - alpha * v

            beta = w.norm().item()
            alphas.append(alpha)

            if beta < tol:
                break

            betas.append(beta)
            v_prev = v
            v = w / beta
            beta_prev = beta

        n = len(alphas)
        tridiagonal = torch.zeros(n, n, device=DEVICE)
        for i, alpha in enumerate(alphas):
            tridiagonal[i, i] = alpha
        for i, beta in enumerate(betas[: n - 1]):
            tridiagonal[i, i + 1] = beta
            tridiagonal[i + 1, i] = beta

        vals, vecs = torch.linalg.eigh(tridiagonal)
        eigenvalues.append(vals.detach().cpu().numpy())
        weights.append((vecs[0, :] ** 2).detach().cpu().numpy())

    return np.concatenate(eigenvalues), np.concatenate(weights)


def make_rademacher_vector(num_params: int) -> torch.Tensor:
    v = torch.randint(0, 2, (num_params,), device=DEVICE, dtype=torch.float32)
    return 2.0 * v - 1.0


def esd_curve(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    points: int,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    weights = weights[order]

    lo = float(eigenvalues.min())
    hi = float(eigenvalues.max())
    width = max(hi - lo, 1e-8)
    x = np.linspace(lo - 0.05 * width, hi + 0.05 * width, points)

    if sigma is None:
        sigma = max(0.01 * width, 1e-4)

    density = np.zeros_like(x)
    normalizer = sigma * math.sqrt(2.0 * math.pi)
    for lam, weight in zip(eigenvalues, weights):
        density += weight * np.exp(-0.5 * ((x - lam) / sigma) ** 2) / normalizer

    dx = x[1] - x[0]
    density = density / (density.sum() * dx + 1e-12)
    return x, density


def reorder_by_kmeans(cov: np.ndarray, clusters: int = 10) -> np.ndarray:
    clusters = min(clusters, cov.shape[0])
    labels = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit_predict(cov)
    order = np.argsort(labels)
    return cov[order][:, order]


def snapshot_geometry(
    tag: str,
    gd_model: QNetwork,
    brownian_model: QNetwork,
    probe_inputs: torch.Tensor,
    config: Figure2Config,
) -> Dict[str, object]:
    probe_inputs = probe_inputs.to(DEVICE)

    gd_targets, shared_noise = make_probe_targets(gd_model, probe_inputs)
    brownian_targets, _ = make_probe_targets(
        brownian_model,
        probe_inputs,
        noise=shared_noise,
    )

    snapshot: Dict[str, object] = {
        "tag": tag,
        "covariance": {
            "gradient": gradient_covariance(
                gd_model,
                probe_inputs,
                gd_targets,
                config.cov_batch_size,
            ),
            "brownian": gradient_covariance(
                brownian_model,
                probe_inputs,
                brownian_targets,
                config.cov_batch_size,
            ),
        },
    }

    if config.compute_hessian:
        num_params = sum(p.numel() for p in flat_parameters(gd_model))
        lanczos_initial_vectors = [
            make_rademacher_vector(num_params)
            for _ in range(config.lanczos_vectors)
        ]
        gd_eig, gd_weight = hessian_esd(
            gd_model,
            probe_inputs,
            gd_targets,
            config.lanczos_iter,
            config.lanczos_vectors,
            initial_vectors=lanczos_initial_vectors,
        )
        bm_eig, bm_weight = hessian_esd(
            brownian_model,
            probe_inputs,
            brownian_targets,
            config.lanczos_iter,
            config.lanczos_vectors,
            initial_vectors=lanczos_initial_vectors,
        )
        snapshot["esd"] = {
            "gradient": (gd_eig, gd_weight),
            "brownian": (bm_eig, bm_weight),
        }
        snapshot["top_eigenvalue"] = {
            "gradient": float(gd_eig.max()),
            "brownian": float(bm_eig.max()),
        }

    return snapshot


def run_experiment(config: Figure2Config) -> Dict[str, object]:
    set_seed(config.seed)
    dataset = load_true_label_mnist(config.data_root, config.download)
    env = EasyMDP(dataset)

    gd_model = QNetwork(config.hidden_dim).to(DEVICE)
    brownian_model = copy.deepcopy(gd_model).to(DEVICE)
    target_model = copy.deepcopy(gd_model).to(DEVICE)
    optimizer = torch.optim.SGD(gd_model.parameters(), lr=config.lr)

    replay = ReplayBuffer(config.replay_capacity)
    populate_replay(env, gd_model, replay, config.prefill_steps, epsilon=1.0)

    probe_batch = replay.sample(config.probe_batch_size)
    probe_inputs = probe_batch[0]

    logs: List[Dict[str, float]] = []
    snapshots: Dict[str, Dict[str, object]] = {}
    snapshot_targets = set(config.snapshot_target_updates)
    missing_snapshots = {
        target_update
        for target_update in snapshot_targets
        if target_update * config.target_update_period > config.train_steps
    }
    if missing_snapshots:
        print(
            "Warning: these requested target-update snapshots will not be "
            f"reached with train_steps={config.train_steps}: "
            f"{sorted(missing_snapshots)}"
        )

    print(f"Device: {DEVICE}")
    print(f"Config: {asdict(config)}")

    for step in tqdm(range(1, config.train_steps + 1)):
        for _ in range(config.online_collection_per_step):
            collect_transition(env, gd_model, replay, config.epsilon)

        batch = replay.sample(config.batch_size)
        loss, update_norm = sgd_step_and_update_norm(
            gd_model,
            target_model,
            optimizer,
            batch,
            config.gamma,
            config.lr,
        )
        brownian_step(brownian_model, update_norm)

        if step % config.target_update_period == 0:
            target_model.load_state_dict(gd_model.state_dict())
            target_update = step // config.target_update_period
            accuracy = evaluate_policy(gd_model, env, num_samples=1024)
            log_row = {
                "step": float(step),
                "target_update": float(target_update),
                "loss": float(loss),
                "update_norm": float(update_norm),
                "accuracy": float(accuracy),
            }
            logs.append(log_row)
            print(
                "target update "
                f"{int(log_row['target_update'])}: "
                f"step={step}, loss={loss:.4f}, "
                f"update_norm={update_norm:.5f}, accuracy={accuracy:.3f}"
            )

            if target_update in snapshot_targets:
                tag = f"iter{target_update}"
                print(f"Computing probe geometry after target update {target_update}...")
                snapshots[tag] = snapshot_geometry(
                    tag,
                    gd_model,
                    brownian_model,
                    probe_inputs,
                    config,
                )

    if not snapshots:
        print("No requested target-update snapshots were reached; computing final geometry.")
        snapshots["final"] = snapshot_geometry(
            "final",
            gd_model,
            brownian_model,
            probe_inputs,
            config,
        )

    return {
        "config": asdict(config),
        "logs": logs,
        "snapshots": snapshots,
        "models": {
            "gradient": gd_model,
            "brownian": brownian_model,
            "target": target_model,
        },
        "env": env,
        "probe_inputs": probe_inputs,
    }


def plot_results(
    results: Dict[str, object],
    output_path: Optional[str] = None,
) -> plt.Figure:
    config = Figure2Config(**results["config"])
    snapshots = results["snapshots"]
    snapshot_keys = list(snapshots.keys())
    if len(snapshot_keys) < 2:
        snapshot_keys = snapshot_keys * 2

    has_esd = "esd" in snapshots[snapshot_keys[0]]
    rows = 2 if has_esd else 1
    fig, axes = plt.subplots(rows, 4, figsize=(14, 4.2 * rows))

    if rows == 1:
        axes = np.asarray([axes])

    colors = {"gradient": "#2166ac", "brownian": "#d6604d"}
    labels = {"gradient": "Gradient descent", "brownian": "Brownian motion"}

    if has_esd:
        for col, key in enumerate(snapshot_keys[:2]):
            ax = axes[0, col]
            for trajectory in ["gradient", "brownian"]:
                eig, weight = snapshots[key]["esd"][trajectory]
                x, density = esd_curve(
                    eig,
                    weight,
                    points=config.esd_points,
                    sigma=config.esd_sigma,
                )
                ax.plot(
                    x,
                    density,
                    color=colors[trajectory],
                    label=labels[trajectory],
                )
            ax.set_title(f"Hessian ESD: {key}")
            ax.set_xlabel("Eigenvalue")
            ax.set_ylabel("Density")
            ax.set_xlim(*config.esd_xlim)
            ax.set_ylim(*config.esd_ylim)
            ax.set_xticks(
                np.arange(
                    config.esd_xlim[0],
                    config.esd_xlim[1] + 0.5 * config.esd_xtick_step,
                    config.esd_xtick_step,
                )
            )
            ax.set_yticks(
                np.arange(
                    config.esd_ylim[0],
                    config.esd_ylim[1] + 0.5 * config.esd_ytick_step,
                    config.esd_ytick_step,
                )
            )
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0, 2].axis("off")
        axes[0, 3].axis("off")
        cov_row = 1
    else:
        cov_row = 0

    cov_panels = []
    for key in snapshot_keys[:2]:
        cov_panels.extend(
            [
                (key, "gradient", f"Gradient descent\n{key}"),
                (key, "brownian", f"Brownian motion\n{key}"),
            ]
        )
    for col, (snap_key, trajectory, title) in enumerate(cov_panels):
        ax = axes[cov_row, col]
        cov = reorder_by_kmeans(snapshots[snap_key]["covariance"][trajectory])
        im = ax.imshow(
            cov,
            cmap="coolwarm_r",
            vmin=config.covariance_vmin,
            vmax=config.covariance_vmax,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(
            [
                config.covariance_vmin,
                0.0,
                config.covariance_vmax,
            ]
        )

    fig.suptitle(
        "Lyle et al. Figure 2 reproduction: true-label MNIST classification MDP",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {path}")

    return fig


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-hessian", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--prefill-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--target-update-period", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--cov-batch-size", type=int)
    parser.add_argument("--probe-batch-size", type=int)
    parser.add_argument("--lanczos-iter", type=int)
    parser.add_argument("--lanczos-vectors", type=int)
    parser.add_argument("--output-path", type=str)
    return parser


def config_from_args(args: argparse.Namespace) -> Figure2Config:
    values = asdict(Figure2Config())
    if args.paper_scale:
        values.update(PAPER_SCALE_OVERRIDES)

    for key in [
        "train_steps",
        "prefill_steps",
        "batch_size",
        "replay_capacity",
        "target_update_period",
        "hidden_dim",
        "cov_batch_size",
        "probe_batch_size",
        "lanczos_iter",
        "lanczos_vectors",
        "output_path",
    ]:
        value = getattr(args, key)
        if value is not None:
            values[key] = value

    values["download"] = bool(args.download)
    values["compute_hessian"] = not bool(args.no_hessian)
    return Figure2Config(**values)


def main() -> None:
    args = make_arg_parser().parse_args()
    config = config_from_args(args)
    results = run_experiment(config)
    plot_results(results, config.output_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
