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




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def load_true_label_mnist(data_root: str, download: bool) -> datasets.MNIST:
    transform = transforms.ToTensor()
    return datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )




def populate_replay(
    env: EasyMDP,
    model: QNetwork,
    replay: ReplayBuffer,
    steps: int,
    epsilon: float,
) -> None:
    for _ in range(steps):
        collect_transition(env, model, replay, epsilon)











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
    brownian_targets, _ = make_probe_targets(brownian_model, probe_inputs, noise=shared_noise)

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

    hessian_gd = Hessian(gd_model, F.mse_loss, data=(probe_inputs, gd_targets), cuda=False)
    hessian_bm = Hessian(brownian_model, F.mse_loss, data=(probe_inputs, brownian_targets), cuda=False)

    gd_eig, gd_weight = hessian_gd.eigenvalues()
    bm_eig, bm_weight = hessian_bm.eigenvalues()

    snapshot["esd"] = {
        "gradient": (gd_eig, gd_weight),
        "brownian": (bm_eig, bm_weight),
    }
    # snapshot["top_eigenvalue"] = {
    #     "gradient": float(gd_eig.max()),
    #     "brownian": float(bm_eig.max()),
    # }

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
    populate_replay(env, gd_model, replay, config.prefill_steps, epsilon=config.epsilon)

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

        collect_transition(env, gd_model, replay, epsilon=config.epsilon)

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
                probe_batch = replay.sample(config.probe_batch_size)
                probe_inputs = probe_batch[0]
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
            ax.set_title(f"Hessian Eigenvalue Spectral Density: {key}")
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

    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {path}")

    return fig
