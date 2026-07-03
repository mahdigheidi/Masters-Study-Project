"""
Reproduction of Figure 2 — Lyle et al., ICML 2023
"Understanding Plasticity in Neural Networks"  arXiv:2303.01486

FIX: pyHessian's density() calls torch.eig() which was removed in PyTorch 2.x.
     We replace the broken call with a self-contained HessianESD class that
     implements the same Stochastic Lanczos Quadrature (SLQ) algorithm using
     only modern torch.linalg.eigh(). No pyHessian import needed.

Hessian-vector products use torch.autograd.grad (not backward+create_graph),
which avoids the memory-leak warning from the reference cycle in newer PyTorch.

Install:
    pip install torch torchvision scipy matplotlib
"""

import copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42);  np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL  (2-layer MLP, paper Appendix A.1)
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA  — random-label MNIST
# ══════════════════════════════════════════════════════════════════════════════

def get_mnist_random_labels(batch_size=512, probe_size=256, seed=42):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=tf)
    rng = np.random.RandomState(seed)
    dataset.targets = torch.tensor(rng.randint(0, 10, len(dataset.targets)))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    probe_sub    = Subset(dataset, list(range(probe_size)))
    probe_loader = DataLoader(probe_sub, batch_size=probe_size, shuffle=False)
    probe_X, _   = next(iter(probe_loader))
    return train_loader, probe_X.to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROBE LOSS  (paper Section 4.2)
#     ℓ_probe(θ) = ‖f_θ(X) − sg[f_θ(X) + ε]‖²,   ε ~ N(0,1)
#     The stop-gradient target is pre-computed under torch.no_grad().
# ══════════════════════════════════════════════════════════════════════════════

def make_probe_targets(model, X):
    """Pre-compute stop-gradient targets. Call once before any grad computation."""
    with torch.no_grad():
        return (model(X) + torch.randn_like(model(X))).detach()

def probe_loss_from_targets(model, X, targets):
    """Probe loss given pre-computed stop-gradient targets."""
    return F.mse_loss(model(X), targets)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HESSIAN ESD  — self-contained SLQ, no pyHessian
#
#  What pyHessian's density() does (and what we replicate):
#    For each of n_v Rademacher random vectors v:
#      Run n_iter steps of the Lanczos algorithm to build a tridiagonal T.
#      Compute eigvals/vecs of T with torch.linalg.eigh  ← the fixed call.
#      The squared first component of each eigenvector is the SLQ weight.
#    Return all (eigenvalue, weight) pairs; the caller smooths them into a curve.
#
#  The only change vs pyHessian: torch.eig → torch.linalg.eigh.
#  We also use torch.autograd.grad for HVPs (cleaner, no backward cycle warning).
# ══════════════════════════════════════════════════════════════════════════════

def _flat_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def _hvp(loss_fn, params, v):
    """
    Hessian-vector product via double autodiff.
    Uses torch.autograd.grad throughout to avoid the backward() reference-cycle warning.
    """
    grads = torch.autograd.grad(loss_fn(), params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    gv   = (flat_grad * v.detach()).sum()
    hvp  = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([h.reshape(-1).detach() for h in hvp])

def compute_esd(model, X, targets, n_iter=100, n_v=3):
    """
    Stochastic Lanczos Quadrature for the Hessian empirical spectral density.
    Replicates pyHessian density() with the single fix:
        torch.eig  →  torch.linalg.eigh
    (T is symmetric by construction so eigh is exact and cheaper than eig.)

    Returns: eigenvalues (np.array), weights (np.array) — concatenated over n_v.
    """
    model.eval()
    params = _flat_params(model)
    d      = sum(p.numel() for p in params)

    all_eig, all_w = [], []

    for _ in range(n_v):
        # Rademacher random vector
        v = torch.randint(0, 2, (d,), device=DEVICE).float() * 2 - 1
        v /= v.norm()

        alpha_list, beta_list = [], []
        v_prev = torch.zeros_like(v)

        for j in range(n_iter):
            # loss_fn is a closure so HVP uses fresh graph each call
            loss_fn = lambda: probe_loss_from_targets(model, X, targets)
            Hv = _hvp(loss_fn, params, v)

            alpha = (v * Hv).sum().item()   # dot product v·Hv, then scalar
            alpha_list.append(alpha)

            w = Hv - alpha * v - (beta_list[-1] * v_prev if beta_list else 0)

            beta = w.norm().item()
            if beta < 1e-6:
                break
            beta_list.append(beta)
            v_prev = v.clone()
            v = w / beta

        # Build symmetric tridiagonal T
        n = len(alpha_list)
        T = torch.zeros(n, n)
        for i in range(n):
            T[i, i] = alpha_list[i]
        for i, b in enumerate(beta_list):
            if i + 1 < n:
                T[i, i+1] = b
                T[i+1, i] = b

        # ── THE FIX ───────────────────────────────────────────────────────────
        # pyHessian:  a_, b_ = torch.eig(T, eigenvectors=True)   # REMOVED in PyTorch 2.x
        # Fixed:      torch.linalg.eigh is correct for real symmetric matrices
        #             and returns real eigenvalues directly (no complex casting needed).
        eigvals, eigvecs = torch.linalg.eigh(T)          # ← modern replacement
        # ─────────────────────────────────────────────────────────────────────

        weights = eigvecs[0, :] ** 2   # squared first-row components = SLQ weights

        all_eig.append(eigvals.numpy())
        all_w.append(weights.numpy())

    return np.concatenate(all_eig), np.concatenate(all_w)


def esd_to_curve(eigenvalues, weights, n_points=600, sigma=0.5):
    """Smooth raw SLQ output into a plottable density curve (KDE over Dirac deltas)."""
    order = np.argsort(eigenvalues)
    eigenvalues, weights = eigenvalues[order], weights[order]
    x = np.linspace(eigenvalues.min() - 1, eigenvalues.max() + 1, n_points)
    density = np.zeros_like(x)
    for lam, w in zip(eigenvalues, weights):
        density += w * np.exp(-0.5 * ((x - lam) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    dx = x[1] - x[0]
    density /= (density.sum() * dx + 1e-10)
    return x, density


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GRADIENT COVARIANCE  (paper eq. 3)
# ══════════════════════════════════════════════════════════════════════════════

def gradient_covariance(model, probe_X, targets, k=64):
    """k×k cosine-similarity matrix of per-sample probe-loss gradients."""
    model.eval()
    X_sub, t_sub = probe_X[:k], targets[:k]
    grads = []
    for i in range(k):
        model.zero_grad()
        loss = probe_loss_from_targets(model, X_sub[i:i+1], t_sub[i:i+1])
        loss.backward()
        g = torch.cat([p.grad.reshape(-1).clone()
                       for p in model.parameters() if p.grad is not None])
        grads.append(g)
    model.zero_grad()
    G     = torch.stack(grads)
    norms = G.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return ((G / norms) @ (G / norms).T).detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING + BROWNIAN MOTION
# ══════════════════════════════════════════════════════════════════════════════

def gd_step(model, opt, X, y):
    model.train()
    opt.zero_grad()
    F.cross_entropy(model(X), y).backward()
    opt.step()

def brownian_step(model, step_norm):
    with torch.no_grad():
        noise = [torch.randn_like(p) for p in model.parameters()]
        scale = step_norm / (torch.sqrt(sum((n**2).sum() for n in noise)) + 1e-10)
        for p, n in zip(model.parameters(), noise):
            p.add_(n * scale)

def update_norm(m_before, m_after):
    return torch.sqrt(sum(
        ((pa.data - pb.data)**2).sum()
        for pb, pa in zip(m_before.parameters(), m_after.parameters())
    )).item()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    n_target_updates = 5,
    iters_per_update = 20,
    k_cov            = 64,    # set 512 to match paper exactly
    n_iter_lanczos   = 100,
    n_v_lanczos      = 3,
    snapshot_iters   = (1, 20),
):
    train_loader, probe_X = get_mnist_random_labels()
    data_iter = iter(train_loader)
    def next_batch():
        nonlocal data_iter
        try:    return next(data_iter)
        except: data_iter = iter(train_loader); return next(data_iter)

    model_gd = MLP().to(DEVICE)
    model_bm = copy.deepcopy(model_gd)
    opt_gd   = torch.optim.SGD(model_gd.parameters(), lr=0.01)

    results = {"init": {}, "after": {}, "cov": {"gd": {}, "bm": {}}}

    # ── ESD at init ────────────────────────────────────────────────────────
    print("ESD at initialization …")
    for model, key in [(model_gd, "gd"), (model_bm, "bm")]:
        tgts = make_probe_targets(model, probe_X)
        ev, ew = compute_esd(model, probe_X, tgts, n_iter_lanczos, n_v_lanczos)
        results["init"][key] = (ev, ew)

    # ── Coupled trajectories ───────────────────────────────────────────────
    for upd in range(n_target_updates):
        print(f"Target update {upd+1}/{n_target_updates}")
        for loc_iter in range(1, iters_per_update + 1):
            X_b, y_b = next_batch()
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

            m_before = copy.deepcopy(model_gd)
            gd_step(model_gd, opt_gd, X_b, y_b)
            step_norm = update_norm(m_before, model_gd)
            brownian_step(model_bm, step_norm)

            if upd == 0 and loc_iter in snapshot_iters:
                tag = f"iter{loc_iter}"
                print(f"  Gradient covariance @ local iter {loc_iter} …")
                for model, key in [(model_gd, "gd"), (model_bm, "bm")]:
                    tgts = make_probe_targets(model, probe_X)
                    results["cov"][key][tag] = gradient_covariance(
                        model, probe_X, tgts, k_cov)

    # ── ESD after training ─────────────────────────────────────────────────
    print("ESD after target updates …")
    for model, key in [(model_gd, "gd"), (model_bm, "bm")]:
        tgts = make_probe_targets(model, probe_X)
        ev, ew = compute_esd(model, probe_X, tgts, n_iter_lanczos, n_v_lanczos)
        results["after"][key] = (ev, ew)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COL_GD, COL_BM = "#2166ac", "#d6604d"

def plot_esd(ax, records, title):
    for ev, ew, col, lbl in records:
        x, dens = esd_to_curve(ev, ew)
        ax.plot(x, dens, color=col, lw=1.8, label=lbl)
        ax.fill_between(x, dens, alpha=0.10, color=col)
    ax.set_xlim(-2, 50);  ax.set_ylim(bottom=0)
    ax.set_xlabel("Eigenvalue", fontsize=9);  ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, framealpha=0.5)

def make_figure2(results, save_path="lyle2023_figure2_fixed.png"):
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.38, hspace=0.45,
                            left=0.07, right=0.97, top=0.91, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_esd(ax1, [
        (*results["init"]["gd"],  COL_GD, "Gradient Descent"),
        (*results["init"]["bm"],  COL_BM, "Brownian Motion"),
    ], "Initialization")

    plot_esd(ax2, [
        (*results["after"]["gd"], COL_GD, "Gradient Descent"),
        (*results["after"]["bm"], COL_BM, "Brownian Motion"),
    ], "After 5 target updates")

    cov_panels = [
        ("gd", "iter1",  "Gradient Descent\n1 Iteration"),
        ("bm", "iter1",  "Brownian Motion\n1 Iteration"),
        ("gd", "iter20", "Gradient Descent\n20 Iterations"),
        ("bm", "iter20", "Brownian Motion\n20 Iterations"),
    ]
    for col_idx, (traj, key, title) in enumerate(cov_panels):
        ax = fig.add_subplot(gs[1, col_idx])
        C  = results["cov"][traj].get(key)
        if C is not None:
            im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Input index", fontsize=8)
        ax.set_ylabel("Input index", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.text(0.01, 0.75, "Loss landscape\ncurvature", va="center", fontsize=9, fontweight="bold")
    fig.text(0.01, 0.30, "Gradient\nCovariance",      va="center", fontsize=9, fontweight="bold")
    fig.suptitle(
        "Figure 2 — Lyle et al. 2023  (torch.linalg.eigh fix, no pyHessian)\n"
        "Hessian ESD and Gradient Covariance: Gradient Descent vs. Brownian Motion",
        fontsize=10, fontweight="bold", y=0.99)

    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_experiment(
        n_target_updates = 5,
        iters_per_update = 2000,
        k_cov            = 64,    # increase to 512 to match paper exactly
        n_iter_lanczos   = 100,
        n_v_lanczos      = 3,
        snapshot_iters   = (1, 20),
    )
    make_figure2(results)