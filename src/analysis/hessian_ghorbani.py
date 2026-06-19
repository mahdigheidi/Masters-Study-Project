"""Hessian spectral density via the Ghorbani et al. SLQ procedure.

Ghorbani, Krishnan, and Xiao (2019) estimate neural-network Hessian spectra
with Hessian-vector products and stochastic Lanczos quadrature instead of
forming the full Hessian matrix.  This module implements that procedure for
small research notebooks: a loss closure defines the scalar objective,
automatic differentiation supplies Hessian-vector products, Lanczos builds a
small tridiagonal matrix, and Gaussian smoothing turns the resulting quadrature
nodes and weights into a plottable spectral density.

The same file also defines the stop-gradient probe loss used in
"Understanding Plasticity in Neural Networks":

    L_probe(theta) = || f_theta(X) - sg[f_theta0(X) + epsilon] ||^2.

The target is created once and detached before Hessian-vector products are
computed.  This is essential; resampling epsilon inside the closure would make
the Hessian estimate noisy and would not match the paper's probe objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LossClosure = Callable[[], torch.Tensor]
HVPFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class SLQResult:
    eigenvalues: np.ndarray
    weights: np.ndarray
    grid: np.ndarray
    density: np.ndarray


def trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def num_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in trainable_parameters(model))


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def make_rademacher_vector(
    dimension: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    vector = torch.randint(
        low=0,
        high=2,
        size=(dimension,),
        device=device,
        generator=generator,
    ).to(dtype=dtype)
    return vector.mul(2.0).sub(1.0)


def hessian_vector_product(
    model: nn.Module,
    loss_closure: LossClosure,
    vector: torch.Tensor,
) -> torch.Tensor:
    params = trainable_parameters(model)
    model.zero_grad(set_to_none=True)

    loss = loss_closure()
    gradients = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
    )
    flat_gradient = flatten_tensors(gradients)
    gradient_vector_product = torch.dot(flat_gradient, vector.detach())

    hvp = torch.autograd.grad(
        gradient_vector_product,
        params,
        retain_graph=False,
    )
    model.zero_grad(set_to_none=True)
    return flatten_tensors(hvp).detach()


def lanczos_tridiagonal(
    hvp_fn: HVPFunction,
    dimension: int,
    num_steps: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    initial_vector: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
    full_reorthogonalization: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_vector is None:
        vector = make_rademacher_vector(dimension, device=device, dtype=dtype)
    else:
        vector = initial_vector.to(device=device, dtype=dtype).clone()

    vector = vector / vector.norm().clamp_min(1e-12)
    previous_vector = torch.zeros_like(vector)
    previous_beta = torch.zeros((), device=device, dtype=dtype)

    alphas: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []
    basis: List[torch.Tensor] = []

    for step in range(num_steps):
        if full_reorthogonalization:
            basis.append(vector.clone())

        work = hvp_fn(vector)
        if step > 0:
            work = work - previous_beta * previous_vector

        alpha = torch.dot(vector, work)
        work = work - alpha * vector

        if full_reorthogonalization:
            for basis_vector in basis:
                work = work - torch.dot(work, basis_vector) * basis_vector

        beta = work.norm()
        alphas.append(alpha.detach())

        if beta.item() < tol or step == num_steps - 1:
            break

        betas.append(beta.detach())
        previous_vector = vector
        vector = work / beta.clamp_min(1e-12)
        previous_beta = beta

    return torch.stack(alphas), torch.stack(betas) if betas else torch.empty(0, device=device)


def tridiagonal_eigendecomposition(
    alphas: torch.Tensor,
    betas: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = alphas.numel()
    tridiagonal = torch.diag(alphas)
    if betas.numel() > 0:
        tridiagonal = tridiagonal + torch.diag(betas, diagonal=1)
        tridiagonal = tridiagonal + torch.diag(betas, diagonal=-1)

    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    quadrature_weights = eigenvectors[0].pow(2)
    return eigenvalues, quadrature_weights


def smooth_spectral_density(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    num_points: int = 600,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    weights = weights[order]

    low = float(eigenvalues.min())
    high = float(eigenvalues.max())
    width = max(high - low, 1e-8)
    grid = np.linspace(low - 0.05 * width, high + 0.05 * width, num_points)

    if sigma is None:
        sigma = max(0.01 * width, 1e-4)

    density = np.zeros_like(grid)
    normalizer = sigma * np.sqrt(2.0 * np.pi)
    for eigenvalue, weight in zip(eigenvalues, weights):
        density += weight * np.exp(-0.5 * ((grid - eigenvalue) / sigma) ** 2) / normalizer

    dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
    density = density / (density.sum() * dx + 1e-12)
    return grid, density


def hessian_spectral_density(
    model: nn.Module,
    loss_closure: LossClosure,
    num_lanczos_steps: int = 50,
    num_probe_vectors: int = 3,
    density_points: int = 600,
    density_sigma: Optional[float] = None,
    seed: Optional[int] = None,
    full_reorthogonalization: bool = True,
) -> SLQResult:
    params = trainable_parameters(model)
    if not params:
        raise ValueError("Model has no trainable parameters.")

    device = params[0].device
    dtype = params[0].dtype
    dimension = sum(parameter.numel() for parameter in params)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    eigenvalues: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    def hvp_fn(vector: torch.Tensor) -> torch.Tensor:
        return hessian_vector_product(model, loss_closure, vector)

    for _ in range(num_probe_vectors):
        initial_vector = make_rademacher_vector(
            dimension,
            device=device,
            dtype=dtype,
            generator=generator if seed is not None else None,
        )
        alphas, betas = lanczos_tridiagonal(
            hvp_fn,
            dimension=dimension,
            num_steps=num_lanczos_steps,
            device=device,
            dtype=dtype,
            initial_vector=initial_vector,
            full_reorthogonalization=full_reorthogonalization,
        )
        values, quadrature_weights = tridiagonal_eigendecomposition(alphas, betas)
        eigenvalues.append(values.detach().cpu().numpy())
        weights.append(quadrature_weights.detach().cpu().numpy())

    raw_eigenvalues = np.concatenate(eigenvalues)
    raw_weights = np.concatenate(weights)
    grid, density = smooth_spectral_density(
        raw_eigenvalues,
        raw_weights,
        num_points=density_points,
        sigma=density_sigma,
    )
    return SLQResult(
        eigenvalues=raw_eigenvalues,
        weights=raw_weights,
        grid=grid,
        density=density,
    )


@torch.no_grad()
def make_probe_targets(
    model: nn.Module,
    inputs: torch.Tensor,
    epsilon: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    outputs = model(inputs)
    if epsilon is None:
        epsilon = torch.randn_like(outputs)
    return (outputs + epsilon).detach(), epsilon.detach()


def probe_loss_from_targets(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(model(inputs), targets)


def make_probe_loss_closure(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> LossClosure:
    def closure() -> torch.Tensor:
        return probe_loss_from_targets(model, inputs, targets)

    return closure


def probe_adaptation_curve(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []

    for step in range(steps + 1):
        loss = probe_loss_from_targets(model, inputs, targets)
        losses.append(float(loss.detach().cpu().item()))

        if step == steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return losses


def gradient_covariance_for_probe_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    if max_samples is not None:
        inputs = inputs[:max_samples]
        targets = targets[:max_samples]

    gradients = []
    params = trainable_parameters(model)

    for x, target in zip(inputs, targets):
        model.zero_grad(set_to_none=True)
        loss = probe_loss_from_targets(
            model,
            x.unsqueeze(0),
            target.unsqueeze(0),
        )
        loss.backward()
        gradient = flatten_tensors(
            parameter.grad for parameter in params if parameter.grad is not None
        )
        gradients.append(gradient.detach())

    model.zero_grad(set_to_none=True)
    gradient_matrix = torch.stack(gradients)
    gradient_matrix = F.normalize(gradient_matrix, p=2, dim=1)
    return (gradient_matrix @ gradient_matrix.T).detach().cpu()
