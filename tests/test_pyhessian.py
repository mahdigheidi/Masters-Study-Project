"""Validate the vendored PyHessian (Ghorbani et al.) estimators against a
closed-form answer.

For ``model(x) = w * x`` with MSE loss ``mean((w*x - t)^2)``, the Hessian
w.r.t. the scalar parameter ``w`` is the 1x1 matrix ``2 * mean(x**2)``. With a
single parameter the trace and the top eigenvalue both equal that scalar
exactly, so both estimators should recover it (Hutchinson's trace estimator
is exact for a 1-dimensional Hessian, and one-parameter power iteration
converges immediately).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.pyhessian import Hessian
from src.pyhessian.density_plot import density_generate


def _build_scalar_regression_problem():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)

    x = torch.tensor([[1.0], [2.0], [3.0]])
    t = torch.tensor([[1.0], [1.0], [1.0]])
    criterion = nn.MSELoss()

    analytic_hessian = float(2.0 * (x**2).mean())
    return model, criterion, (x, t), analytic_hessian


def test_trace_matches_analytic_hessian_for_scalar_linear_model():
    model, criterion, data, analytic_hessian = _build_scalar_regression_problem()

    hessian_comp = Hessian(model, criterion, data=data, cuda=False)
    trace_estimates = hessian_comp.trace(maxIter=50, tol=1e-4)

    assert trace_estimates[-1] == pytest.approx(analytic_hessian, rel=1e-3)


def test_top_eigenvalue_matches_analytic_hessian_for_scalar_linear_model():
    model, criterion, data, analytic_hessian = _build_scalar_regression_problem()

    hessian_comp = Hessian(model, criterion, data=data, cuda=False)
    eigenvalues, _ = hessian_comp.eigenvalues(maxIter=50, tol=1e-5, top_n=1)

    assert eigenvalues[0] == pytest.approx(analytic_hessian, rel=1e-3)


def test_density_places_its_mass_near_the_analytic_eigenvalue():
    model, criterion, data, analytic_hessian = _build_scalar_regression_problem()

    hessian_comp = Hessian(model, criterion, data=data, cuda=False)
    eigen_lists, weight_lists = hessian_comp.density(iter=5, n_v=1)

    eigenvalues = eigen_lists[0]
    weights = weight_lists[0]
    weighted_mean = sum(e * w for e, w in zip(eigenvalues, weights))

    assert weighted_mean == pytest.approx(analytic_hessian, rel=1e-2)
    assert sum(weights) == pytest.approx(1.0, rel=1e-6)


def test_density_generate_integrates_to_one():
    # Mirrors a real SLQ result: most weight concentrated at/near eigenvalue 0.
    eigenvalues = [[0.0, 0.05, -0.1, 5.0, 10.0]]
    weights = [[0.6, 0.15, 0.1, 0.1, 0.05]]

    density, grids = density_generate(eigenvalues, weights, num_bins=2000)

    integral = np.sum(density) * (grids[1] - grids[0])
    assert integral == pytest.approx(1.0, rel=1e-3)


def test_density_generate_wider_kernel_lowers_the_peak():
    # A too-narrow kernel renders concentrated SLQ weight as an artificially
    # tall, needle-thin spike (the bug this test guards against); widening
    # sigma_squared must spread the same probability mass into a lower peak.
    eigenvalues = [[0.0, 0.05, -0.1, 5.0, 10.0]]
    weights = [[0.6, 0.15, 0.1, 0.1, 0.05]]

    narrow_density, _ = density_generate(eigenvalues, weights, sigma_squared=1e-4)
    wide_density, _ = density_generate(eigenvalues, weights, sigma_squared=0.02)

    assert wide_density.max() < narrow_density.max()
