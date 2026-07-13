#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# `iter` shadows the builtin, but it mirrors the vendored PyHessian parameter
# name used by existing notebook callers -- renaming would break them.
def get_esd_plot(  # pylint: disable=redefined-builtin
    curves, iter, seed, out_dir=".", sigma_squared=0.02, num_bins=2000
):
    """Overlay one or more named eigenvalue-density curves on the same axes.

    ``curves`` is a sequence of ``(eigenvalues, weights, label, color)``
    tuples -- e.g. one for the gradient-descent trajectory and one for the
    Brownian-motion trajectory (Lyle et al. Figure 2 compares exactly these
    two on the same panel, distinguished by color and a legend).
    """
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'Eigenvalue Density (Iteration {iter}, Seed {seed})', fontsize=11)

    for raw_eigenvalues, weights, label, color in curves:
        eigenvalues = np.real(np.asarray(raw_eigenvalues))
        density, grids = density_generate(
            eigenvalues, weights, num_bins=num_bins, sigma_squared=sigma_squared,
        )
        # Left: linear axes, matching how Lyle et al. (Figure 2) present this
        # estimate -- see the density_generate docstring/comment for why the
        # bandwidth is tuned the way it is.
        ax_linear.plot(grids, density, color=color, label=label)
        # Right: log-scale density (Ghorbani et al.'s own plotting convention).
        # Outlier eigenvalues can carry a vanishingly small fraction of the SLQ
        # weight (often <0.01% here) -- too small to render as a visible bump
        # on a linear axis no matter how the kernel is tuned, but real and
        # visible once density is put on a log scale.
        ax_log.semilogy(grids, density + 1e-12, color=color, label=label)

    ax_linear.set_title('Linear scale', fontsize=9)
    ax_linear.set_ylabel('Density', fontsize=10, labelpad=10)
    ax_linear.set_xlabel('Eigenvalue', fontsize=10, labelpad=10)
    ax_linear.tick_params(labelsize=8)
    ax_linear.legend(fontsize=8)

    ax_log.set_title('Log scale (reveals low-weight outliers)', fontsize=9)
    ax_log.set_ylabel('Density (log scale)', fontsize=10, labelpad=10)
    ax_log.set_xlabel('Eigenvalue', fontsize=10, labelpad=10)
    ax_log.tick_params(labelsize=8)
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'esd_plot_{iter}_{seed}.png', dpi=300)


def density_generate(eigenvalues,
                     weights,
                     num_bins=2000,
                     # Small/under-parameterized networks (like the toy MLPs in this
                     # repo) can have a large fraction of the Lanczos/SLQ weight
                     # concentrated at a single near-zero node. The upstream
                     # PyHessian default (1e-5) was tuned for much larger models and
                     # renders that as an extremely tall, needle-thin spike here
                     # instead of a legible density curve -- widen the kernel so the
                     # rendered scale is comparable to the reference figure.
                     sigma_squared=0.02,
                     overhead=0.01):

    eigenvalues = np.real(np.asarray(eigenvalues))
    weights = np.real(np.asarray(weights))

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
