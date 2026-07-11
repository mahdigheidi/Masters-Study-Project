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


def get_esd_plot(eigenvalues, weights, iter, seed, out_dir="."):
    eigenvalues = np.real(np.asarray(eigenvalues))
    density, grids = density_generate(eigenvalues, weights)
    plt.title(f'Eigenvalue Density (Iteration {iter}, Seed {seed})', fontsize=10)
    # Lyle et al. (Figure 2) plot the Ghorbani et al. Lanczos/Gaussian-kernel
    # density estimate on linear axes, not Ghorbani et al.'s own log-scale
    # convention -- match the paper we're reproducing here.
    plt.plot(grids, density)
    plt.ylabel('Density', fontsize=10, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=10, labelpad=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'esd_plot_{iter}_{seed}.png', dpi=300)


def density_generate(eigenvalues,
                     weights,
                     num_bins=2000,
                     sigma_squared=1e-4,
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
