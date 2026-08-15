"""Recreate the report's Hessian-eigenvector saddle illustration from scratch.

Renders f(x, y) = 0.4x^3 - x*y^3 + 1.2x^2 - 1.5y^2 + 0.3xy around the origin,
marks the saddle point at (0, 0), and draws the principal eigenvectors of the
Hessian there.  Everything numerical (gradient, Hessian, eigenvalues,
eigenvectors) is computed in code -- nothing is copied from the reference image
-- and the analytic derivatives are cross-checked against finite differences at
runtime, so the printed panel is guaranteed to match the plotted surface.

Run from the repository root::

    python scripts/plot_hessian_saddle.py                 # 600 dpi PNG + PDF + SVG
    python scripts/plot_hessian_saddle.py --dpi 1200      # even larger raster
    python scripts/plot_hessian_saddle.py --out-dir /tmp  # elsewhere

The PDF/SVG exports are vector graphics -- resolution-independent, so they stay
sharp at any zoom and are the best choice for the LaTeX report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def f(x, y):
    return 0.4 * x**3 - x * y**3 + 1.2 * x**2 - 1.5 * y**2 + 0.3 * x * y


def gradient(x, y):
    fx = 1.2 * x**2 - y**3 + 2.4 * x + 0.3 * y
    fy = -3.0 * x * y**2 - 3.0 * y + 0.3 * x
    return np.array([fx, fy])


def hessian(x, y):
    fxx = 2.4 * x + 2.4
    fyy = -6.0 * x * y - 3.0
    fxy = -3.0 * y**2 + 0.3
    return np.array([[fxx, fxy], [fxy, fyy]])


def check_derivatives(x0, y0, eps=1e-5):
    """Cross-check the analytic gradient/Hessian with central differences."""
    num_grad = np.array([
        (f(x0 + eps, y0) - f(x0 - eps, y0)) / (2 * eps),
        (f(x0, y0 + eps) - f(x0, y0 - eps)) / (2 * eps),
    ])
    num_hess = np.empty((2, 2))
    num_hess[0, 0] = (f(x0 + eps, y0) - 2 * f(x0, y0) + f(x0 - eps, y0)) / eps**2
    num_hess[1, 1] = (f(x0, y0 + eps) - 2 * f(x0, y0) + f(x0, y0 - eps)) / eps**2
    num_hess[0, 1] = num_hess[1, 0] = (
        f(x0 + eps, y0 + eps) - f(x0 + eps, y0 - eps)
        - f(x0 - eps, y0 + eps) + f(x0 - eps, y0 - eps)
    ) / (4 * eps**2)

    assert np.allclose(gradient(x0, y0), num_grad, atol=1e-6), "gradient mismatch"
    assert np.allclose(hessian(x0, y0), num_hess, atol=1e-4), "Hessian mismatch"


def principal_directions(matrix):
    """Eigenvalues (descending) and unit eigenvectors with a fixed sign convention."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # eigh's signs are arbitrary; make v1 point toward +x and v2 toward +y so the
    # arrows render consistently from run to run.
    if eigenvectors[0, 0] < 0:
        eigenvectors[:, 0] = -eigenvectors[:, 0]
    if eigenvectors[1, 1] < 0:
        eigenvectors[:, 1] = -eigenvectors[:, 1]
    return eigenvalues, eigenvectors


RED, BLUE = "#d62728", "#1f77b4"


def draw_figure(point=(0.0, 0.0)):
    x0, y0 = point
    check_derivatives(x0, y0)

    grad = gradient(x0, y0)
    hess = hessian(x0, y0)
    eigenvalues, eigenvectors = principal_directions(hess)
    assert np.allclose(grad, 0.0), "(0, 0) must be a critical point"
    assert eigenvalues[0] > 0 > eigenvalues[1], "(0, 0) must be a saddle"

    print(f"gradient at {point}: {grad}  (critical point)")
    print(f"H{point} =\n{hess}")
    print(f"eigenvalues: {eigenvalues.round(4)}")
    print(f"eigenvectors (columns):\n{eigenvectors.round(4)}")

    # The cubic terms explode away from the origin (f(2, 0) = 8 already), so a
    # +/-2.5 window turns the plot into vertical walls. +/-1.25 keeps the true
    # surface inside a reference-like +/-3 z-range while showing the saddle.
    grid = np.linspace(-1.25, 1.25, 220)
    xs, ys = np.meshgrid(grid, grid)
    zs = f(xs, ys)

    fig = plt.figure(figsize=(12.0, 9.0))
    ax = fig.add_axes((0.02, 0.03, 0.72, 0.86), projection="3d")
    # Draw order decides visibility: surface first, then the marker and arrows,
    # so they render on top instead of z-fighting with the surface.
    ax.computed_zorder = False
    ax.plot_surface(
        xs, ys, zs, cmap="coolwarm", rstride=2, cstride=2,
        linewidth=0, antialiased=True, alpha=0.92,
    )

    z0 = f(x0, y0)
    lift = 0.06
    ax.scatter([x0], [y0], [z0 + lift], color="black", s=45, depthshade=False)
    arrow_length = 0.95
    for idx, color in ((0, RED), (1, BLUE)):
        vx, vy = eigenvectors[:, idx]
        ax.quiver(
            x0, y0, z0 + lift, vx, vy, 0.0,
            length=arrow_length, color=color, linewidth=3.0, arrow_length_ratio=0.16,
        )

    ax.set_xlabel("X", fontsize=12, labelpad=8)
    ax.set_ylabel("Y", fontsize=12, labelpad=8)
    ax.set_zlabel("Z", fontsize=12, labelpad=6)
    ax.set_zlim(-3.4, 3.0)
    ax.view_init(elev=20, azim=-58)

    # fig.suptitle("Random 3D Function with Hessian Eigenvectors at (0, 0)",
    #              fontsize=16, fontweight="bold", x=0.44, y=0.965)
    fig.text(0.44, 0.9,
             r"$f(x,\,y) = 0.4x^3 \;-\; xy^3 \;+\; 1.2x^2 \;-\; 1.5y^2 \;+\; 0.3xy$",
             fontsize=16, fontweight="bold", ha="center")

    legend_handles = [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=8,
               label="Saddle Point (0, 0)"),
        Line2D([], [], color=RED, linewidth=2.8,
               label="Eigenvector $v_1$\n($\\lambda_1 > 0$, direction of\npositive curvature)"),
        Line2D([], [], color=BLUE, linewidth=2.8,
               label="Eigenvector $v_2$\n($\\lambda_2 < 0$, direction of\nnegative curvature)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.90),
               fontsize=11, labelspacing=1.1, borderpad=0.9, framealpha=0.95,
               edgecolor="0.6")

    # Info panel: Hessian, eigenvalues, eigenvectors -- all computed above.
    panel_x = 0.765
    box = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="0.6")
    matrix_text = (
        f"H(0,0) = ⌈ {hess[0, 0]:5.2f}  {hess[0, 1]:5.2f} ⌉\n"
        f"         ⌊ {hess[1, 0]:5.2f}  {hess[1, 1]:5.2f} ⌋"
    )
    fig.text(panel_x, 0.415, matrix_text, fontsize=11.5, family="monospace",
             va="top", bbox=box)
    fig.text(panel_x, 0.325, "Eigenvalues:", fontsize=11.5, va="top")
    fig.text(panel_x + 0.015, 0.293,
             f"$\\lambda_1 \\approx {eigenvalues[0]:.2f}$   ($> 0$)",
             fontsize=11.5, color=RED, va="top")
    fig.text(panel_x + 0.015, 0.261,
             f"$\\lambda_2 \\approx {eigenvalues[1]:.2f}$   ($< 0$)",
             fontsize=11.5, color=BLUE, va="top")
    fig.text(panel_x, 0.222, "Eigenvectors (unit):", fontsize=11.5, va="top")
    fig.text(panel_x + 0.015, 0.188,
             f"$v_1 \\approx [{eigenvectors[0, 0]:.3f},\\; {eigenvectors[1, 0]:.3f}]^T$",
             fontsize=11.5, color=RED, va="top")
    fig.text(panel_x + 0.015, 0.150,
             f"$v_2 \\approx [{eigenvectors[0, 1]:.3f},\\; {eigenvectors[1, 1]:.3f}]^T$",
             fontsize=11.5, color=BLUE, va="top")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpi", type=int, default=1200,
                        help="raster resolution for the PNG export (default: 600)")
    parser.add_argument("--out-dir", type=Path, default=Path("../reports/assets"),
                        help="output directory (default: reports/assets)")
    parser.add_argument("--stem", default="hessian_saddle_eigenvectors",
                        help="output filename stem")
    args = parser.parse_args()

    fig = draw_figure()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = args.out_dir / f"{args.stem}.{suffix}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"saved {path}" + (f"  ({args.dpi} dpi)" if suffix == "png" else "  (vector)"))


if __name__ == "__main__":
    main()
