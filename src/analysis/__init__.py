"""Analysis utilities for plasticity and loss-geometry notebooks."""

from .hessian_ghorbani import (
    SLQResult,
    gradient_covariance_for_probe_loss,
    hessian_spectral_density,
    make_probe_loss_closure,
    make_probe_targets,
    probe_adaptation_curve,
    probe_loss_from_targets,
)

__all__ = [
    "SLQResult",
    "gradient_covariance_for_probe_loss",
    "hessian_spectral_density",
    "make_probe_loss_closure",
    "make_probe_targets",
    "probe_adaptation_curve",
    "probe_loss_from_targets",
]
