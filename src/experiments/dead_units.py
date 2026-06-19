"""Dead ReLU measurements used in the Section 5 falsification experiments.

A unit is counted as dead on a probe batch when its post-ReLU activation is
never positive for any sampled input.  The helper registers hooks on explicit
``nn.ReLU`` modules, which is why the reusable model definitions use module
ReLUs instead of functional calls.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


def _as_batches(data) -> Iterable[torch.Tensor]:
    if isinstance(data, torch.Tensor):
        yield data
        return
    for batch in data:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


def _unit_alive(activation: torch.Tensor) -> torch.Tensor:
    if activation.dim() == 2:
        return (activation > 0).any(dim=0)
    if activation.dim() == 3:
        return (activation > 0).any(dim=(0, 1))
    return (activation > 0).any(dim=(0, *range(2, activation.dim())))


@torch.no_grad()
def dead_unit_statistics(
    model: nn.Module,
    data,
    device: torch.device | str | None = None,
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    alive_by_layer: Dict[str, torch.Tensor] = {}
    latest_activations: Dict[str, torch.Tensor] = {}
    hooks = []

    def hook_fn(name: str):
        def hook(_, __, output):
            latest_activations[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        for batch in _as_batches(data):
            latest_activations.clear()
            model(batch.to(device))
            for name, activation in latest_activations.items():
                alive = _unit_alive(activation).detach().cpu()
                if name not in alive_by_layer:
                    alive_by_layer[name] = alive.clone()
                else:
                    alive_by_layer[name] |= alive
    finally:
        for hook_handle in hooks:
            hook_handle.remove()

    total_units = int(sum(alive.numel() for alive in alive_by_layer.values()))
    dead_units = int(sum((~alive).sum().item() for alive in alive_by_layer.values()))

    return {
        "dead_units": float(dead_units),
        "total_units": float(total_units),
        "dead_fraction": float(dead_units / total_units) if total_units else 0.0,
    }


@torch.no_grad()
def compute_dead_units(
    model: nn.Module,
    data,
    device: torch.device | str | None = None,
    return_fraction: bool = False,
):
    stats = dead_unit_statistics(model, data, device=device)
    return stats["dead_fraction"] if return_fraction else stats["dead_units"]
