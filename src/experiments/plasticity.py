"""Plasticity probes from Sections 2.2 and 3.1.

The paper measures whether a trained network can still update its predictions
toward arbitrary new learning signals on the inputs it has seen.  For a probe
batch ``X``, we sample a freshly initialized network ``f_omega`` and build
random regression targets ``g(X) = a + sin(1e5 * f_omega(X))``, where ``a`` is
the current network's mean prediction.  A cloned checkpoint is then optimized
on this target for a fixed budget, and the final loss is compared with the
loss obtained from an earlier or randomly initialized reference checkpoint.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
ModelFactory = Callable[[], nn.Module]


@dataclass
class PlasticityProbeConfig:
    steps: int = 500
    num_tasks: int = 10
    batch_size: Optional[int] = None
    target_scale: float = 1e5
    log_every: Optional[int] = None


@dataclass
class ProbeTaskResult:
    initial_loss: float
    final_loss: float
    learning_curve: List[Dict[str, float]]


@dataclass
class PlasticityResult:
    probe_loss: float
    baseline_probe_loss: Optional[float]
    plasticity_loss: Optional[float]
    task_losses: List[float]
    baseline_task_losses: Optional[List[float]]


def default_probe_optimizer(lr: float = 1e-3) -> OptimizerFactory:
    def build(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr)

    return build


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _sample_indices(inputs: torch.Tensor, batch_size: Optional[int]) -> Optional[torch.Tensor]:
    if batch_size is None or batch_size >= inputs.size(0):
        return None
    return torch.randint(0, inputs.size(0), (batch_size,), device=inputs.device)


@torch.no_grad()
def make_random_function_targets(
    model: nn.Module,
    random_model: nn.Module,
    inputs: torch.Tensor,
    target_scale: float = 1e5,
) -> torch.Tensor:
    model.eval()
    random_model.eval()
    inputs = inputs.to(_device_of(model))
    random_model = random_model.to(inputs.device)

    offset = model(inputs).mean(dim=0, keepdim=True)
    random_outputs = random_model(inputs)
    return (offset + torch.sin(target_scale * random_outputs)).detach()


def train_probe_task(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer_factory: OptimizerFactory,
    config: PlasticityProbeConfig,
) -> ProbeTaskResult:
    probe_model = copy.deepcopy(model).to(_device_of(model))
    inputs = inputs.to(_device_of(probe_model))
    targets = targets.to(_device_of(probe_model))
    optimizer = optimizer_factory(probe_model)
    learning_curve: List[Dict[str, float]] = []

    with torch.no_grad():
        initial_loss = float(F.mse_loss(probe_model(inputs), targets).cpu().item())

    if config.log_every is not None:
        learning_curve.append({"step": 0.0, "loss": initial_loss})

    probe_model.train()
    for step in tqdm(range(1, config.steps + 1)):
        # print(f"Probe training step {step}/{config.steps}")
        indices = _sample_indices(inputs, config.batch_size)
        if indices is None:
            batch_targets = targets
            batch_inputs = inputs
        else:
            batch_inputs = inputs.index_select(0, indices)
            batch_targets = targets.index_select(0, indices)

        loss = F.mse_loss(probe_model(batch_inputs), batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if config.log_every is not None and step % config.log_every == 0:
            with torch.no_grad():
                full_loss = F.mse_loss(probe_model(inputs), targets)
            learning_curve.append(
                {"step": float(step), "loss": float(full_loss.cpu().item())}
            )

    with torch.no_grad():
        final_loss = float(F.mse_loss(probe_model(inputs), targets).cpu().item())

    if config.log_every is not None and (
        not learning_curve or learning_curve[-1]["step"] != float(config.steps)
    ):
        learning_curve.append({"step": float(config.steps), "loss": final_loss})

    return ProbeTaskResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        learning_curve=learning_curve,
    )


def run_random_probe_task(
    model: nn.Module,
    random_model_factory: ModelFactory,
    inputs: torch.Tensor,
    optimizer_factory: OptimizerFactory,
    config: PlasticityProbeConfig,
) -> ProbeTaskResult:
    print(f"Running random probe task with {config.steps} steps and batch size {config.batch_size}.")
    random_model = random_model_factory().to(_device_of(model))
    targets = make_random_function_targets(
        model,
        random_model,
        inputs,
        target_scale=config.target_scale,
    )
    print(f"Random targets generated with shape {targets.shape}.")
    return train_probe_task(model, inputs, targets, optimizer_factory, config)


def estimate_probe_loss(
    model: nn.Module,
    random_model_factory: ModelFactory,
    inputs: torch.Tensor,
    optimizer_factory: OptimizerFactory,
    config: PlasticityProbeConfig,
) -> List[ProbeTaskResult]:
    return [
        run_random_probe_task(
            model,
            random_model_factory,
            inputs,
            optimizer_factory,
            config,
        )
        for _ in range(config.num_tasks)
    ]


def measure_plasticity_loss(
    model: nn.Module,
    random_model_factory: ModelFactory,
    inputs: torch.Tensor,
    optimizer_factory: OptimizerFactory,
    config: PlasticityProbeConfig,
    baseline_model: Optional[nn.Module] = None,
    baseline_probe_loss: Optional[float] = None,
) -> PlasticityResult:
    task_results = estimate_probe_loss(
        model,
        random_model_factory,
        inputs,
        optimizer_factory,
        config,
    )
    task_losses = [result.final_loss for result in task_results]
    probe_loss = float(np.mean(task_losses))

    baseline_task_losses = None
    if baseline_probe_loss is None and baseline_model is not None:
        baseline_results = estimate_probe_loss(
            baseline_model,
            random_model_factory,
            inputs,
            optimizer_factory,
            config,
        )
        baseline_task_losses = [result.final_loss for result in baseline_results]
        baseline_probe_loss = float(np.mean(baseline_task_losses))

    plasticity_loss = (
        None if baseline_probe_loss is None else float(probe_loss - baseline_probe_loss)
    )

    return PlasticityResult(
        probe_loss=probe_loss,
        baseline_probe_loss=baseline_probe_loss,
        plasticity_loss=plasticity_loss,
        task_losses=task_losses,
        baseline_task_losses=baseline_task_losses,
    )


def probe_learning_curves(
    checkpoints: Dict[str, nn.Module],
    random_model_factory: ModelFactory,
    inputs: torch.Tensor,
    optimizer_factory: OptimizerFactory,
    config: PlasticityProbeConfig,
) -> List[Dict[str, float]]:
    if config.log_every is None:
        raise ValueError("Set PlasticityProbeConfig.log_every to record learning curves.")

    rows: List[Dict[str, float]] = []
    for checkpoint_name, model in checkpoints.items():
        task_results = estimate_probe_loss(
            model,
            random_model_factory,
            inputs,
            optimizer_factory,
            config,
        )
        for task_id, result in enumerate(task_results):
            for point in result.learning_curve:
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "task_id": float(task_id),
                        "step": point["step"],
                        "loss": point["loss"],
                        "final_loss": result.final_loss,
                    }
                )
    return rows
