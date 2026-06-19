import copy
import torch


@torch.no_grad()
def brownian_update(model, step_norm):
    for p in model.parameters():
        noise = torch.randn_like(p)

        noise = noise / noise.norm()

        noise = noise * step_norm

        p.add_(noise)


def compute_update_norm(model_before, model_after):
    total = 0.0

    for p1, p2 in zip(
        model_before.parameters(),
        model_after.parameters(),
    ):
        total += ((p1 - p2) ** 2).sum()

    return total.sqrt()


def run_brownian_experiment(
    model,
    optimizer,
    loss_fn,
    dataloader,
    steps=100,
):
    gd_model = copy.deepcopy(model)
    brownian_model = copy.deepcopy(model)

    for step in range(steps):
        x, y = next(iter(dataloader))

        # ----- gradient descent -----

        old_model = copy.deepcopy(gd_model)

        optimizer.zero_grad()

        logits = gd_model(x)

        loss = loss_fn(logits, y)

        loss.backward()

        optimizer.step()

        update_norm = compute_update_norm(old_model, gd_model)

        # ----- brownian motion -----

        brownian_update(brownian_model, update_norm)

        print(f"Step {step} complete")