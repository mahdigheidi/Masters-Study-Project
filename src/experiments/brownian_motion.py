import torch


@torch.no_grad()
def brownian_update(model, step_norm):
    """Perturb every parameter with noise whose combined L2 norm equals ``step_norm``.

    Scaling is computed across the whole flattened parameter vector (not
    per-tensor), so this is directly comparable to an SGD step of the same
    global update norm.
    """
    if step_norm <= 0:
        return

    noise = [torch.randn_like(p) for p in model.parameters()]
    total_norm = torch.sqrt(sum((n * n).sum() for n in noise))
    scale = step_norm / (float(total_norm.item()) + 1e-12)

    for p, n in zip(model.parameters(), noise):
        p.add_(n * scale)


def compute_update_norm(model_before, model_after):
    total = 0.0

    for p1, p2 in zip(
        model_before.parameters(),
        model_after.parameters(),
    ):
        total += ((p1 - p2) ** 2).sum()

    return total.sqrt()
