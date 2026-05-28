import torch


@torch.no_grad()
def compute_weight_norm(model):
    total_norm = 0.0

    for p in model.parameters():
        total_norm += p.norm(2).item() ** 2

    return total_norm ** 0.5