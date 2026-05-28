import torch


@torch.no_grad()
def compute_weight_rank(weight_matrix, threshold=1e-5):
    _, S, _ = torch.linalg.svd(weight_matrix)

    return (S > threshold).sum().item()