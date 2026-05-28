import torch


@torch.no_grad()
def compute_feature_rank(features, threshold=1e-5):
    """
    features: [batch, dim]
    """

    features = features - features.mean(0)

    _, S, _ = torch.linalg.svd(features)

    rank = (S > threshold).sum().item()

    return rank