class StopGradNoiseLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, target=None):
        eps = torch.randn_like(output)

        loss = (output - output.detach() + eps) ** 2

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        return loss