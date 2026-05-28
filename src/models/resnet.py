import torchvision.models as models
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, num_actions=10):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_actions,
        )

    def forward(self, x):
        return self.model(x)