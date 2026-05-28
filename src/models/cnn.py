import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        input_channels=1,
        num_actions=10,
        use_layernorm=False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64 * 22 * 22, 256)
        self.fc2 = nn.Linear(256, 256)

        self.output = nn.Linear(256, num_actions)

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.ln1 = nn.LayerNorm([64, 24, 24])
            self.ln2 = nn.LayerNorm([64, 22, 22])
            self.ln3 = nn.LayerNorm(256)
            self.ln4 = nn.LayerNorm(256)

    def forward(self, x):
        x = self.conv1(x)

        if self.use_layernorm:
            x = self.ln1(x)

        x = F.relu(x)

        x = self.conv2(x)

        if self.use_layernorm:
            x = self.ln2(x)

        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        if self.use_layernorm:
            x = self.ln3(x)

        x = F.relu(x)

        x = self.fc2(x)

        if self.use_layernorm:
            x = self.ln4(x)

        x = F.relu(x)

        return self.output(x)