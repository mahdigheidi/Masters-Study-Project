import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_actions=10,
        hidden_dim=512,
        use_layernorm=False,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_actions)

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        if self.use_layernorm:
            x = self.ln1(x)

        x = F.relu(x)

        x = self.fc2(x)

        if self.use_layernorm:
            x = self.ln2(x)

        x = F.relu(x)

        return self.output(x)