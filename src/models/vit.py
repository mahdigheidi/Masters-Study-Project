import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=1,
        heads=4,
        mlp_dim=1024,
        channels=1,
        dropout=0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_embedding = nn.Conv2d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        x = x.flatten(2).transpose(1, 2)

        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, : x.size(1)]

        x = self.transformer(x)

        cls_output = x[:, 0]

        return self.mlp_head(cls_output)
