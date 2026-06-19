"""Small Vision Transformer used in Appendix A.2.

The paper uses a ViT-style model trained from scratch with convolutional patch
embeddings of patch size 3, model dimension 256, feedforward width 1024, one
transformer block, and dropout 0.1.  The convolutional embedding does not
require the image side length to be divisible by the patch size; it simply
creates all valid non-overlapping 3x3 patches for the input resolution.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=28,
        input_shape: Optional[Sequence[int]] = None,
        patch_size=3,
        num_classes=10,
        num_actions=None,
        output_dim=None,
        dim=256,
        depth=1,
        heads=4,
        mlp_dim=1024,
        channels=1,
        dropout=0.1,
    ):
        super().__init__()

        if input_shape is not None:
            channels = int(input_shape[0])
            image_size = int(input_shape[1])

        if num_actions is not None:
            num_classes = int(num_actions)
        if output_dim is not None:
            num_classes = int(output_dim)

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.feature_dim = int(dim)

        self.patch_embedding = nn.Conv2d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        with torch.no_grad():
            dummy_shape = input_shape or (channels, image_size, image_size)
            dummy = torch.zeros(1, *dummy_shape)
            embedded = self.patch_embedding(dummy)
            num_patches = embedded.shape[-2] * embedded.shape[-1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    @property
    def last_layer(self) -> nn.Linear:
        return self.mlp_head[-1]

    def reset_last_layer(self) -> None:
        self.mlp_head[-1].reset_parameters()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)

        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_output = self.forward_features(x)
        return self.mlp_head(cls_output)
