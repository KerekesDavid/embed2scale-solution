from pathlib import Path
from logging import Logger

import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNEncoder(nn.Module):
    def __init__(
        self,
        input_size: list[list[int]],
        encoder_weights: str | Path,
        output_dim: int = 1024,
        embed_dim: int = 2048,
        activation_function=nn.GELU(),
        dropout: nn.Dropout = nn.Dropout(p=0.1),
        input_dropout: nn.Dropout = None,
        input_norm: bool = True,
        temporal_dropout: bool = None,
    ) -> None:
        super().__init__()

        self.input_dim = np.sum([np.prod(il) for il in input_size])
        self.output_dim = output_dim
        self.encoder_weights = encoder_weights

        self.activation_function = (
            activation_function if activation_function else nn.Identity()
        )
        self.dropout = dropout if dropout else nn.Identity()
        self.input_dropout = input_dropout if input_dropout else nn.Identity()
        self.temporal_dropout = temporal_dropout if temporal_dropout else nn.Identity()
        self.input_norm = nn.LayerNorm(self.input_dim) if input_norm else nn.Identity()

        self.net = nn.Sequential(
            self.input_norm,
            self.input_dropout,
            nn.Linear(self.input_dim, embed_dim),
            self.activation_function,
            self.dropout,
            nn.Linear(embed_dim, self.output_dim),
        )

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        if type(self.temporal_dropout) is not nn.Identity:
            x = torch.cat(list(x.values()), dim=1)  # b c t
            x = rearrange(x, "b c t -> b t c")
            x = self.temporal_dropout(x)
            x = rearrange(x, "b t c -> b (t c)")
        else:
            x = torch.cat([rearrange(t, "b c t -> b (c t)") for t in x.values()], dim=1)

        return self.net(x)
