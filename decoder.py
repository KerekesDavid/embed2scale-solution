import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from encoder import FFNEncoder


class LinearDecoder(nn.Module):
    """
    Simplest linear decoder
    """

    def __init__(
        self,
        encoder: FFNEncoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Linear(encoder.output_dim, encoder.input_dim)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        input_shapes = {k: t.shape for k, t in x.items()}

        feat = self.encoder(x)

        logits = self.net(feat)
        logits = logits.split([np.prod(s[1:]) for s in input_shapes.values()], dim=1)

        logits = {
            k: logits[i].reshape(input_shapes[k])
            for i, k in enumerate(input_shapes.keys())
        }
        return logits
