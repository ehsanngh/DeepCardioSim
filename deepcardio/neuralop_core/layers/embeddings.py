# Source: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/embeddings.py

import torch
from torch import nn

class SinusoidalEmbedding2D(nn.Module):
    def __init__(self, num_channels, max_positions=None, endpoint=False):
        """SinusoidalEmbedding2D applies a 2d sinusoidal positional encoding 

        Parameters
        ----------
        num_channels : int
            number of input channels
        max_positions : int, optional
            maximum positions to encode, by default 10000
        endpoint : bool, optional
            whether to set endpoint, by default False
        """
        if max_positions is None:
            max_positions = 10000
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x