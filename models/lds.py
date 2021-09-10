from typing import List, Tuple, Callable, Union, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.mlp import MLP


class LDS(nn.Module):
    def __init__(self, K=2, future_lens=16,
                 hidden_size=64, output_sizes=[64,32]):
        super(LDS, self).__init__()
        self.nx = 2
        self.ny = future_lens * self.nx
        self.K = K

        self.mlp = MLP(hidden_size, output_sizes)
        self.A = nn.Linear(output_sizes[-1], self.ny * self.K)
        self.b = nn.Linear(output_sizes[-1], self.ny * self.K)

        self.y_hat = None

    def forward(self, history_features):
        h = history_features
        z = torch.randn((h.shape[0], self.ny), device=h.device)
        z = z.repeat_interleave(self.K, dim=0)
        h = self.mlp(h)

        # (B * nk, ny)
        a = self.A(h).view(-1, self.ny)
        b = self.b(h).view(-1, self.ny)
        x = a * z + b
        return x, a, b

    def clone(self):
        clone = LDS(K=self.K)
        clone.load_state_dict(self.state_dict())
        return clone
