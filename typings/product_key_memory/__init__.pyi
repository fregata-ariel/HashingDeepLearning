from __future__ import annotations
import torch
from torch import nn

class PKM(nn.Module):
    def __init__(self, dim: int, num_keys: int = ...) -> None: ...
    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor: ...
