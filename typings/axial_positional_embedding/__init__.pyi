from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn

class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, axial_shape: Sequence[int]
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
