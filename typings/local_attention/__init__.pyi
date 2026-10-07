from __future__ import annotations
import torch
from torch import nn

class LocalAttention(nn.Module):
    def __init__(
        self,
        *,
        window_size: int,
        causal: bool,
        dropout: float,
        shared_qk: bool,
        look_forward: int,
    ) -> None: ...
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        input_mask: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...
