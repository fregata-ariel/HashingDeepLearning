from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TypeAlias, cast

import torch
from torch import nn

from .reformer_pytorch import LSHAttention, LSHSelfAttention

Recording: TypeAlias = dict[str, torch.Tensor]


class Recorder(nn.Module):
    """Record LSH attention matrices and bucket ids from a wrapped model."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.iter = 0
        self.recordings: defaultdict[int, list[Recording]] = defaultdict(list)
        self.net = net
        self.on = True
        self.ejected = False

    def eject(self) -> nn.Module:
        self.ejected = True
        self.clear()
        self.unwire()
        return self.net

    def wire(self) -> None:
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = True
            if isinstance(module, LSHSelfAttention):
                module.callback = self.record

    def unwire(self) -> None:
        for module in self.net.modules():
            if isinstance(module, LSHAttention):
                module._return_attn = False
            if isinstance(module, LSHSelfAttention):
                module.callback = None

    def turn_on(self) -> None:
        self.on = True

    def turn_off(self) -> None:
        self.on = False

    def clear(self) -> None:
        self.recordings = defaultdict(list)
        self.iter = 0

    def record(
        self, attn: torch.Tensor, buckets: torch.Tensor
    ) -> None:
        if not self.on:
            return
        data: Recording = {
            "attn": attn.detach().cpu(),
            "buckets": buckets.detach().cpu(),
        }
        self.recordings[self.iter].append(data)

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        if self.ejected:
            raise RuntimeError("Recorder has already been ejected and disposed")
        if self.on:
            self.wire()
        out = cast(torch.Tensor, self.net(x, **kwargs))
        self.iter += 1
        self.unwire()
        return out
