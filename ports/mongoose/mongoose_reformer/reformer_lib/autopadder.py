from __future__ import annotations

from typing import TypeAlias, cast

import torch
import torch.nn.functional as F
from torch import nn

from .reformer_pytorch import (
    LSHSelfAttention,
    Reformer,
    ReformerLM,
    ReformerLM_tune,
    Reformer_tune,
)

PadderNet: TypeAlias = (
    Reformer | ReformerLM | LSHSelfAttention | ReformerLM_tune | Reformer_tune
)


def pad_to_multiple(
    tensor: torch.Tensor,
    seqlen: int,
    multiple: int,
    dim: int = -1,
) -> torch.Tensor:
    ratio = seqlen / multiple
    if ratio.is_integer():
        return tensor
    remainder = math.ceil(ratio) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)


import math


class Autopadder(nn.Module):
    """Pad Reformer inputs/masks to the active hash bucket multiple."""

    def __init__(self, net: PadderNet) -> None:
        super().__init__()
        self.net = net

        if isinstance(net, (ReformerLM, ReformerLM_tune)):
            reformer: Reformer | Reformer_tune | LSHSelfAttention = net.reformer
            self.pad_dim = -1
        else:
            reformer = net
            self.pad_dim = -2

        if isinstance(reformer, Reformer_tune):
            self.bucket_size = reformer.bucket_size_list[0]
        else:
            self.bucket_size = reformer.bucket_size

        self.num_mem_kv = reformer.num_mem_kv
        self.full_attn_thres = reformer.full_attn_thres

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        _batch, t = x.shape[:2]
        m = self.num_mem_kv

        keys = cast(torch.Tensor | None, kwargs.get("keys"))
        input_mask = cast(
            torch.Tensor | None, kwargs.get("input_mask")
        )
        input_attn_mask = cast(
            torch.Tensor | None, kwargs.get("input_attn_mask")
        )

        k_len = 0 if keys is None else keys.shape[1]
        seqlen = t + m + k_len

        if seqlen > self.full_attn_thres:
            if input_mask is None:
                input_mask = torch.ones(
                    x.shape[:2],
                    device=x.device,
                    dtype=torch.bool,
                )

            x = pad_to_multiple(
                x,
                seqlen,
                self.bucket_size * 2,
                dim=self.pad_dim,
            )

            new_mask = F.pad(
                input_mask,
                (0, x.shape[1] - input_mask.shape[1]),
                value=False,
            )
            kwargs["input_mask"] = new_mask

            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_attn_mask = F.pad(
                    input_attn_mask,
                    (0, offset, 0, offset),
                    value=False,
                )
                kwargs["input_attn_mask"] = new_attn_mask

        out = cast(torch.Tensor, self.net(x, **kwargs))
        return out[:, 0:t]
