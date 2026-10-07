from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeAlias, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .autopadder import Autopadder
from .reformer_pytorch import ReformerLM, ReformerLM_tune

FilterLogitsFn: TypeAlias = Callable[[torch.Tensor, float], torch.Tensor]
TokenInput: TypeAlias = torch.Tensor | Sequence[torch.Tensor]


def top_p(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = (
        sorted_indices_to_remove[:, :-1].clone()
    )
    sorted_indices_to_remove[:, 0] = False
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    k = max(1, int((1 - thres) * logits.shape[-1]))
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class TrainingWrapper(nn.Module):
    """Language-model loss/generation wrapper around a Reformer LM."""

    def __init__(
        self,
        net: ReformerLM | ReformerLM_tune,
        ignore_index: int = -100,
        pad_value: int = 0,
    ) -> None:
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        eos_token: int | None = None,
        temperature: float = 1.0,
        filter_logits_fn: FilterLogitsFn = top_k,
        filter_thres: float = 0.9,
        **kwargs: object,
    ) -> torch.Tensor:
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        _batch, t = start_tokens.shape
        self.net.eval()
        out = start_tokens
        input_mask = cast(
            torch.Tensor | None, kwargs.pop("input_mask", None)
        )
        if input_mask is None:
            input_mask = torch.full_like(
                out, True, dtype=torch.bool, device=out.device
            )

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            input_mask = input_mask[:, -self.max_seq_len :]
            logits = self.net(
                x, input_mask=input_mask, **kwargs
            )[:, -1, :]
            filtered_logits = filter_logits_fn(
                logits, filter_thres
            )
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)
            if eos_token is not None and bool((sample == eos_token).all()):
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        self.net.train(was_training)
        return out

    def forward(
        self,
        x: TokenInput,
        loss_weight: torch.Tensor | None = None,
        return_loss: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        pad = partial(
            pad_sequence,
            batch_first=True,
            padding_value=self.pad_value,
        )

        if not return_loss:
            tokens = x if isinstance(x, torch.Tensor) else pad(list(x))
            return cast(torch.Tensor, self.net(tokens, **kwargs))

        if isinstance(x, torch.Tensor):
            xi = x[:, :-1]
            xo = x[:, 1:]
        else:
            xi = pad([tensor[:-1] for tensor in x])
            xo = pad([tensor[1:] for tensor in x])

        out = self.net(xi, **kwargs)
        if loss_weight is not None:
            num_token = torch.sum(loss_weight)
            masked_loss = loss_weight[:, 1:] * F.cross_entropy(
                out.transpose(1, 2),
                xo,
                reduction="none",
                ignore_index=self.ignore_index,
            )
            return torch.sum(masked_loss) / num_token
        return F.cross_entropy(
            out.transpose(1, 2),
            xo,
            ignore_index=self.ignore_index,
        )
