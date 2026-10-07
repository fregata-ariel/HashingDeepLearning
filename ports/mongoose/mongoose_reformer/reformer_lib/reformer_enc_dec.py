from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast
import re

import torch
from torch import nn

from .generative_tools import TrainingWrapper
from .reformer_pytorch import ReformerLM

ENC_PREFIX = "enc_"
DEC_PREFIX = "dec_"


class _ReformerFactory(Protocol):
    def __call__(self, **kwargs: object) -> ReformerLM: ...


class _GenerateWrapper(Protocol):
    def __call__(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        **kwargs: object,
    ) -> torch.Tensor: ...


class _ForwardWrapper(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor: ...


def group_dict_by_key(
    cond: Callable[[str], bool],
    data: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    matching: dict[str, object] = {}
    remaining: dict[str, object] = {}
    for key, value in data.items():
        if cond(key):
            matching[key] = value
        else:
            remaining[key] = value
    return matching, remaining


def string_begins_with(prefix: str, value: str) -> bool:
    return bool(re.match(f"^{re.escape(prefix)}", value))


def group_by_key_prefix(
    prefix: str,
    data: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    return group_dict_by_key(
        lambda key: string_begins_with(prefix, key), data
    )


def group_by_key_prefix_and_remove_prefix(
    prefix: str,
    data: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    kwargs_with_prefix, kwargs = group_by_key_prefix(prefix, data)
    kwargs_without_prefix = {
        key[len(prefix) :]: value
        for key, value in kwargs_with_prefix.items()
    }
    return kwargs_without_prefix, kwargs


def extract_enc_dec_kwargs(
    kwargs: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    enc_kwargs, remainder = group_by_key_prefix_and_remove_prefix(
        ENC_PREFIX, kwargs
    )
    dec_kwargs, remainder = group_by_key_prefix_and_remove_prefix(
        DEC_PREFIX, remainder
    )
    return enc_kwargs, dec_kwargs, remainder


def extract_and_set_enc_dec_kwargs(
    kwargs: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    enc_kwargs, dec_kwargs, remainder = extract_enc_dec_kwargs(kwargs)
    if "input_mask" in enc_kwargs:
        dec_kwargs.setdefault("context_mask", enc_kwargs["input_mask"])
    return enc_kwargs, dec_kwargs, remainder


class ReformerEncDec(nn.Module):
    """Typed encoder/decoder wrapper around two ReformerLM instances."""

    def __init__(
        self,
        dim: int,
        ignore_index: int = -100,
        pad_value: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)

        if "return_embedding" in enc_kwargs:
            raise ValueError(
                "cannot manually set encoder return embeddings"
            )
        if "dim" in dec_kwargs or "dim" in enc_kwargs:
            raise ValueError("dim is set for both encoder and decoder")

        enc_kwargs["dim"] = dim
        dec_kwargs["dim"] = dim
        enc_kwargs["return_embeddings"] = True
        dec_kwargs["causal"] = True

        enc_kwargs.setdefault("bucket_size", 64)
        enc_bucket = cast(int, enc_kwargs["bucket_size"])
        dec_kwargs.setdefault("bucket_size", enc_bucket * 2)

        factory = cast(_ReformerFactory, ReformerLM)
        enc = factory(**enc_kwargs)
        dec = factory(**dec_kwargs)

        self.enc = TrainingWrapper(
            enc, ignore_index=ignore_index, pad_value=pad_value
        )
        self.dec = TrainingWrapper(
            dec, ignore_index=ignore_index, pad_value=pad_value
        )

    def generate(
        self,
        seq_in: torch.Tensor,
        seq_out_start: torch.Tensor,
        seq_len: int,
        **kwargs: object,
    ) -> torch.Tensor:
        enc_kwargs, dec_kwargs, remainder = (
            extract_and_set_enc_dec_kwargs(kwargs)
        )
        enc_keys = self.enc(seq_in, **enc_kwargs)
        merged_dec = {**dec_kwargs, **remainder}
        generate_call = cast(
            _GenerateWrapper, self.dec.generate
        )
        return generate_call(
            seq_out_start,
            seq_len,
            keys=enc_keys,
            **merged_dec,
        )

    def forward(
        self,
        seq_in: torch.Tensor,
        seq_out: torch.Tensor,
        return_loss: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        enc_kwargs, dec_kwargs, _ = extract_and_set_enc_dec_kwargs(
            kwargs
        )
        enc_keys = self.enc(seq_in, **enc_kwargs)
        forward_call = cast(_ForwardWrapper, self.dec)
        return forward_call(
            seq_out,
            return_loss=return_loss,
            keys=enc_keys,
            **dec_kwargs,
        )
