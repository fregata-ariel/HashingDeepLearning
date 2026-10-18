"""Selected deterministic tiny CPU helpers derived from models/utils.py.

TRACE_TEST_ID: MAGICPIG-SERVER-CPU. No model weights, RNG or FlashInfer kernels.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def causal_mask(length: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Finite-min additive mask: row/query sees keys up to its own position."""
    if length <= 0 or dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("positive length and floating dtype required")
    mask = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype)
    positions = torch.arange(length)
    return mask.masked_fill(positions[None, :] <= positions[:, None], 0)


def repeat_kv(value: Tensor, repeats: int) -> Tensor:
    """Map each KV head to consecutive query groups; repeats1 preserves alias."""
    if value.ndim != 4 or repeats < 1:
        raise ValueError("rank4 and positive repeats required")
    batch, heads, length, dim = value.shape
    if repeats == 1:
        return value
    return value[:, :, None, :, :].expand(batch, heads, repeats, length, dim).reshape(batch, heads*repeats, length, dim)


def rotate_half(value: Tensor) -> Tensor:
    """Use split-half RoPE convention [-second_half, first_half]."""
    if value.ndim == 0 or value.shape[-1] <= 0 or value.shape[-1] % 2:
        raise ValueError("positive even last dimension required")
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def rotary(value: Tensor, cosine: Tensor, sine: Tensor, positions: Tensor) -> Tensor:
    """Index absolute positions then broadcast over heads of [batch,head,seq,dim]."""
    if value.ndim != 4 or positions.dtype != torch.int64 or tuple(positions.shape) != (value.shape[0], value.shape[2]) or cosine.shape != sine.shape or cosine.ndim != 2 or cosine.shape[1] != value.shape[-1]:
        raise ValueError("invalid RoPE shapes/positions")
    if bool((positions < 0).any()) or bool((positions >= cosine.shape[0]).any()):
        raise ValueError("RoPE position out of range")
    return value*cosine[positions].unsqueeze(1) + rotate_half(value)*sine[positions].unsqueeze(1)


def rms_norm(value: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """CPU FP32 RMS scalar definition, returned in the input floating dtype."""
    if value.ndim != 3 or tuple(weight.shape) != (value.shape[-1],) or not math.isfinite(epsilon) or epsilon <= 0 or value.dtype not in (torch.float32, torch.bfloat16) or weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("invalid RMSNorm shape/epsilon")
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(value.dtype)
