"""Actual synchronous SYCL CPU local products for the bounded SBLocal port.

No Torch arithmetic fallback: every numeric operation calls the shared library.
FP32 CPU tensors, contiguous copies at the ABI; first-order autograd only.
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import torch
import numpy as np
import numpy.typing as npt
from typing import cast
from torch.autograd.function import FunctionCtx, once_differentiable

Array = npt.NDArray[np.float32 | np.int32 | np.int64]
class _Context(FunctionCtx):
    dims: tuple[int, ...]
    saved_tensors: tuple[torch.Tensor, ...]

_library: ctypes.CDLL | None = None


def load_library(path: Path) -> None:
    global _library
    _library = ctypes.CDLL(str(path.resolve()))
    _library.sb_run.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 8
    _library.sb_run.restype = ctypes.c_int
    _library.sb_last_error.restype = ctypes.c_char_p
    _library.sb_describe.restype = ctypes.c_int
    if _library.sb_describe():
        raise RuntimeError(_library.sb_last_error().decode())


def require_library() -> None:
    if _library is None:
        raise RuntimeError("load an actual SYCL library before using the adapter")


def _tensor(value: torch.Tensor, rank: int, *, lengths: bool = False) -> Array:
    dtype = torch.int64 if lengths else torch.float32
    if value.device.type != "cpu" or value.dtype != dtype or value.ndim != rank:
        raise ValueError(f"expected rank{rank} CPU {dtype}")
    if not lengths and not bool(torch.isfinite(value).all()):
        raise ValueError("finite tensors required")
    return value.detach().contiguous().numpy()


def _run(operation: int, dims: tuple[int, ...], a: Array, b: Array,
         grad: Array | None, mask: Array | None, lengths: Array | None,
         first: tuple[int, ...], second: tuple[int, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _library is None:
        raise RuntimeError("load an actual SYCL library before using the adapter")
    shape = np.asarray(dims, dtype=np.int32)
    out1 = np.empty(first, dtype=np.float32)
    out2 = None if second is None else np.empty(second, dtype=np.float32)
    arrays = [shape, a, b, grad, mask, lengths, out1, out2]
    pointers = [None if x is None else ctypes.c_void_p(x.ctypes.data) for x in arrays]
    if _library.sb_run(operation, *pointers):
        raise ValueError(_library.sb_last_error().decode())
    return torch.from_numpy(out1), None if out2 is None else torch.from_numpy(out2)


class _Dot(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Context, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor, window: int) -> torch.Tensor:
        qa, ka, ma, la = _tensor(q, 4), _tensor(k, 4), _tensor(mask, 2), _tensor(lengths, 1, lengths=True)
        b, h, nq, e = q.shape
        if k.shape[:2] != (b, h) or k.shape[3] != e or mask.shape != (nq, k.shape[2]) or lengths.shape != (b,):
            raise ValueError("dot shapes")
        if mask.requires_grad:
            raise ValueError("additive mask is constant in this contract")
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("positive integer window required")
        dims = (b, h, nq, k.shape[2], e, 1, window)
        ctx.save_for_backward(q, k, lengths)
        ctx.dims = dims
        return _run(0, dims, qa, ka, None, ma, la, (b, h, nq, window))[0]

    @staticmethod
    @once_differentiable
    def backward(ctx: _Context, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None, None]:
        q, k, lengths = ctx.saved_tensors
        dq, dk = _run(1, ctx.dims, _tensor(q, 4), _tensor(k, 4), _tensor(grad, 4), None,
                      _tensor(lengths, 1, lengths=True), tuple(q.shape), tuple(k.shape))
        return dq, dk, None, None, None


class _Average(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _Context, attention: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        aa, va = _tensor(attention, 4), _tensor(values, 4)
        b, h, nq, window = attention.shape
        if values.shape[:2] != (b, h):
            raise ValueError("weighted-average shapes")
        dims = (b, h, nq, values.shape[2], 1, values.shape[3], window)
        ctx.save_for_backward(attention, values)
        ctx.dims = dims
        return _run(2, dims, aa, va, None, None, None, (b, h, nq, values.shape[3]))[0]

    @staticmethod
    @once_differentiable
    def backward(ctx: _Context, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        a, v = ctx.saved_tensors
        return _run(3, ctx.dims, _tensor(a, 4), _tensor(v, 4), _tensor(grad, 4), None, None,
                    tuple(a.shape), tuple(v.shape))


def local_dot_product(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor,
                      lengths: torch.Tensor, window: int) -> torch.Tensor:
    return cast(torch.Tensor, _Dot.apply(q, k, mask, lengths, window))


def local_weighted_average(attention: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, _Average.apply(attention, values))
