from __future__ import annotations

import importlib
from typing import Protocol, cast

import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kernel = r'''
extern "C"
__global__ void fingerprint(const float* src, const int k, const int L, long* fp)
{
        int offset = (k * L * blockIdx.x) + (k * blockIdx.y + threadIdx.x);
        long value = (threadIdx.x >= k || src[offset] <= 0) ? 0 : 1;
        value <<= threadIdx.x;
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
                value |= __shfl_down_sync(0xFFFFFFFF, value, offset, 32);
        }
        if(!threadIdx.x)
        {
                int fp_offset = L * blockIdx.x + blockIdx.y;
                fp[fp_offset] = value;
        }
}
'''


class _FingerprintKernel(Protocol):
    def __call__(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[int],
        strm: int,
    ) -> None: ...


def _load_fingerprint_kernel() -> _FingerprintKernel | None:
    if not use_cuda:
        return None
    try:
        module = importlib.import_module(
            "mongoose_slide.slide_lib.cupy_kernel"
        )
    except (ImportError, OSError):
        return None
    kernel_type = cast(type[_FingerprintKernel], getattr(module, "cupyKernel"))
    return kernel_type(kernel, "fingerprint")


class SimHash:
    """Packed sign-random-projection hash used by the MONGOOSE scheduler.

    hash() returns shape (N, L), with one packed k-bit code per table.
    TRACE_TEST_ID: MONGOOSE-REFORMER-HASH-SHAPES.
    """

    def __init__(
        self,
        d_: int,
        k_: int,
        L_: int,
        weights: torch.Tensor | None = None,
        seed_: int = 8191,
    ) -> None:
        self.d = d_
        self.k = k_
        self.L = L_
        self.fp: _FingerprintKernel | None = _load_fingerprint_kernel()
        if weights is None:
            self.rp = SimHash.generate(d_, k_, L_, seed_)
        else:
            self.rp = SimHash.generate_from_weight(weights)

    @staticmethod
    def generate_from_weight(weights: torch.Tensor) -> torch.Tensor:
        matrix = weights
        positive = torch.gt(matrix, 0).int()
        negative = (matrix < 0.0).int()
        result = (positive - negative).float()
        return result.to(weights.device)

    @staticmethod
    def generate(d: int, k: int, L: int, seed: int) -> torch.Tensor:
        print("random generate hash table weight")
        rand_gen = np.random.RandomState(seed)
        matrix = rand_gen.randn(d, k * L)
        positive = np.greater_equal(matrix, 0.0)
        negative = np.less(matrix, 0.0)
        result = positive.astype(np.float32) - negative.astype(np.float32)
        return torch.from_numpy(result).to(device)

    def hash(
        self, data: torch.Tensor, transpose: bool = False
    ) -> torch.Tensor:
        """Hash data with CUDA when available, otherwise the reference path."""
        N, _D = data.size()
        rp = self.rp.to(data.device)
        srp = torch.matmul(data, rp)
        result = self.fingerprint(srp, N)
        if transpose:
            result = torch.t(result)
        return result

    def fingerprint(self, srp: torch.Tensor, N: int) -> torch.Tensor:
        if srp.is_cuda and self.fp is not None:
            result = torch.zeros(
                N, self.L, dtype=torch.long, device=srp.device
            )
            self.fp(
                grid=(N, self.L, 1),
                block=(32, 1, 1),
                args=[srp.data_ptr(), self.k, self.L, result.data_ptr()],
                strm=torch.cuda.current_stream().cuda_stream,
            )
            return result.int()

        if self.k > 63:
            raise ValueError("reference SimHash fingerprint supports k <= 63")
        signs = (srp.reshape(N, self.L, self.k) > 0).to(torch.int64)
        shifts = (
            1
            << torch.arange(
                self.k, device=srp.device, dtype=torch.int64
            )
        ).view(1, 1, -1)
        return torch.sum(signs * shifts, dim=-1).to(torch.int32)
