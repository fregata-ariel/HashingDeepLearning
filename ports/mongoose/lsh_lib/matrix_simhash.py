from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from mongoose_slide.slide_lib.cupy_kernel import cupyKernel

kernel = r'''
extern "C"
__global__ void fingerprint(const float* src, const int k, const int L, long* fp)
{
        int offset = (k * L * blockIdx.x) + (k * blockIdx.y + threadIdx.x);
        long value = (threadIdx.x >= k || src[offset] <= 0) ? 0 : 1;
        value <<= threadIdx.x;

        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
                value |= __shfl_down_sync(0xFFFFFFFF, value, offset);
        }

        if(!threadIdx.x)
        {
                int fp_offset = L * blockIdx.x + blockIdx.y;
                fp[fp_offset] = value;
        }
}
'''


class SimHash:
    """GPU-only matrix SimHash helper used by legacy native-LSH experiments.

    Input is a floating tensor shaped (N, d). hash() returns an int64 tensor
    shaped (N, L), with one packed k-bit code per table. Unlike the maintained
    CPU-capable MONGOOSE-SLIDE SimHash, this helper requires CUDA when hashes
    are actually generated.
    """

    def __init__(
        self,
        d_: int,
        k_: int,
        L_: int,
        seed_: int = 8191,
        srp_list: Sequence[SimHash] | None = None,
    ) -> None:
        self.d = d_
        self.k = k_
        self.L = L_
        self.fp = cupyKernel(kernel, "fingerprint")
        if srp_list is None:
            self.rp = SimHash.generate(d_, k_, L_, seed_)
        else:
            self.rp = SimHash.generate_from_list(srp_list)

    @staticmethod
    def generate_from_list(
        srp_list: Sequence[SimHash],
    ) -> torch.Tensor:
        if not srp_list:
            raise ValueError("srp_list must contain at least one SimHash")
        matrices = [item.rp for item in srp_list]
        return torch.cat(matrices, dim=1)

    @staticmethod
    def generate(
        d: int, k: int, L: int, seed: int
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("matrix_simhash requires CUDA")
        rand_gen = np.random.RandomState(seed)
        matrix = rand_gen.randn(d, k * L)
        positive = np.greater_equal(matrix, 0.0)
        negative = np.less(matrix, 0.0)
        result = (
            positive.astype(np.float32)
            - negative.astype(np.float32)
        )
        return torch.from_numpy(result).to("cuda")

    def hash(
        self, data: torch.Tensor, transpose: bool = False
    ) -> torch.Tensor:
        """Hash a CUDA float matrix into packed per-table integer codes."""
        batch, _dim = data.size()
        srp = torch.matmul(data, self.rp.to(data.device))
        result = self.fingerprint(srp, batch)
        if transpose:
            result = torch.t(result)
        return result

    def fingerprint(
        self, srp: torch.Tensor, batch: int
    ) -> torch.Tensor:
        if not srp.is_cuda:
            raise ValueError("matrix_simhash fingerprint expects CUDA input")
        result = torch.zeros(
            batch,
            self.L,
            dtype=torch.long,
            device=srp.device,
        )
        self.fp(
            grid=(batch, self.L, 1),
            block=(32, 1, 1),
            args=[srp.data_ptr(), self.k, self.L, result.data_ptr()],
            strm=torch.cuda.current_stream().cuda_stream,
        )
        return result
