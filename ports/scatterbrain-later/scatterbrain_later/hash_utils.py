"""Selected SMYRF-derived transform/rank hashing; GPL-3.0 notice in ../NOTICE.

Original operations: HazyResearch/fly 74cc6b1 hash_utils.py. Removed unrelated
hash classes/metrics, NumPy/tqdm and abstract bases; added types/CPU contracts
and explicit allocation dtype. Returned hash projections remain unfloored.
"""
from __future__ import annotations

# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/utils.py
import math
import torch


def cpu_factory(device: torch.device | str, dtype: torch.dtype) -> None:
    if torch.device(device).type != 'cpu' or dtype not in (torch.float32, torch.float64):
        raise ValueError('only CPU float32/float64 is supported')


def uniform(a: float, b: float, shape: tuple[int, ...], device: torch.device | str = 'cpu',
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    cpu_factory(device, dtype)
    if not (math.isfinite(a) and math.isfinite(b) and b > a):
        raise ValueError('uniform bounds must be finite and increasing')
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('uniform shape must contain positive integers')
    return (b - a) * torch.rand(shape, device=device, dtype=dtype) + a


class XBOXPLUS:
    """Selected actual asymmetric transform; no quantization is introduced."""
    q_norm_sq: torch.Tensor
    k_norm_sq: torch.Tensor
    MQ_sq_MK_sq: torch.Tensor

    def set_norms(self, queries: torch.Tensor, keys: torch.Tensor) -> None:
        self.q_norm_sq = queries.norm(p=2, dim=-1, keepdim=True).square()
        self.k_norm_sq = keys.norm(p=2, dim=-1, keepdim=True).square()
        MQ_sq = torch.amax(self.q_norm_sq, dim=-2, keepdim=True)
        MK_sq = torch.amax(self.k_norm_sq, dim=-2, keepdim=True)
        self.MQ_sq_MK_sq = MQ_sq + MK_sq

    def K(self, x: torch.Tensor) -> torch.Tensor:
        ext = (self.MQ_sq_MK_sq - self.k_norm_sq).sqrt()
        return torch.cat([x, ext, torch.zeros_like(ext)], dim=-1)

    def Q(self, x: torch.Tensor) -> torch.Tensor:
        ext = (self.MQ_sq_MK_sq - self.q_norm_sq).sqrt()
        return torch.cat([x, torch.zeros_like(ext), ext], dim=-1)


class E2LSH:
    def __init__(self, n_hashes: int, dim: int, r: float, device: torch.device | str = 'cpu',
                 dtype: torch.dtype = torch.float32) -> None:
        """Draw alpha/beta in the caller's supported CPU computation dtype.

        TRACE_TEST_ID: SCATTERBRAIN-SBSMYRF-DTYPE-CPU.
        """
        cpu_factory(device, dtype)
        if type(n_hashes) is not int or n_hashes <= 0 or type(dim) is not int or dim <= 0:
            raise ValueError('hash counts/dimensions must be positive integers')
        if not math.isfinite(r) or r <= 0:
            raise ValueError('hash radius must be positive and finite')
        self.alpha = torch.normal(0, 1, (dim, n_hashes), device=device, dtype=dtype)
        self.beta = uniform(0, r, shape=(1, n_hashes), device=device, dtype=dtype)
        self.dim = dim
        self.r = r

    def __call__(self, vecs: torch.Tensor) -> torch.Tensor:
        if (vecs.ndim != 3 or vecs.shape[-1] != self.dim or vecs.dtype != self.alpha.dtype
                or vecs.device != self.alpha.device or not bool(torch.isfinite(vecs).all())):
            raise ValueError('hash input must match finite CPU allocation dtype/device/dimension')
        projection = vecs @ self.alpha
        projection_shift = projection + self.beta
        return projection_shift.permute(2, 0, 1)


def lsh_clustering(queries: torch.Tensor, keys: torch.Tensor, n_hashes: int, r: float = 1,
                   key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_factory(queries.device, queries.dtype)
    if (queries.ndim != 3 or keys.ndim != 3 or queries.shape[0] != keys.shape[0]
            or queries.shape[-1] != keys.shape[-1] or queries.dtype != keys.dtype
            or queries.device != keys.device or min(*queries.shape, *keys.shape) <= 0
            or not bool(torch.isfinite(queries).all() and torch.isfinite(keys).all())):
        raise ValueError('incompatible finite query/key hash inputs')
    if key_padding_mask is not None:
        if (key_padding_mask.dtype != torch.bool or key_padding_mask.device != keys.device
                or key_padding_mask.shape != (1, keys.shape[0], keys.shape[1])):
            raise ValueError('hash mask must be CPU bool [1,batch_heads,keys]')
    e2lsh = E2LSH(n_hashes=n_hashes, dim=queries.shape[-1], r=r, device=queries.device, dtype=queries.dtype)
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    if key_padding_mask is not None:
        keys_hashed.masked_fill_(~key_padding_mask, float('inf'))
        # Historical self-attention heuristic, retained for FP32 parity.
        if queries.shape[-2] == key_padding_mask.shape[-1]:
            queries_hashed.masked_fill_(~key_padding_mask, float('inf'))
    return queries_hashed.argsort(dim=-1), keys_hashed.argsort(dim=-1)
