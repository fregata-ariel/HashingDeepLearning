from __future__ import annotations

import torch


def constrained_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
) -> torch.Tensor: ...


def csrmm(
    values: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    dense: torch.Tensor,
    output_dim: int,
    transpose_sparse: bool,
    transpose_dense: bool,
) -> torch.Tensor: ...
