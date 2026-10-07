from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch.autograd.function import Function, FunctionCtx

import query_mul


class QueryMulFn(Function):
    """Autograd bridge for a CSR-constrained dense matrix product.

    A and B are floating dense matrices. rowPtrC and colIdxC are integer CSR
    index tensors describing which output entries are materialized. forward()
    returns the native constrained-GEMM values tensor. backward() returns dense
    gradients for A/B when requested and None for the CSR structure tensors.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        A: torch.Tensor,
        B: torch.Tensor,
        rowPtrC: torch.Tensor,
        colIdxC: torch.Tensor,
    ) -> torch.Tensor:
        A_cont = A.detach().t().contiguous().t()
        B_cont = B.detach().t().contiguous().t()
        ctx.save_for_backward(A_cont, B_cont, rowPtrC, colIdxC)
        return query_mul.constrained_gemm(
            A_cont, B_cont, rowPtrC, colIdxC
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        saved = cast(
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            getattr(ctx, "saved_tensors"),
        )
        A_cont, B_cont, rowPtrC, colIdxC = saved
        needs = cast(
            tuple[bool, ...], getattr(ctx, "needs_input_grad")
        )

        grad_A: torch.Tensor | None = None
        grad_B: torch.Tensor | None = None

        if needs[0]:
            grad_A = query_mul.csrmm(
                grad,
                rowPtrC,
                colIdxC,
                B_cont,
                A_cont.shape[0],
                False,
                True,
            )
        if needs[1]:
            grad_B = query_mul.csrmm(
                grad,
                rowPtrC,
                colIdxC,
                A_cont,
                B_cont.shape[1],
                True,
                False,
            ).t()
        return grad_A, grad_B, None, None


QueryMulCallable = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
query_mul_fn = cast(QueryMulCallable, QueryMulFn.apply)
