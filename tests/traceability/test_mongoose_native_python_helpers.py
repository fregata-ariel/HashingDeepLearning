"""Runtime contracts for native-LSH Python helper modules."""
from __future__ import annotations

import importlib
import sys
import types

import torch

from lsh_lib.matrix_simhash import SimHash

# TRACE_TEST_ID: MONGOOSE-NATIVE-PYTHON-HELPERS


def test_matrix_simhash_composition_without_cuda_launch() -> None:
    left = SimHash.__new__(SimHash)
    right = SimHash.__new__(SimHash)
    left.rp = torch.tensor([[1.0], [-1.0]])
    right.rp = torch.tensor([[0.5], [0.25]])

    combined = SimHash.generate_from_list([left, right])
    assert combined.shape == (2, 2)
    assert torch.equal(combined[:, 0], left.rp[:, 0])
    assert torch.equal(combined[:, 1], right.rp[:, 0])

    if not torch.cuda.is_available():
        try:
            SimHash.generate(2, 1, 1, 7)
        except RuntimeError as error:
            assert "requires CUDA" in str(error)
        else:
            raise AssertionError("matrix_simhash unexpectedly generated on CPU")


def test_query_mul_autograd_contract_with_fake_native_module() -> None:
    fake = types.ModuleType("query_mul")

    def constrained_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
    ) -> torch.Tensor:
        del row_ptr, col_idx
        return A @ B

    def csrmm(
        values: torch.Tensor,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        dense: torch.Tensor,
        output_dim: int,
        transpose_sparse: bool,
        transpose_dense: bool,
    ) -> torch.Tensor:
        del row_ptr, col_idx, output_dim
        if not transpose_sparse and transpose_dense:
            return values @ dense.t()
        if transpose_sparse and not transpose_dense:
            return values.t() @ dense
        raise AssertionError("unexpected csrmm transpose contract")

    fake.constrained_gemm = constrained_gemm
    fake.csrmm = csrmm
    sys.modules["query_mul"] = fake
    sys.modules.pop("lsh_lib.query_mul_interface", None)

    module = importlib.import_module("lsh_lib.query_mul_interface")
    query_mul_fn = module.query_mul_fn

    A = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    B = torch.tensor(
        [[0.5, -1.0], [2.0, 1.5]], requires_grad=True
    )
    row_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
    col_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    actual = query_mul_fn(A, B, row_ptr, col_idx)
    expected = A @ B
    assert torch.allclose(actual, expected)

    actual.sum().backward()
    assert A.grad is not None and B.grad is not None

    ref_A = A.detach().clone().requires_grad_(True)
    ref_B = B.detach().clone().requires_grad_(True)
    (ref_A @ ref_B).sum().backward()
    assert ref_A.grad is not None and ref_B.grad is not None
    assert torch.allclose(A.grad, ref_A.grad)
    assert torch.allclose(B.grad, ref_B.grad)


if __name__ == "__main__":
    test_matrix_simhash_composition_without_cuda_launch()
    test_query_mul_autograd_contract_with_fake_native_module()
    print("TRACE_TEST_PASS MONGOOSE-NATIVE-PYTHON-HELPERS")
