"""Cython buffer-contract oracle for the MONGOOSE native LSH wrapper."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from clsh import pyLSH

# TRACE_TEST_ID: MONGOOSE-CYTHON-LSH-BOUNDARY


def _must_reject(callable_obj: Callable[[], object]) -> None:
    try:
        callable_obj()
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid NumPy buffer was unexpectedly accepted")


def test_cython_lsh_dtype_shape_and_contiguity_contract() -> None:
    lsh = pyLSH(1, 2, 1)

    fp0 = np.array([1, 10], dtype=np.int32)
    fp1 = np.array([1, 20], dtype=np.int32)
    lsh.insert(fp0, 0)
    lsh.insert(fp1, 1)
    assert set(lsh.query(fp0)) == {0, 1}

    batch = np.array([[1, 10], [2, 20]], dtype=np.int32, order="C")
    lsh.clear()
    lsh.insert_multi(batch, 2)
    assert set(lsh.query(batch[0])) == {0}
    assert set(lsh.query(batch[1])) == {1}

    mask = np.zeros((2, 2), dtype=np.float32, order="C")
    lsh.query_multi_mask(batch, mask, 2, 2)
    assert np.array_equal(mask, np.eye(2, dtype=np.float32))

    # np.ndarray[int] in clsh.pyx is C int (int32 on the runner).
    _must_reject(
        lambda: lsh.query(np.array([1, 10], dtype=np.int64))
    )

    # mode="c" rejects strided/non-C-contiguous fingerprints.
    non_contiguous = np.arange(8, dtype=np.int32).reshape(2, 4)[:, ::2]
    assert not non_contiguous.flags.c_contiguous
    _must_reject(lambda: lsh.insert_multi(non_contiguous, 2))

    # query_multi_mask requires float32 C-contiguous storage.
    wrong_mask = np.zeros((2, 2), dtype=np.float64)
    _must_reject(lambda: lsh.query_multi_mask(batch, wrong_mask, 2, 2))

    labels = np.array([[0, -1], [1, -1]], dtype=np.int64, order="C")
    samples = {0, 1, 2}
    sample_list, probs = lsh.multi_label(labels, samples)
    assert set(sample_list[:2]) == {0, 1}
    assert 2 in sample_list
    assert probs.dtype == np.float32
    assert probs.shape[0] == 2
    assert probs[0, sample_list.index(0)] == 1.0
    assert probs[1, sample_list.index(1)] == 1.0


if __name__ == "__main__":
    test_cython_lsh_dtype_shape_and_contiguity_contract()
    print("TRACE_TEST_PASS MONGOOSE-CYTHON-LSH-BOUNDARY")
