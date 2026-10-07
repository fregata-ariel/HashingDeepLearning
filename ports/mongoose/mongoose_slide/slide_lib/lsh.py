from __future__ import annotations

from collections.abc import MutableSet
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from lsh_lib.clsh import pyLSH
else:
    from clsh import pyLSH

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

Int32Array: TypeAlias = npt.NDArray[np.int32]
Int64Array: TypeAlias = npt.NDArray[np.int64]
Float32Array: TypeAlias = npt.NDArray[np.float32]


class HashFunction(Protocol):
    """Hash callable consumed by the Cython-backed LSH wrapper."""

    def hash(
        self, data: torch.Tensor, transpose: bool = False
    ) -> torch.Tensor: ...


def _fingerprint_array(values: torch.Tensor) -> Int32Array:
    """Return a C-contiguous int32 fingerprint buffer for clsh.pyx."""
    array = values.detach().to("cpu", dtype=torch.int32).numpy()
    return cast(
        Int32Array,
        np.ascontiguousarray(array, dtype=np.int32),
    )


def _int64_array(values: torch.Tensor | Int64Array) -> Int64Array:
    if isinstance(values, torch.Tensor):
        raw = values.detach().to("cpu", dtype=torch.int64).numpy()
    else:
        raw = values
    return cast(
        Int64Array,
        np.ascontiguousarray(raw, dtype=np.int64),
    )


class LSH:
    """Typed Python owner for the native MONGOOSE LSH table.

    Fingerprints passed into pyLSH are always C-contiguous np.int32;
    labels are np.int64 and query masks are np.float32. These are the
    buffer contracts verified by MONGOOSE-CYTHON-LSH-BOUNDARY.
    """

    def __init__(
        self,
        func_: HashFunction,
        K_: int,
        L_: int,
        threads_: int = 8,
    ) -> None:
        self.func = func_
        self.K = K_
        self.L = L_
        self.lsh_ = pyLSH(self.K, self.L, threads_)
        self.sample_size = 0
        self.count = 0

    def setSimHash(self, func_: HashFunction) -> None:
        self.func = func_

    def resetLSH(self, func_: HashFunction) -> None:
        self.func = func_
        self.clear()

    def stats(self) -> int:
        avg_size = self.sample_size // max(self.count, 1)
        print("hashtable avg_size", avg_size)
        self.sample_size = 0
        self.count = 0
        return avg_size

    def remove_insert(
        self,
        item_id: int,
        old_item: torch.Tensor,
        new_fp: Int32Array,
    ) -> None:
        old_fp = _fingerprint_array(self.func.hash(old_item))
        self.lsh_.remove(
            cast(Int32Array, np.ascontiguousarray(np.squeeze(old_fp))),
            item_id,
        )
        self.lsh_.insert(
            cast(Int32Array, np.ascontiguousarray(new_fp, dtype=np.int32)),
            item_id,
        )

    def insert(self, item_id: int, item: torch.Tensor) -> None:
        fp = _fingerprint_array(self.func.hash(item))
        self.lsh_.insert(
            cast(Int32Array, np.ascontiguousarray(np.squeeze(fp))),
            item_id,
        )

    def insert_fp(self, item_id: int, fp: Int32Array) -> None:
        self.lsh_.insert(
            cast(Int32Array, np.ascontiguousarray(fp, dtype=np.int32)),
            item_id,
        )

    def insert_multi(self, items: torch.Tensor, N: int) -> None:
        fp = _fingerprint_array(self.func.hash(items))
        self.lsh_.insert_multi(fp, N)

    def query(self, item: torch.Tensor) -> set[int]:
        fp = _fingerprint_array(self.func.hash(item))
        return self.lsh_.query(
            cast(Int32Array, np.ascontiguousarray(np.squeeze(fp)))
        )

    def query_fp(self, fp: Int32Array) -> set[int]:
        contiguous = cast(
            Int32Array,
            np.ascontiguousarray(fp, dtype=np.int32),
        )
        return self.lsh_.query(contiguous)

    def query_multi(
        self, items: torch.Tensor, N: int
    ) -> tuple[set[int], Int32Array]:
        fp = _fingerprint_array(
            self.func.hash(items, transpose=False)
        )
        return self.lsh_.query_multi(fp, N), fp

    def query_multi_mask(
        self,
        item: torch.Tensor,
        M: int,
        N: int,
    ) -> tuple[torch.Tensor, Int32Array]:
        fp = _fingerprint_array(self.func.hash(item))
        mask = torch.zeros(M, N, dtype=torch.float32)
        mask_array = cast(
            Float32Array,
            np.ascontiguousarray(mask.numpy(), dtype=np.float32),
        )
        self.lsh_.query_multi_mask(fp, mask_array, M, N)
        return torch.from_numpy(mask_array).to(device), fp

    def accidental_match(
        self,
        labels: torch.Tensor | Int64Array,
        samples: MutableSet[int],
        N: int,
    ) -> None:
        self.lsh_.accidental_match(
            _int64_array(labels), samples, N
        )

    def multi_label(
        self,
        labels: torch.Tensor | Int64Array,
        samples: MutableSet[int],
    ) -> tuple[list[int], Float32Array]:
        """Return sampled ids and a float32 target matrix.

        samples is mutated by the native implementation when label ids
        must be added to the sampled set.
        """
        return self.lsh_.multi_label(
            _int64_array(labels), samples
        )

    def multi_label_nonunion(
        self,
        labels: torch.Tensor | Int64Array,
        samples: Int64Array,
    ) -> tuple[Int64Array, Float32Array]:
        contiguous_samples = cast(
            Int64Array,
            np.ascontiguousarray(samples, dtype=np.int64),
        )
        return self.lsh_.multi_label_nonunion(
            _int64_array(labels), contiguous_samples
        )

    def query_remove_matrix(
        self,
        items: torch.Tensor,
        labels: Int32Array,
        total_size: int,
    ) -> tuple[Int64Array, Int32Array]:
        batch_size, _dim = items.size()
        fp = _fingerprint_array(self.func.hash(items))
        label_buffer = cast(
            Int32Array,
            np.ascontiguousarray(labels, dtype=np.int32),
        )
        result, total_count = self.lsh_.query_matrix(
            fp, label_buffer, batch_size, total_size
        )
        self.sample_size += total_count
        self.count += batch_size
        return result, fp

    def query_remove(
        self, item: torch.Tensor, label: int
    ) -> list[int]:
        result = self.query(item)
        result.discard(label)
        self.sample_size += len(result)
        self.count += 1
        return list(result)

    def print_stats(self) -> list[int]:
        print("in lsh new")
        return self.lsh_.print_stats()

    def clear(self) -> None:
        self.lsh_.clear()
