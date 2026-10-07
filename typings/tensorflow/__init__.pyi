from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import overload

import numpy as np
import numpy.typing as npt


class DType: ...
int64: DType
float32: DType


class Tensor:
    def __add__(self, other: Tensor) -> Tensor: ...
    def __radd__(self, other: Tensor) -> Tensor: ...
    def __getitem__(self, key: object) -> Tensor: ...


class SparseTensor(Tensor):
    def __init__(
        self,
        indices: Tensor,
        values: Tensor,
        dense_shape: Sequence[int],
    ) -> None: ...


class Operation: ...


class _GPUOptions:
    allow_growth: bool


class ConfigProto:
    gpu_options: _GPUOptions

    def __init__(
        self,
        *,
        inter_op_parallelism_threads: int | None = ...,
        intra_op_parallelism_threads: int | None = ...,
    ) -> None: ...


def placeholder(
    dtype: DType,
    shape: Sequence[int | None],
) -> Tensor: ...


def truncated_normal(
    shape: Sequence[int],
    *,
    stddev: float,
) -> Tensor: ...


def Variable(initial_value: Tensor) -> Tensor: ...


def matmul(a: Tensor, b: Tensor) -> Tensor: ...


def transpose(a: Tensor) -> Tensor: ...


def argmax(a: Tensor, axis: int) -> Tensor: ...


def reduce_mean(a: Tensor) -> Tensor: ...


def sparse_tensor_dense_matmul(a: SparseTensor, b: Tensor) -> Tensor: ...


def global_variables_initializer() -> Operation: ...


class _NN:
    def relu(self, x: Tensor) -> Tensor: ...

    def top_k(
        self,
        x: Tensor,
        *,
        k: int,
        sorted: bool,
    ) -> tuple[Tensor, Tensor]: ...

    def softmax_cross_entropy_with_logits(
        self,
        *,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor: ...

    def sampled_softmax_loss(
        self,
        weights: Tensor,
        biases: Tensor,
        labels: Tensor,
        inputs: Tensor,
        num_sampled: int,
        num_classes: int,
        *,
        remove_accidental_hits: bool,
        num_true: int,
        partition_strategy: str,
    ) -> Tensor: ...


nn: _NN


class _AdamOptimizer:
    def __init__(self, learning_rate: float) -> None: ...
    def minimize(self, loss: Tensor) -> Operation: ...


class _Train:
    AdamOptimizer: type[_AdamOptimizer]


train: _Train


class Session:
    def __init__(self, *, config: ConfigProto) -> None: ...

    @overload
    def run(
        self,
        fetches: Operation,
        feed_dict: Mapping[object, object] | None = ...,
    ) -> None: ...

    @overload
    def run(
        self,
        fetches: Tensor,
        feed_dict: Mapping[object, object] | None = ...,
    ) -> npt.NDArray[np.int64]: ...
