from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

SparseIndex: TypeAlias = tuple[int, int]
DenseTargets: TypeAlias = npt.NDArray[np.float32]
DenseBatch: TypeAlias = tuple[list[SparseIndex], list[float], DenseTargets]
SampledBatch: TypeAlias = tuple[list[SparseIndex], list[float], list[list[int]]]
TestBatch: TypeAlias = tuple[list[SparseIndex], list[float], list[list[int]]]


def _full_batches(
    files: Sequence[str], batch_size: int
) -> Iterator[list[str]]:
    while True:
        lines: list[str] = []
        for filename in files:
            with open(filename, "r", encoding="utf-8") as handle:
                handle.readline()
                while True:
                    remaining = batch_size - len(lines)
                    lines += list(islice(handle, remaining))
                    if len(lines) != batch_size:
                        break
                    yield lines
                    lines = []


def data_generator(
    files: Sequence[str],
    batch_size: int,
    n_classes: int,
) -> Iterator[DenseBatch]:
    """Yield sparse feature coordinates and dense float32 label targets."""
    for lines in _full_batches(files, batch_size):
        indices: list[SparseIndex] = []
        values: list[float] = []
        targets = np.zeros(
            (batch_size, n_classes), dtype=np.float32
        )

        for row, line in enumerate(lines):
            items = line.strip().split(" ")
            labels = [int(item) for item in items[0].split(",")]
            probability = 1.0 / len(labels)
            for label in labels:
                targets[row, label] = probability

            for item in items[1:]:
                feature, value = item.split(":")
                indices.append((row, int(feature)))
                values.append(float(value))

        yield indices, values, targets


def data_generator_ss(
    files: Sequence[str],
    batch_size: int,
    n_classes: int,
    max_label: int,
) -> Iterator[SampledBatch]:
    """Yield sparse inputs and fixed-width sampled-softmax label ids."""
    for lines in _full_batches(files, batch_size):
        indices: list[SparseIndex] = []
        values: list[float] = []
        targets: list[list[int]] = []

        for row, line in enumerate(lines):
            items = line.strip().split(" ")
            labels = [int(item) for item in items[0].split(",")]
            if max_label >= len(labels):
                selected = labels + [n_classes] * (
                    max_label - len(labels)
                )
            else:
                chosen = np.random.choice(
                    labels, max_label, replace=False
                )
                selected = cast(list[int], chosen.tolist())
            targets.append(selected)

            for item in items[1:]:
                feature, value = item.split(":")
                indices.append((row, int(feature)))
                values.append(float(value))

        yield indices, values, targets


def data_generator_tst(
    files: Sequence[str],
    batch_size: int,
) -> Iterator[TestBatch]:
    """Yield sparse inputs and unpadded integer label lists."""
    for lines in _full_batches(files, batch_size):
        indices: list[SparseIndex] = []
        values: list[float] = []
        targets: list[list[int]] = []

        for row, line in enumerate(lines):
            items = line.strip().split(" ")
            targets.append(
                [int(item) for item in items[0].split(",")]
            )
            for item in items[1:]:
                feature, value = item.split(":")
                indices.append((row, int(feature)))
                values.append(float(value))

        yield indices, values, targets
