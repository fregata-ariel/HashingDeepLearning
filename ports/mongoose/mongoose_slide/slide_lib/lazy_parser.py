from __future__ import annotations

from typing import TypeAlias

import torch
from torch.utils.data import Dataset

DatasetItem: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class _BaseSparseDataset(Dataset[DatasetItem]):
    def __init__(self, filename: str, fraction: float = 1.0) -> None:
        super().__init__()
        self.N = 0
        self.D = 0
        self.L = 0
        self.max_L = 0
        self.max_D = 0
        self.data: list[DatasetItem] = []
        self._fraction = fraction
        self.build(filename)

    def build(self, filename: str) -> None:
        """Parse label,id:value records; feature values are intentionally ignored."""
        with open(filename, encoding="utf-8") as handle:
            metadata = handle.readline().split()
            total = int(metadata[0])
            self.N = int(total * self._fraction)
            self.D = int(metadata[1])
            self.L = int(metadata[2])

            for index in range(self.N):
                items = handle.readline().split()
                if not items:
                    raise ValueError(
                        f"unexpected end of dataset at record {index}"
                    )
                labels = [
                    int(value)
                    for value in items[0].split(",")
                ]
                self.max_L = max(self.max_L, len(labels))
                ids = [
                    int(item.split(":", 1)[0])
                    for item in items[1:]
                ]
                self.max_D = max(self.max_D, len(ids))
                self.data.append(
                    (
                        torch.tensor(labels, dtype=torch.long),
                        torch.tensor(ids, dtype=torch.long),
                    )
                )
                if index % 100000 == 0:
                    print(index)

    @staticmethod
    def pad(
        item: torch.Tensor, width: int, value: int
    ) -> torch.Tensor:
        result = torch.full(
            (width,), value, dtype=torch.long
        )
        result[: len(item)] = item
        return result

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> DatasetItem:
        labels, data = self.data[idx]
        return (
            self.pad(labels, self.max_L, -1),
            self.pad(data, self.max_D, self.D),
        )


class MultiLabelDataset(_BaseSparseDataset):
    """Full sparse multi-label dataset."""

    def __init__(self, filename: str) -> None:
        super().__init__(filename, fraction=1.0)


class ValidDataset(_BaseSparseDataset):
    """Historical validation subset using the first 2.5% of records."""

    def __init__(self, filename: str) -> None:
        super().__init__(filename, fraction=0.025)
