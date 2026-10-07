from __future__ import annotations

import math
from typing import TypeAlias, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mongoose_slide.slide_lib.lsh import Int32Array, LSH
from mongoose_slide.slide_lib.simHash import SimHash

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

WeightPair: TypeAlias = dict[int, tuple[torch.Tensor, torch.Tensor]]
TrainForwardResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    float,
    WeightPair,
    Int32Array,
    float,
    float,
]
ForwardResult: TypeAlias = torch.Tensor | TrainForwardResult
MiningResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]
DatasetItem: TypeAlias = tuple[torch.Tensor, torch.Tensor]

np.random.seed(1234)
torch.manual_seed(1234)


class LSHSampledLayer(nn.Module):
    """Sample output classes through a learned SimHash/LSH index.

    x is float (batch, D); y is int64 (batch, label_slots). Training returns
    sampled logits/targets plus the C-contiguous int32 hash-code matrix.
    Evaluation returns the dense logits tensor.
    """

    def __init__(
        self,
        hash_weight: torch.Tensor,
        layer_size: int,
        K: int,
        L: int,
        num_class: int,
    ) -> None:
        super().__init__()
        self.D = layer_size
        self.K = K
        self.L = L
        self.num_class = num_class
        self.hash_weight = hash_weight
        self.store_query = True

        self.params = nn.Linear(layer_size, num_class)
        # The released SLIDE path stores bias as a column vector because it is
        # concatenated with class weights before hashing.
        self.params.bias = nn.Parameter(torch.empty(num_class, 1))
        self.init_weights(self.params.weight, self.params.bias)

        self.lsh: LSH
        self.initializeLSH()
        self.count = 0
        self.sample_size = 0

        self.thresh_hash = SimHash(self.D + 1, 1, self.L)
        self.thresh = 0.3
        weight_with_bias = torch.cat(
            (self.params.weight, self.params.bias), dim=1
        )
        self.hashcodes = self.thresh_hash.hash(weight_with_bias)

    def initializeLSH(self) -> None:
        self.lsh = LSH(
            SimHash(
                self.D + 1, self.K, self.L, self.hash_weight
            ),
            self.K,
            self.L,
        )
        weight_to_lsh = torch.cat(
            (self.params.weight, self.params.bias), dim=1
        )
        self.lsh.insert_multi(
            weight_to_lsh.to(device).data, self.num_class
        )

    def setSimHash(
        self,
        seed: int,
        hashweight: torch.Tensor | None = None,
    ) -> None:
        del seed
        print("update simhash")
        if hashweight is not None:
            self.lsh.setSimHash(
                SimHash(
                    self.D + 1, self.K, self.L, hashweight
                )
            )

    def rebuild(self) -> None:
        weight_to_lsh = torch.cat(
            (self.params.weight, self.params.bias), dim=1
        )
        check = self.thresh_hash.hash(weight_to_lsh)
        distance = check - self.hashcodes
        changed = float(torch.sum(torch.abs(distance)).item())
        if changed > self.thresh * distance.numel():
            print("Rebuild LSH")
            self.lsh.clear()
            self.lsh.insert_multi(
                weight_to_lsh.to(device).data, self.num_class
            )
            self.hashcodes = check
        else:
            print("No need")

    @staticmethod
    def init_weights(
        weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

    def train_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        triplet_flag: bool,
        debug: bool = False,
    ) -> TrainForwardResult:
        del triplet_flag, debug
        batch_size, _dim = x.size()
        query_to_lsh = torch.cat(
            (
                x,
                torch.ones(
                    batch_size, 1, device=x.device, dtype=x.dtype
                ),
            ),
            dim=1,
        )
        sampled_ids, hashcode = self.lsh.query_multi(
            query_to_lsh.data, batch_size
        )

        sampled_ip = 0.0
        sampled_cos = 0.0
        retrieved_size = 0.0

        sid_list, target_matrix = self.lsh.multi_label(
            y, sampled_ids
        )
        new_targets = torch.from_numpy(target_matrix).to(device)

        sample_ids = torch.from_numpy(
            np.asarray(sid_list, dtype=np.int64)
        ).to(device=device, dtype=torch.long)
        sample_size = int(sample_ids.size(0))
        sample_weights = F.embedding(
            sample_ids, self.params.weight, sparse=True
        )
        bias = self.params.bias
        if bias is None:
            raise RuntimeError("LSHSampledLayer bias unexpectedly missing")
        sample_bias = bias.squeeze()[sample_ids]
        sample_product = sample_weights.matmul(x.t()).t()
        sample_logits = sample_product + sample_bias
        self.lsh.sample_size += sample_size
        weight_pair: WeightPair = {}
        return (
            sample_logits,
            new_targets,
            sample_size,
            retrieved_size,
            weight_pair,
            hashcode,
            sampled_ip,
            sampled_cos,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        triplet_flag: bool,
        debug: bool,
    ) -> ForwardResult:
        if self.training:
            return self.train_forward(x, y, triplet_flag, debug)
        bias = self.params.bias
        if bias is None:
            raise RuntimeError("LSHSampledLayer bias unexpectedly missing")
        return (
            torch.matmul(x, self.params.weight.t())
            + bias.squeeze()
        )


class Net(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_size: int,
        hash_weight: torch.Tensor,
        K: int,
        L: int,
    ) -> None:
        super().__init__()
        stdv = 1.0 / math.sqrt(input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.fc = nn.Embedding(
            self.input_size + 1,
            128,
            padding_idx=input_size,
            sparse=True,
        )
        self.bias = nn.Parameter(torch.empty(layer_size))
        self.bias.data.uniform_(-stdv, stdv)
        self.lshLayer = LSHSampledLayer(
            hash_weight, layer_size, K, L, output_size
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        triplet_flag: bool,
        debug: bool,
    ) -> ForwardResult:
        emb = torch.sum(self.fc(x), dim=1)
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        query = F.relu(emb + self.bias)
        return cast(
            ForwardResult,
            self.lshLayer(query, y, triplet_flag, debug),
        )

    def forward_full(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t1: float,
        t2: float,
    ) -> MiningResult:
        del y, t1, t2
        with torch.no_grad():
            batch_size, _dim = x.size()
            emb = torch.sum(self.fc(x), dim=1)
            emb = emb / torch.norm(
                emb, dim=1, keepdim=True
            )
            query = F.relu(emb + self.bias)
            product = torch.matmul(
                query, self.lshLayer.params.weight.t()
            )

            positive_rank = int(
                self.output_size * (1 - 0.001)
            )
            negative_rank = int(
                self.output_size * (1 - 0.5)
            )
            positive_ip = float(
                torch.mean(
                    torch.kthvalue(product, positive_rank)[0]
                ).item()
            )
            positive_ip = max(0.0, positive_ip)
            negative_ip = float(
                torch.mean(
                    torch.kthvalue(product, negative_rank)[0]
                ).item()
            )

            positive_mask = product > positive_ip
            negative_mask = product < negative_ip

            query_with_bias = torch.cat(
                (
                    query.data,
                    torch.ones(
                        batch_size, 1, device=query.device
                    ),
                ),
                dim=1,
            )
            retrieved, _ = self.lshLayer.lsh.query_multi_mask(
                query_with_bias,
                batch_size,
                self.output_size,
            )
            retrieved_bool = retrieved.bool()
            positive_mask &= ~retrieved_bool
            negative_mask &= retrieved_bool

            num_negative = int(
                torch.sum(negative_mask).item()
            )
            num_positive = int(
                torch.sum(positive_mask).item()
            )

            row, column = torch.where(positive_mask)
            p_arc = query_with_bias[row].detach()
            bias = self.lshLayer.params.bias
            if bias is None:
                raise RuntimeError("sampled-layer bias missing")
            p_pos = torch.cat(
                (
                    self.lshLayer.params.weight[column],
                    bias[column],
                ),
                dim=1,
            ).detach()

            row, column = torch.where(negative_mask)
            n_arc = query_with_bias[row].detach()
            n_neg = torch.cat(
                (
                    self.lshLayer.params.weight[column],
                    bias[column],
                ),
                dim=1,
            ).detach()

            if num_positive < num_negative:
                ids = torch.randperm(num_negative)[:num_positive]
                n_arc = n_arc[ids]
                n_neg = n_neg[ids]
                num_negative = int(n_arc.size(0))
            else:
                ids = torch.randperm(num_positive)[:num_negative]
                p_arc = p_arc[ids]
                p_pos = p_pos[ids]

        return p_arc, p_pos, n_arc, n_neg, num_negative


class MultiLabelDataset(Dataset[DatasetItem]):
    """Parse sparse multi-label text records into padded int64 tensors."""

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.N = 0
        self.D = 0
        self.L = 0
        self.max_L = 0
        self.max_D = 0
        self.data: list[DatasetItem] = []
        self.build(filename)

    def build(self, filename: str) -> None:
        with open(filename, encoding="utf-8") as handle:
            metadata = handle.readline().split()
            self.N = int(metadata[0])
            self.D = int(metadata[1])
            self.L = int(metadata[2])
            for _ in range(self.N):
                items = handle.readline().split()
                labels = [int(x) for x in items[0].split(",")]
                self.max_L = max(self.max_L, len(labels))
                ids = [
                    int(item.split(":")[0])
                    for item in items[1:]
                ]
                self.max_D = max(self.max_D, len(ids))
                self.data.append(
                    (
                        torch.tensor(labels, dtype=torch.long),
                        torch.tensor(ids, dtype=torch.long),
                    )
                )

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
        labels, indices = self.data[idx]
        return (
            self.pad(labels, self.max_L, -1),
            self.pad(indices, self.max_D, self.D),
        )
