from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mongoose_slide.slide_lib.lsh import LSH
from mongoose_slide.slide_lib.simHash import SimHash

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

SampleMetric: TypeAlias = float | torch.Tensor
SampledResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    SampleMetric,
    SampleMetric,
]


class LSHSoftmax(nn.Module):
    """Sample an output-class subset with the native SimHash/LSH path.

    Training inputs are floating tensors shaped (batch, D); labels are int64
    tensors shaped (batch, label_slots). The sampled logits and float32 target
    matrix share the sampled-class axis. In evaluation, forward returns the
    dense (batch, N) logits tensor instead of the training 5-tuple.
    """

    def __init__(
        self, N: int, D: int, K: int, L: int, freq: int
    ) -> None:
        super().__init__()
        self.D = D
        self.N = N
        self.K = K
        self.L = L
        self.freq = freq
        self.count = 0
        self.sample_size = 0
        self.lsh = LSH(SimHash(D, K, L), K, L)

        self.params = nn.Linear(D, N)
        bias = self.params.bias
        if bias is None:
            raise RuntimeError("LSHSoftmax requires a bias vector")
        self.init_weights(self.params.weight, bias)

    @staticmethod
    def init_weights(
        weight: torch.Tensor, bias: torch.Tensor
    ) -> None:
        initrange = 0.05
        weight.data.uniform_(-initrange, initrange)
        bias.data.fill_(0)

    def build(self, lsh: LSH) -> None:
        lsh.clear()
        lsh.insert_multi(
            self.params.weight.to(device).data, self.N
        )

    def sampled(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        debug: bool = False,
    ) -> SampledResult:
        """Return sampled logits/targets and optional retrieval diagnostics.

        labels are normalized to the native int64 buffer contract by LSH.
        The internal hash-code array is C-contiguous int32 with shape
        (batch, L), as validated at the Cython boundary.
        """
        if self.lsh.count % self.freq == 0:
            print("RESET HASH!!!")
            self.build(self.lsh)

        batch_size, _dim = inputs.size()
        sid, hashcode = self.lsh.query_multi(
            inputs.data, batch_size
        )
        sampled_ip: SampleMetric = 0.0
        sampled_cos: SampleMetric = 0.0

        if debug:
            product_list: list[float] = []
            dif_list: list[float] = []
            cos_list: list[float] = []
            for index, fingerprint in enumerate(hashcode):
                vector = inputs[index]
                sid_debug = self.lsh.query_fp(fingerprint)
                if not sid_debug:
                    continue

                retrieved = torch.from_numpy(
                    np.asarray(
                        list(sid_debug), dtype=np.int64
                    )
                ).to(device=device, dtype=torch.long)
                random_sample_ids = torch.randint(
                    0,
                    self.N,
                    (len(sid_debug),),
                    device=device,
                )
                random_weights = F.embedding(
                    random_sample_ids,
                    self.params.weight,
                    sparse=True,
                )
                random_product = random_weights.matmul(
                    vector.t()
                ).t()
                random_cos = 1 - torch.acos(
                    F.cosine_similarity(
                        vector.repeat(
                            1, random_weights.size(0)
                        ).view(
                            random_weights.size(0), -1
                        ),
                        random_weights,
                    )
                ) / torch.pi

                weights = F.embedding(
                    retrieved, self.params.weight, sparse=True
                )
                product = weights.matmul(vector.t()).t()
                cos = 1 - torch.acos(
                    F.cosine_similarity(
                        vector.repeat(
                            1, weights.size(0)
                        ).view(weights.size(0), -1),
                        weights,
                    )
                ) / torch.pi

                cos_list.append(
                    float(
                        torch.mean(cos).item()
                        - torch.mean(random_cos).item()
                    )
                )
                product_list.append(
                    float(torch.mean(product).item())
                )
                dif_list.append(
                    float(
                        torch.mean(product).item()
                        - torch.mean(random_product).item()
                    )
                )

            dif_tensor = torch.from_numpy(
                np.asarray(dif_list, dtype=np.float32)
            )
            cos_tensor = torch.from_numpy(
                np.asarray(cos_list, dtype=np.float32)
            )
            print(
                "mean of sampe - random ",
                torch.mean(dif_tensor),
            )
            print(
                "mean of sampe - random cos",
                torch.mean(cos_tensor),
            )
            sampled_ip = torch.mean(dif_tensor)
            sampled_cos = torch.mean(cos_tensor)

        sid_list, target_matrix = self.lsh.multi_label(
            labels, sid
        )
        new_targets = torch.from_numpy(target_matrix).to(device)

        sample_ids = torch.from_numpy(
            np.asarray(sid_list, dtype=np.int64)
        ).to(device=device, dtype=torch.long)
        sample_size = int(sample_ids.size(0))
        self.lsh.sample_size += sample_size
        self.lsh.count += 1

        sample_weights = F.embedding(
            sample_ids, self.params.weight, sparse=True
        )
        bias = self.params.bias
        if bias is None:
            raise RuntimeError("LSHSoftmax bias unexpectedly missing")
        sample_bias = bias[sample_ids]
        sample_logits = (
            sample_weights.matmul(inputs.t()).t()
            + sample_bias
        )
        return (
            sample_logits,
            new_targets,
            sample_size,
            sampled_ip,
            sampled_cos,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor | SampledResult:
        if self.training:
            return self.sampled(inputs, labels, debug)
        bias = self.params.bias
        if bias is None:
            raise RuntimeError("LSHSoftmax bias unexpectedly missing")
        return torch.matmul(inputs, self.params.weight.t()) + bias
