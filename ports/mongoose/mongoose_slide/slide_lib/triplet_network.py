from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class TripletNet(nn.Module):
    """Learn the SLIDE-side pairwise hash projection.

    This is the released pairwise/BCE objective, not Equation 3. arc and pair
    are floating tensors shaped (batch, layer_size); label is a floating
    binary target vector whose batch dimension is repeated across L tables.
    forward() returns one scalar BCE loss tensor.

    TRACE_TEST_ID: MONGOOSE-SLIDE-PAIRWISE-LOSS.
    """

    def __init__(
        self, margin: float, K: int, L: int, layer_size: int
    ) -> None:
        super().__init__()
        self.K = K
        self.L = L
        self.dense1 = nn.Linear(layer_size, K * L)
        bias = self.dense1.bias
        assert bias is not None
        self.init_weights(self.dense1.weight, bias)
        bias.requires_grad = False
        self.margin = margin
        self.device = device

    @staticmethod
    def init_weights(weight: torch.Tensor, bias: torch.Tensor) -> None:
        weight.data.normal_(0, 1)
        bias.data.fill_(0)

    def forward(
        self,
        arc: torch.Tensor,
        pair: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the differentiable pairwise hash-agreement BCE loss."""
        emb_arc = self.dense1(arc)
        emb_pair = self.dense1(pair)

        emb_arc_chunk = torch.cat(torch.chunk(emb_arc, self.L, dim=1))
        emb_pair_chunk = torch.cat(torch.chunk(emb_pair, self.L, dim=1))

        label_chunk = label.repeat(self.L)
        assert emb_arc_chunk.size() == emb_pair_chunk.size()
        assert emb_arc_chunk.size()[0] == label_chunk.size()[0]

        beta = 1.0
        emb_arc_chunk = torch.tanh(beta * emb_arc_chunk)
        emb_pair_chunk = torch.tanh(beta * emb_pair_chunk)

        output = torch.sum(emb_arc_chunk * emb_pair_chunk, dim=1)
        assert output.size() == label_chunk.size()
        return F.binary_cross_entropy(torch.sigmoid(output), label_chunk)
