"""Numerical oracle for MONGOOSE's SLIDE-side pairwise hash learner."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mongoose_slide.slide_lib.triplet_network import TripletNet

# TRACE_TEST_ID: MONGOOSE-SLIDE-PAIRWISE-LOSS


def test_slide_pairwise_bce_objective() -> None:
    model = TripletNet(margin=1.0, K=2, L=1, layer_size=2)
    with torch.no_grad():
        model.dense1.weight.copy_(torch.eye(2))
        model.dense1.bias.zero_()

    arc = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    pair = torch.tensor([[1.0, 0.0], [-1.0, -1.0]])
    label = torch.tensor([1.0, 0.0])

    actual = model(arc, pair, label)

    emb_arc = torch.tanh(arc)
    emb_pair = torch.tanh(pair)
    agreement = torch.sum(emb_arc * emb_pair, dim=1)
    expected = F.binary_cross_entropy(torch.sigmoid(agreement), label)

    assert torch.allclose(actual, expected)


if __name__ == "__main__":
    test_slide_pairwise_bce_objective()
    print("TRACE_TEST_PASS MONGOOSE-SLIDE-PAIRWISE-LOSS")
