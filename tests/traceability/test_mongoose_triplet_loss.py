"""Numerical oracle for MONGOOSE Reformer-side learnable-LSH loss."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from mongoose_reformer.reformer_lib.reformer_pytorch import TripletLSHAttention

# TRACE_TEST_ID: MONGOOSE-TRIPLET-LOSS

def test_triplet_loss_matches_per_example_hinge() -> None:
    layer = TripletLSHAttention(alpha=0.25, dim=2, seq_len=4, heads=1,
                                bucket_size=2, n_hashes=1, dropout=0.0)
    with torch.no_grad():
        layer.rotations.weight.copy_(torch.eye(2))

    x = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    n = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], requires_grad=True)

    actual = layer.triplet_forward(x, p, n)
    ex = layer.rotations(x.detach())
    ep = layer.rotations(p.detach())
    en = layer.rotations(n.detach())
    expected = torch.relu(
        F.cosine_similarity(ex, en, dim=-1)
        - F.cosine_similarity(ex, ep, dim=-1)
        + layer.alpha
    ).mean()
    assert torch.allclose(actual, expected)

    actual.backward()
    assert layer.rotations.weight.grad is not None
    assert x.grad is None and p.grad is None and n.grad is None

if __name__ == "__main__":
    test_triplet_loss_matches_per_example_hinge()
    print("TRACE_TEST_PASS MONGOOSE-TRIPLET-LOSS")
