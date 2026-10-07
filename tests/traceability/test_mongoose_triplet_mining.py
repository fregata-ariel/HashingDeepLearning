"""Numerical oracle for released MONGOOSE triplet example mining."""
from __future__ import annotations

import torch

from mongoose_reformer.reformer_lib.reformer_pytorch import mine_triplet_examples

# TRACE_TEST_ID: MONGOOSE-TRIPLET-MINING


def test_triplet_example_indices_and_detach() -> None:
    qk = torch.tensor(
        [[[10.0, 0.0], [20.0, 1.0], [30.0, 2.0], [40.0, 3.0]]],
        requires_grad=True,
    )
    candidate_indices = torch.tensor(
        [[
            [[0, 1, 2, 3], [1, 0, 3, 2]],
            [[2, 3, 0, 1], [3, 2, 1, 0]],
        ]],
        dtype=torch.long,
    )
    attention_probs = torch.tensor(
        [[
            [[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.2, 0.6], [0.1, 0.7, 0.1, 0.1]],
        ]]
    )
    negative_samples = torch.tensor(
        [[[1, 3], [0, 2]]], dtype=torch.long
    )
    sorted_query_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    pos, neg = mine_triplet_examples(
        qk,
        attention_probs,
        candidate_indices,
        sorted_query_indices,
        negative_samples,
    )

    expected_pos_indices = torch.tensor([[2, 1, 1, 2]], dtype=torch.long)
    expected_neg_indices = torch.tensor([[1, 2, 2, 1]], dtype=torch.long)
    expected_pos = qk.detach().gather(
        1, expected_pos_indices[:, :, None].expand(-1, -1, qk.shape[-1])
    )
    expected_neg = qk.detach().gather(
        1, expected_neg_indices[:, :, None].expand(-1, -1, qk.shape[-1])
    )

    assert torch.equal(pos, expected_pos)
    assert torch.equal(neg, expected_neg)
    assert pos.requires_grad is False
    assert neg.requires_grad is False


if __name__ == "__main__":
    test_triplet_example_indices_and_detach()
    print("TRACE_TEST_PASS MONGOOSE-TRIPLET-MINING")
