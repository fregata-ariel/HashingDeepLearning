"""Shape/domain oracle for MONGOOSE SimHash and Reformer learnable LSH."""
from __future__ import annotations

import torch

from mongoose_slide.slide_lib.simHash import SimHash
from mongoose_reformer.reformer_lib.reformer_pytorch import TripletLSHAttention

# TRACE_TEST_ID: MONGOOSE-REFORMER-HASH-SHAPES


def test_simhash_and_reformer_hash_shapes() -> None:
    simhash = SimHash(d_=3, k_=2, L_=3, seed_=17)
    data = torch.tensor(
        [[1.0, -2.0, 0.5], [-1.5, 0.25, 2.0]], dtype=torch.float32
    )
    fp = simhash.hash(data)
    assert fp.shape == (2, 3)
    assert fp.dtype == torch.int32
    assert torch.all(fp >= 0)
    assert torch.all(fp < (1 << 2))
    assert simhash.hash(data, transpose=True).shape == (3, 2)

    attn = TripletLSHAttention(
        alpha=1.0,
        dim=4,
        seq_len=8,
        heads=2,
        bucket_size=2,
        n_hashes=2,
        dropout=0.0,
    )

    # qk has already been merged across batch and attention heads when it
    # reaches TripletLSHAttention. Here B*H == 2 and dim_head == 2.
    qk = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0],
             [0.0, -1.0], [-1.0, -1.0], [0.5, 0.5], [-0.5, 0.5]],
            [[0.5, -0.5], [1.0, 0.5], [-0.5, 1.0], [-1.0, -0.5],
             [0.25, 0.75], [0.75, -0.25], [-0.75, 0.25], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    n_buckets = 8 // 2
    rotations = attn.extract_rotations(batch_size=qk.shape[0])
    assert rotations.shape == (2, 2, 2, n_buckets)
    assert rotations.requires_grad is False

    buckets = attn.hash_vectors(n_buckets, qk, rotations=rotations)
    assert buckets.shape == (2, 2 * 8)
    assert buckets.dtype == torch.int64
    assert torch.all(buckets >= 0)
    assert torch.all(buckets < 2 * n_buckets)


if __name__ == "__main__":
    test_simhash_and_reformer_hash_shapes()
    print("TRACE_TEST_PASS MONGOOSE-REFORMER-HASH-SHAPES")
