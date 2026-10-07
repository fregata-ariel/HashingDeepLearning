"""Runtime contract for typed MONGOOSE-SLIDE hash/LSH/softmax wrappers."""
from __future__ import annotations

import numpy as np
import torch

from mongoose_slide.slide_lib.lsh import LSH
from mongoose_slide.slide_lib.lsh_softmax import LSHSoftmax
from mongoose_slide.slide_lib.simHash import SimHash

# TRACE_TEST_ID: MONGOOSE-SLIDE-LSH-WRAPPER


def test_typed_slide_lsh_runtime_contract() -> None:
    torch.manual_seed(11)

    hasher = SimHash(d_=2, k_=1, L_=2, seed_=13)
    lsh = LSH(hasher, K_=1, L_=2, threads_=1)

    items = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    lsh.insert_multi(items, 2)

    for item_id in range(2):
        result = lsh.query(items[item_id : item_id + 1])
        assert item_id in result

    union, fp = lsh.query_multi(items, 2)
    assert {0, 1}.issubset(union)
    assert fp.dtype == np.int32
    assert fp.flags.c_contiguous
    assert fp.shape == (2, 2)

    labels = torch.tensor([[0], [1]], dtype=torch.int64)
    sample_ids, targets = lsh.multi_label(labels, set(union))
    assert targets.dtype == np.float32
    assert targets.shape == (2, len(sample_ids))
    assert targets[0, sample_ids.index(0)] == 1.0
    assert targets[1, sample_ids.index(1)] == 1.0

    mask, mask_fp = lsh.query_multi_mask(items, 2, 2)
    assert mask.dtype == torch.float32
    assert mask.shape == (2, 2)
    assert mask_fp.dtype == np.int32
    assert mask_fp.flags.c_contiguous


def test_typed_lsh_softmax_train_and_eval_contract() -> None:
    torch.manual_seed(19)
    layer = LSHSoftmax(N=4, D=2, K=1, L=2, freq=1)

    inputs = torch.tensor(
        [[1.0, 0.5], [-0.25, 1.0]], dtype=torch.float32
    )
    labels = torch.tensor([[0], [3]], dtype=torch.int64)

    layer.train()
    train_result = layer(inputs, labels)
    assert isinstance(train_result, tuple)
    logits, targets, sample_size, sampled_ip, sampled_cos = train_result
    assert logits.shape == targets.shape
    assert logits.shape[0] == inputs.shape[0]
    assert targets.dtype == torch.float32
    assert sample_size == logits.shape[1]
    assert isinstance(sampled_ip, (float, torch.Tensor))
    assert isinstance(sampled_cos, (float, torch.Tensor))

    layer.eval()
    dense_logits = layer(inputs, labels)
    assert torch.is_tensor(dense_logits)
    assert dense_logits.shape == (2, 4)


if __name__ == "__main__":
    test_typed_slide_lsh_runtime_contract()
    test_typed_lsh_softmax_train_and_eval_contract()
    print("TRACE_TEST_PASS MONGOOSE-SLIDE-LSH-WRAPPER")
