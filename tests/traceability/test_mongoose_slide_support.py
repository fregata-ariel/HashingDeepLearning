"""Runtime contracts for MONGOOSE-SLIDE support modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from mongoose_slide.slide_lib.adam_base import Adam
from mongoose_slide.slide_lib.cupy_kernel import cupyKernel
from mongoose_slide.slide_lib.lazy_parser import MultiLabelDataset, ValidDataset
from mongoose_slide.slide_lib.network import LSHSampledLayer

# TRACE_TEST_ID: MONGOOSE-SLIDE-SUPPORT-CONTRACTS


def test_custom_adam_dense_and_sparse_state() -> None:
    dense = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    dense.grad = torch.tensor([0.5, -0.25])
    dense_opt = Adam([dense], lr=1e-2)
    before = dense.detach().clone()
    dense_opt.step()
    assert not torch.equal(dense.detach(), before)
    assert dense_opt.state[dense]["step"] == 1

    sparse = torch.nn.Parameter(torch.zeros(4, 2))
    indices = torch.tensor([[1, 3]], dtype=torch.long)
    values = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    sparse.grad = torch.sparse_coo_tensor(
        indices, values, size=(4, 2)
    )
    sparse_opt = Adam([sparse], lr=1e-2)
    sparse_opt.step()
    assert torch.count_nonzero(sparse.detach()[0]) == 0
    assert torch.count_nonzero(sparse.detach()[2]) == 0
    assert torch.count_nonzero(sparse.detach()[1]) > 0
    assert torch.count_nonzero(sparse.detach()[3]) > 0


def test_sparse_dataset_parser_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dataset.txt"
        lines = ["40 8 4"]
        for index in range(40):
            labels = "0,2" if index % 2 == 0 else "1"
            lines.append(f"{labels} 1:1.0 3:0.5")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        train = MultiLabelDataset(str(path))
        labels, features = train[0]
        assert len(train) == 40
        assert labels.dtype == torch.long
        assert features.dtype == torch.long
        assert labels.shape == (2,)
        assert features.shape == (2,)

        valid = ValidDataset(str(path))
        assert len(valid) == 1
        vlabels, vfeatures = valid[0]
        assert vlabels.dtype == torch.long
        assert vfeatures.dtype == torch.long


def test_cuda_wrapper_is_lazy_on_cpu() -> None:
    wrapper = cupyKernel(
        'extern "C" __global__ void noop() {}',
        "noop",
    )
    assert wrapper.compiled is False
    assert wrapper.func is None


def test_sampled_network_train_eval_contract() -> None:
    torch.manual_seed(29)
    dimensions = 2
    K = 1
    L = 2
    classes = 4
    hash_weight = torch.randn(dimensions + 1, K * L)
    layer = LSHSampledLayer(
        hash_weight=hash_weight,
        layer_size=dimensions,
        K=K,
        L=L,
        num_class=classes,
    )

    inputs = torch.tensor(
        [[1.0, 0.25], [-0.5, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0], [3]], dtype=torch.int64)

    layer.train()
    result = layer(inputs, labels, False, False)
    assert isinstance(result, tuple)
    logits, targets, sample_size, _retrieved, _pairs, fp, _ip, _cos = result
    assert logits.shape == targets.shape
    assert sample_size == logits.shape[1]
    assert fp.dtype.name == "int32"
    assert fp.flags.c_contiguous

    layer.eval()
    dense = layer(inputs, labels, False, False)
    assert torch.is_tensor(dense)
    assert dense.shape == (2, classes)


if __name__ == "__main__":
    test_custom_adam_dense_and_sparse_state()
    test_sparse_dataset_parser_contract()
    test_cuda_wrapper_is_lazy_on_cpu()
    test_sampled_network_train_eval_contract()
    print("TRACE_TEST_PASS MONGOOSE-SLIDE-SUPPORT-CONTRACTS")
