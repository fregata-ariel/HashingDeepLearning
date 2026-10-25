"""Small native CUDA, production wrapper, fallback and CUDA RNG contracts.

Original CUDA execution and maintained-wrapper execution are separate profiles.
No fixture downloads models or datasets or substitutes a CPU result for CUDA.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ports/mongoose"))


def original_kernel() -> str:
    tree = ast.parse((ROOT / "third_party/mongoose/mongoose_slide/slide_lib/simHash.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kernel" for t in node.targets):
            return str(ast.literal_eval(node.value))
    raise AssertionError("original fingerprint source missing")


def packing(profile: str) -> dict[str, object]:
    from mongoose_slide.slide_lib.simHash import SimHash, kernel
    raw = None
    if profile == "mongoose-original":
        import cupy as cp
        raw = cp.RawModule(code=original_kernel(), options=("--std=c++11",))
        function = raw.get_function("fingerprint")  # compile/load must actually happen
    cases = 0
    streams = [torch.cuda.default_stream(), torch.cuda.Stream()]
    for stream in streams:
        with torch.cuda.stream(stream):
            for n, k, tables in ((1, 1, 1), (3, 2, 5), (2, 31, 3), (5, 32, 2)):
                values = np.resize(np.array([-1., -0., 0., 1., 2., -2.], np.float32), (n, tables, k)).copy()
                # Force the sign bit, zero codes and all-one codes explicitly.
                values[0, 0, :] = 1
                if n > 1:
                    values[1, 0, :] = -0.
                expected = np.empty((n, tables), dtype=np.int32)
                for row in range(n):
                    for table in range(tables):
                        code = sum(1 << bit for bit in range(k) if float(values[row, table, bit]) > 0)
                        expected[row, table] = np.array(code, dtype=np.uint32).view(np.int32)
                src = torch.tensor(values.reshape(n, -1), device="cuda")
                if profile == "mongoose-original":
                    output = torch.full((n * tables + 2,), -71, dtype=torch.int64, device="cuda")
                    import cupy as cp
                    function((n, tables), (32,), (cp.asarray(src), np.int32(k), np.int32(tables), cp.asarray(output[1:-1])), stream=cp.cuda.ExternalStream(stream.cuda_stream))
                    result = output[1:-1].reshape(n, tables).int()
                    stream.synchronize()
                    assert output[0].item() == -71 and output[-1].item() == -71
                else:
                    hasher = SimHash(1, k, tables, weights=torch.ones(1, k * tables, device="cuda"))
                    if profile == "mongoose-fallback":
                        hasher.fp = None
                    else:
                        assert hasher.fp is not None
                    result = hasher.fingerprint(src, n)
                stream.synchronize()
                assert result.is_cuda and result.dtype == torch.int32
                np.testing.assert_array_equal(result.cpu().numpy(), expected)
                cases += 1
            # Exercise the actual projection -> pack -> transpose caller too.
            if profile != "mongoose-original":
                weights = torch.tensor([[1., -1., 0., 1.], [-1., 1., 1., 0.]], device="cuda")
                hasher = SimHash(2, 2, 2, weights=weights)
                if profile == "mongoose-fallback":
                    hasher.fp = None
                data = torch.tensor([[1., 2.], [-1., 0.]], device="cuda")
                assert torch.equal(hasher.hash(data).cpu(), torch.tensor([[2, 3], [2, 0]], dtype=torch.int32))
                assert torch.equal(hasher.hash(data, transpose=True).cpu(), torch.tensor([[2, 2], [3, 0]], dtype=torch.int32))
                stream.synchronize()
    return {"cases": cases, "streams": 2, "k": [1, 2, 31, 32], "signed_zero_and_bit31": True,
            "archived_kernel_sha256":hashlib.sha256(original_kernel().encode()).hexdigest(),
            "maintained_kernel_sha256":hashlib.sha256(kernel.encode()).hexdigest(),
            "original_cuda": profile == "mongoose-original", "production_wrapper": profile == "mongoose-wrapper",
            "torch_gpu_fallback": profile == "mongoose-fallback"}


def rng_replay() -> dict[str, object]:
    path = ROOT / "ports/mongoose/mongoose_reformer/reformer_lib/reversible.py"
    spec = importlib.util.spec_from_file_location("maintained_reversible", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(173)
    torch.cuda.manual_seed_all(173)
    f = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.3)).cuda()
    g = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2)).cuda()
    regular_f, regular_g = copy.deepcopy(f), copy.deepcopy(g)
    block = module.ReversibleBlock(f, g).train()
    x = torch.randn(2, 3, 8, device="cuda")
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual = block(x)
    assert block.f.cuda_in_fwd and block.g.cuda_in_fwd
    assert block.f.gpu_devices == [0] and block.g.gpu_devices == [0]
    after_forward = torch.cuda.get_rng_state()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    reference_x = x.detach().clone().requires_grad_()
    x1, x2 = reference_x.chunk(2, dim=2)
    y1 = x1 + regular_f(x2)
    y2 = x2 + regular_g(y1)
    expected = torch.cat((y1, y2), dim=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dy = torch.randn_like(actual)
    expected.backward(dy)
    before_backward = torch.cuda.get_rng_state()
    recovered, dx = block.backward_pass(actual.detach(), dy)
    assert torch.equal(before_backward, torch.cuda.get_rng_state()), "backward replay leaked RNG"
    assert torch.equal(after_forward, before_backward) is False  # dy consumed CUDA RNG
    torch.testing.assert_close(recovered, x, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(dx, reference_x.grad, rtol=2e-5, atol=2e-6)
    for a, b in zip(list(f.parameters()) + list(g.parameters()), list(regular_f.parameters()) + list(regular_g.parameters())):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
    torch.cuda.synchronize()
    return {"dropout_backward_parameter_and_input_gradients": True, "cuda_rng_restored": True,
            "devices": 1, "ddp_nccl": "NOT_RUN: requires a separate multi-device profile"}
