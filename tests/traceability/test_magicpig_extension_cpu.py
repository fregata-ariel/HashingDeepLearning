"""TRACE_TEST_ID: MAGICPIG-TORCH-CPU; actual Tensor/native boundary oracles.

Budgets reuse MAGICPIG-ATTENTION-CPU native FP64 oracle: probabilities .002,
BF16 WV .008, base2 metadata .03. Inputs are identically quantized before the
FP64 expectation. No post-owner borrowed view is ever dereferenced.
"""
from __future__ import annotations

import gc
import math
import struct
from types import ModuleType
import weakref

import torch
from torch import Tensor


def rejected(call: object) -> None:
    """Require an operation to reject before mutation."""
    from typing import Callable, cast
    try:
        cast(Callable[[], object], call)()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("invalid operation accepted")


def reference(k: Tensor, v: Tensor, q: Tensor, ids: list[int]) -> tuple[Tensor, Tensor, float, float]:
    """Independent FP64 subset oracle; literal binomial event enumeration K3/L4."""
    keys, values, query = k.double(), v.double(), q.double()
    logits: list[float] = []
    for row in ids:
        dot = sum(float(query[d]) * float(keys[row, d]) for d in range(q.numel()))
        qn = math.sqrt(sum(float(x) ** 2 for x in query))
        kn = math.sqrt(sum(float(x) ** 2 for x in keys[row]))
        cosine = dot / (qn * kn)
        assert -1 < cosine < 1
        hit = (1 - math.acos(cosine) / math.pi) ** 3
        inclusion = sum(math.comb(4, count) * hit ** count * (1-hit) ** (4-count) for count in range(2, 5))
        logits.append(dot / math.sqrt(q.numel()) - math.log(inclusion + 1e-4))
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    denominator = sum(weights)
    probabilities = torch.tensor([x / denominator for x in weights], dtype=torch.float64)
    output = torch.tensor([sum(float(probabilities[i]) * float(values[row, d]) for i, row in enumerate(ids)) for d in range(q.numel())], dtype=torch.float64)
    return probabilities, output, maximum / math.log(2), (maximum + math.log(denominator)) / math.log(2)


def exercise(sparse: ModuleType, lsh: ModuleType, bf16: bool) -> dict[str, object]:
    """Exercise wrapper routes, six actual entries, precommit and borrowed aliases."""
    torch.set_num_threads(4)
    assert sparse.openmp_threads() == 4
    assert hasattr(sparse.SparseAttentionServer, "attention_wrapper_bf16") == bf16
    unallocated = sparse.SparseAttentionServer()
    rejected(unallocated.clear)
    rejected(unallocated.get_score)
    traces: list[list[str]] = []
    checked = 0
    for heads, dim in [(32, 128), (32, 32), (64, 32), (72, 32)]:
        batches, kvheads, capacity = 2, 2, 5
        queryheads = heads // batches
        groups = queryheads // kvheads
        server = sparse.SparseAttentionServer()
        server.alloc(2, queryheads, kvheads, dim, batches, capacity)
        base = torch.arange(batches * kvheads * capacity * dim).reshape(batches, kvheads, capacity, dim)
        keys = (((base * 7 + 3) % 29 - 14).float() / 32).bfloat16()
        values = (((base * 3 + 5) % 37 - 18).float() / 32).bfloat16()
        norms = keys.float().square().sum(-1).sqrt()
        for request in range(batches):
            server.fill(1, request, keys[request], values[request], norms[request])
        qbase = torch.arange(heads * dim).reshape(heads, dim)
        queries = (((qbase * 11 + 1) % 31 - 15).float() / 32).bfloat16()
        qnorm = queries.float().square().sum(-1).sqrt()
        indices = torch.full((heads, capacity), -123, dtype=torch.int32)
        counts = torch.tensor([0 if h % 4 == 0 else 1 if h % 4 == 1 else 3 for h in range(heads)], dtype=torch.int32)
        for h in range(heads):
            ids = [4, 1, 3][:int(counts[h])]
            indices[h, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
        output = torch.full((heads, dim), 9, dtype=torch.bfloat16)
        metadata = torch.full((2, heads), 99, dtype=torch.float32)
        args = (1, 3, 4, output, metadata, queries, qnorm, indices, counts)
        server.attention_wrapper(*args)
        expected_route = 'scheduled_attention' if heads == 32 and dim == 128 else 'dynamic_attention' if heads > 64 else 'attention'
        suffix = '_bf16' if bf16 else ''
        trace = list(server.dispatch_trace)
        assert trace[0] == 'attention_wrapper'
        if bf16:
            assert trace[1] == 'attention_wrapper_bf16'
        expected_trace = ['attention_wrapper'] + (['attention_wrapper_bf16'] if bf16 else [])
        if expected_route == 'scheduled_attention':
            expected_trace += ['scheduled_attention' + suffix, 'attention' + suffix]
        else:
            expected_trace += [expected_route + suffix]
        assert trace == expected_trace
        traces.append(trace)
        if bf16:
            rejected(lambda: server.attention_wrapper(1, 3, 4, output, metadata, queries.float(), qnorm, indices, counts))
        if dim != 128:
            rejected(lambda: server.scheduled_attention(*args))
            if bf16:
                rejected(lambda: server.scheduled_attention_bf16(*args))
        scores = server.get_score().reshape(heads, capacity)
        for h in range(heads):
            n = int(counts[h])
            if not n:
                assert not output[h].count_nonzero()
                assert bool(torch.isneginf(metadata[:, h]).all())
                continue
            request = h // queryheads
            kvhead = (h % queryheads) // groups
            probabilities, expected, maximum, lse = reference(keys[request, kvhead], values[request, kvhead], queries[h], indices[h, :n].tolist())
            assert float((scores[h, :n].double() - probabilities).abs().max()) <= .002
            assert float((output[h].double() - expected).abs().max()) <= .008
            assert abs(float(metadata[0, h]) - maximum) <= .03
            assert abs(float(metadata[1, h]) - lse) <= .03
            checked += 1
        # Invalid late-head domain/index never commits earlier scores/output.
        before = (output.clone(), metadata.clone(), scores.clone())
        indices[-1, 0] = capacity
        rejected(lambda: server.attention_wrapper(*args))
        assert torch.equal(output, before[0]) and torch.equal(metadata, before[1]) and torch.equal(scores, before[2])
        indices[-1, 0] = 4
        bad_norm = qnorm.clone(); bad_norm[-1] = 0
        rejected(lambda: server.attention_wrapper(1, 3, 4, output, metadata, queries, bad_norm, indices, counts))
        rejected(lambda: server.attention_wrapper(1, 3, 4, output.float(), metadata, queries, qnorm, indices, counts))
        rejected(lambda: server.attention_wrapper(1, 3, 4, output, metadata, queries.t().contiguous().t(), qnorm, indices, counts))
        rejected(lambda: server.attention_wrapper(1, 3, 4, output, metadata, queries.to('meta'), qnorm, indices, counts))
        alias = server.get_key_cache(1)
        alias[0, 0, 0, 0] = 2
        assert server.get_key_cache(1)[0, 0, 0, 0] == 2
        server.clear()
        assert not alias.count_nonzero() and not scores.count_nonzero()
        for request in range(batches):
            server.fill(1, request, keys[request], values[request], norms[request])
        assert torch.equal(alias, keys)
        # Getter keeps the direct owner alive; never read after owner destruction.
        owner = weakref.ref(server)
        del scores, args, server
        gc.collect()
        assert owner() is not None
        del alias
        gc.collect()
        # `before` are cloned tensors; no owner-borrowed memory retained here.
        assert owner() is None
    # Actual wheel FBGEMM runtime dispatcher, half-up bit-model/ties-even evidence.
    inputs = torch.tensor([1.00390625, -1.00390625, .125, 0., 1.], dtype=torch.float32).repeat(7)
    converted = sparse.fbgemm_convert(inputs)
    model_bits = [((struct.unpack('I', struct.pack('f', float(x)))[0] + 0x8000) >> 16) & 0xffff for x in inputs]
    actual_bits = converted.view(torch.uint16).tolist()
    assert actual_bits == model_bits, (actual_bits, model_bits)
    assert converted[0] != inputs.bfloat16()[0]  # FBGEMM half-up vs Torch ties-even.
    table = lsh.LSH(); table.alloc(3, 4, 1, 4, 2, 2, 5)
    codes = torch.tensor([[[1, 1, 2], [1, 1, 3], [2, 1, 3], [2, 3, 1]]] * 2, dtype=torch.int32)
    table.fastfill(0, 1, codes)
    result = torch.full((8, 5), -1, dtype=torch.int32); nnz = torch.zeros(8, dtype=torch.int32)
    query = torch.ones((8, 4), dtype=torch.int32)
    table.batch_retrieve(0, query, result, nnz)
    assert nnz[:4].tolist() == [0]*4
    assert nnz[4:].tolist() == [2]*4
    for h in range(4, 8):
        assert set(result[h, :2].tolist()) == {0, 1}
    mask = table.get_mask(); mask.fill_(2); assert bool((table.get_mask() == 2).all())
    table.clear(); assert not mask.count_nonzero()
    table.fastfill(0, 1, codes)
    rejected(lambda: table.fastfill(0, 1, codes.to(torch.int64)))
    rejected(lambda: table.fastfill(0, 1, codes.transpose(1, 2)))
    return {"status": "pass", "bf16": bf16, "heads_checked": checked, "dispatch_traces": traces, "torch": torch.__version__, "cxx11_abi": torch.compiled_with_cxx11_abi(), "fbgemm_model_bit_mismatches": 0, "openmp_thread_limit": 4, "owner_policy": "direct getter keep_alive; derived aliases require owner lifetime or clone"}
