#!/usr/bin/env python3
"""TRACE_TEST_ID: MAGICPIG-CACHE-MERGE-CPU; selected v0.2 CPU cache/merge contract.

Execute the original Python AST statements using deliberately restricted host
substitutes. These checks do not run Torch, FlashInfer, device transfers or the
full extension. Normalizers are base-2 LSE. Numerical expectations are computed
from raw query/key/value tokens, independently of the host merge implementation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from test_magicpig_simhash_cpu import Tensor, TorchAdapter


SOURCE = 'third_party/magicpig/models/attnserver.py'
TEST_ID = 'MAGICPIG-CACHE-MERGE-CPU'


def target(node: ast.stmt) -> str:
    if isinstance(node, ast.Assign):
        return ast.unparse(node.targets[0])
    return ''


def select(source: str) -> dict[str, list[ast.stmt]]:
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LSHSparseAttnServer']
    assert len(classes) == 1
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    assert all(k in methods for k in ('fill', 'plan', 'decode', 'clear'))
    sparse = next(n for n in methods['fill'].body if isinstance(n, ast.If))
    decode = next(n for n in methods['decode'].body if isinstance(n, ast.If))
    partition = sparse.orelse[:10]
    assert [target(n) for n in partition] == [
        'sink_tokens_key', 'sink_tokens_value', 'local_tokens_key', 'local_tokens_value',
        'key', 'value', 'offload_key', 'offload_value', 'offload_key', 'offload_value',
    ]
    centering = sparse.orelse[10:14]
    assert [target(n) for n in centering] == ['avg_k', 'key', 'offload_key', 'kn']
    center_store = sparse.orelse[14:15]
    assert [target(n) for n in center_store] == ['self.avg_k[layer_idx][request_id]']
    sparse_cache = sparse.orelse[15:18]
    assert len(sparse_cache) == 3 and target(sparse_cache[-1]) == 'self.kv_last_page_len[request_id]'
    assert all('.copy_(' in ast.unparse(n) for n in sparse_cache[:2])
    dense_fill = sparse.body
    assert len(dense_fill) == 3 and target(dense_fill[-1]) == 'self.dense_kv_last_page_len[request_id]'
    append = [n for n in decode.orelse if target(n) == 'key_states'
              and 'self.avg_k[layer_idx]' in ast.unparse(n)]
    assert len(append) == 1 and 'self.avg_k[layer_idx]' in ast.unparse(append[0])
    merge = [n for n in decode.orelse if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
             and n.value.func.attr == 'merge_state']
    assert len(merge) == 1 and 'flashinfer.merge_state' in ast.unparse(merge[0])
    plan = methods['plan'].body
    assert len(plan) == 4 and sum(isinstance(n, ast.AugAssign) for n in plan) == 2
    clear = methods['clear'].body
    assert target(clear[0]) == '' and 'self.lsh_retriever.clear()' in ast.unparse(clear[-2])
    assert 'self.attn_server.clear()' in ast.unparse(clear[-1])
    return {'partition': partition, 'centering': centering, 'center_store': center_store,
            'sparse_cache': sparse_cache, 'dense_fill': dense_fill, 'append_centering': append,
            'plan': plan, 'merge': merge, 'clear': clear}


def run_selected(block: list[ast.stmt], env: dict[str, Any], **dependencies: Any) -> None:
    # No archived top-level imports, random projection, file IO or GPU functions.
    globals_ = {'__builtins__': {'range': range}, 'torch': TorchAdapter, **dependencies}
    exec(compile(ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])),
                 'selected-magicpig-attnserver', 'exec'), globals_, env)


class Slot:
    def __init__(self, heads: int, capacity: int, dim: int) -> None:
        self.value = Tensor([0.0] * (heads * capacity * dim), (heads, capacity, dim))

    def __getitem__(self, key: Any) -> SlotView:
        assert isinstance(key, tuple) and len(key) == 3 and key[0] is Ellipsis
        assert key[2] == slice(None) and isinstance(key[1], slice)
        start, stop, step = key[1].indices(self.value.shape[1])
        assert step == 1
        return SlotView(self, start, stop)

    def copy_(self, value: Tensor) -> None:
        SlotView(self, 0, self.value.shape[1]).copy_(value)

    def zero_(self) -> None:
        self.value.data = [0.0] * len(self.value.data)


class SlotView:
    def __init__(self, slot: Slot, start: int, stop: int) -> None:
        self.slot, self.start, self.stop = slot, start, stop

    def copy_(self, value: Tensor) -> None:
        heads, _, dim = self.slot.value.shape
        # Selected source copies have equal shapes. Broadcasting is outside this
        # adapter's contract; rejection fixtures use nonbroadcastable shapes.
        if value.shape != (heads, self.stop - self.start, dim):
            raise ValueError('cache copy shape mismatch')
        for h, t, d in itertools.product(range(heads), range(self.stop - self.start), range(dim)):
            offset = (h * self.slot.value.shape[1] + self.start + t) * dim + d
            self.slot.value.data[offset] = value.index((h, t, d))


class IndexedValues:
    def __init__(self, count: int) -> None:
        self.values = [0] * count

    def __setitem__(self, index: int, value: int) -> None:
        self.values[index] = value

    def __iadd__(self, increment: int) -> IndexedValues:
        self.values = [x + increment for x in self.values]
        return self

    def zero_(self) -> None:
        self.values = [0] * len(self.values)


class StoredMean:
    def __init__(self) -> None:
        self.values: dict[int, Tensor] = {}

    def __setitem__(self, request: int, mean: Tensor) -> None:
        self.values[request] = mean


class CountClear:
    def __init__(self) -> None:
        self.count = 0

    def clear(self) -> None:
        self.count += 1


class Zeroable:
    def __init__(self) -> None:
        self.data = [3.25, -7.0]

    def zero_(self) -> None:
        self.data = [0.0] * len(self.data)


class CapturePlan:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan(self, *args: Any, **kwargs: Any) -> None:
        if any(n < 1 or n > args[6] for n in args[2].values):
            raise ValueError('host page capacity exceeded')
        self.calls.append({'lengths': list(args[2].values), 'heads': (args[3], args[4]),
                           'dim': args[5], 'capacity': args[6], 'kwargs': kwargs})


def source_membership(blocks: dict[str, list[ast.stmt]]) -> tuple[int, int, float, int]:
    """Extract actual fill/centering/copy/append statements; check independent ID sets."""
    count = 0
    centered_checks = 0
    max_shift_error = 0.0
    short_unsupported = 0
    # Shortest supported context contains one offloaded token. Empty offload is
    # deliberately *not* represented as passing production support.
    for length, batch in itertools.product((3, 4, 5, 8), range(2)):
        heads, dim = 2, 2
        values = [float(t + 1) if d == 0 else float(-10 * (t + 1) - h)
                  for t in range(length) for h in range(heads) for d in range(dim)]
        keys = [float((t + 1) * (h + 2) + batch * (h + 1)) if d == 0
                else float((t * t + 1) / (h + 1) - batch * 3)
                for t in range(length) for h in range(heads) for d in range(dim)]
        key_cache = Tensor(keys, (length, heads, dim))
        value_cache = Tensor(values, (length, heads, dim))
        sparse_slots = [[Slot(heads, 4, dim), Slot(heads, 4, dim)] for _ in range(2)]
        obj = SimpleNamespace(num_sink_tokens=1, num_local_tokens=1,
                              avg_k=[StoredMean()], flashinfer_kv_cache=[sparse_slots],
                              kv_last_page_len=IndexedValues(2))
        env: dict[str, Any] = {'self': obj, 'key_cache': key_cache,
                               'value_cache': value_cache, 'seq_len': length,
                               'layer_idx': 0, 'request_id': batch}
        run_selected(blocks['partition'], env)
        run_selected(blocks['centering'], env)
        run_selected(blocks['center_store'], env)
        run_selected(blocks['sparse_cache'], env)
        assert obj.kv_last_page_len.values[batch] == 2
        assert obj.flashinfer_kv_cache[0][batch][0].value is not None
        assert obj.flashinfer_kv_cache[0][batch][1].value is not None
        assert obj.flashinfer_kv_cache[0][batch][0].value[:, :2, :].data == env['key'].data
        assert obj.flashinfer_kv_cache[0][batch][1].value[:, :2, :].data == env['value'].data
        assert obj.flashinfer_kv_cache[0][1 - batch][0].value.data == [0.0] * 16
        assert obj.avg_k[0].values[batch].data == env['avg_k'].data
        for head in range(heads):
            static = [int(env['value'].index((head, i, 0))) - 1 for i in range(2)]
            offload = [int(env['offload_value'].index((head, i, 0))) - 1
                       for i in range(length - 2)]
            appended = [length]
            assert static == [0, length - 1]
            assert offload == list(range(1, length - 1))
            assert len(set(static + offload + appended)) == length + 1
            assert sorted(static + offload + appended) == list(range(length + 1))
            mean = [sum(key_cache.index((t, head, d)) for t in offload) / len(offload)
                    for d in range(dim)]
            for d in range(dim):
                assert abs(env['avg_k'].index((head, 0, d)) - mean[d]) < 1e-12
                for pos, token in enumerate(static):
                    assert abs(env['key'].index((head, pos, d)) -
                               (key_cache.index((token, head, d)) - mean[d])) < 1e-12
                for pos, token in enumerate(offload):
                    assert abs(env['offload_key'].index((head, pos, d)) -
                               (key_cache.index((token, head, d)) - mean[d])) < 1e-12
                centered_checks += len(static) + len(offload)
            # A common key translation must not change any attention probability.
            q = [0.125 * (head + batch + 1), -0.375]
            extra = [12 + batch + head, -3 + batch]
            raw = [[key_cache.index((t, head, d)) for d in range(dim)]
                   for t in range(length)] + [extra]
            shifted = [[k[d] - mean[d] for d in range(dim)] for k in raw]
            weights = lambda inputs: [math.exp(s - max(sum(x * y for x, y in zip(q, k))
                                      for k in inputs)) for s in
                                      [sum(x * y for x, y in zip(q, k)) for k in inputs]]
            a, b = weights(raw), weights(shifted)
            for x, y in zip(a, b):
                max_shift_error = max(max_shift_error, abs(x / sum(a) - y / sum(b)))
            count += 1
    # Actual decode key append uses the fixed per-request offload mean.
    for length in (3, 5):
        all_means = [float(b * 7 + h * 2 + d) for b in range(2) for h in range(2) for d in range(2)]
        means = Tensor(all_means, (2, 2, 1, 2))
        obj = SimpleNamespace(avg_k=[means])
        states = Tensor([float(20 + b * 10 + h * 3 + d + length)
                         for b in range(2) for h in range(2) for d in range(2)], (2, 2, 1, 2))
        env = {'self': obj, 'layer_idx': 0, 'key_states': states}
        run_selected(blocks['append_centering'], env)
        assert env['key_states'].data == [x - y for x, y in zip(states.data, all_means)]
        centered_checks += len(states.data)
    # The archived fill computes mean of an empty offload for seq_len <= sink + local.
    # Characterization is a bounded negative test, not a correction or supported input.
    for length in (1, 2):
        obj = SimpleNamespace(num_sink_tokens=1, num_local_tokens=1)
        env = {'self': obj, 'seq_len': length,
               'key_cache': Tensor([1.0, 2.0] * length, (length, 1, 2)),
               'value_cache': Tensor([1.0, 2.0] * length, (length, 1, 2))}
        run_selected(blocks['partition'], env)
        assert env['offload_key'].shape[1] == 0
        try:
            run_selected(blocks['centering'], env)
        except ZeroDivisionError:
            # Restricted host mean rejects an empty reduction; PyTorch may
            # produce NaNs instead. This does not assert a Torch exception.
            short_unsupported += 1
        else:
            raise AssertionError('host adapter silently accepted empty offload mean')
    assert count == 16 and centered_checks > 50 and max_shift_error < 1e-12
    assert short_unsupported == 2
    return count, centered_checks, max_shift_error, short_unsupported


def dense_and_plan(blocks: dict[str, list[ast.stmt]]) -> tuple[int, int]:
    dense_cases = 0
    for length in (2, 3, 7):
        capacity = 8
        slots = [[Slot(2, capacity, 2), Slot(2, capacity, 2)]]
        obj = SimpleNamespace(dense_layers=[0], flashinfer_kv_cache=[slots],
                              dense_kv_last_page_len=IndexedValues(1))
        # llama.py passes its full max_length buffers, including unused tail.
        key = Tensor([float(x) for x in range(length * 4)] + [0.0] * ((capacity - length) * 4),
                     (capacity, 2, 2))
        value = Tensor([float(100 + x) for x in range(length * 4)] + [0.0] * ((capacity - length) * 4),
                       (capacity, 2, 2))
        run_selected(blocks['dense_fill'], {'self': obj, 'layer_idx': 0, 'request_id': 0,
                                            'key_cache': key, 'value_cache': value,
                                            'seq_len': length})
        assert slots[0][0].value.data == key.transpose(0, 1).data
        assert slots[0][1].value.data == value.transpose(0, 1).data
        assert obj.dense_kv_last_page_len.values == [length]
        dense_cases += 1
    plan_calls = 0
    for batch in (1, 2):
        obj = SimpleNamespace(
            kv_last_page_len=IndexedValues(batch), dense_kv_last_page_len=IndexedValues(batch),
            decode_wrapper=CapturePlan(), dense_decode_wrapper=CapturePlan(),
            kv_page_indptr=[0] * (batch + 1), kv_page_indices=list(range(batch)),
            dense_kv_page_indptr=[0] * (batch + 1), dense_kv_page_indices=list(range(batch)),
            num_attention_heads=8, num_key_value_heads=2, head_dim=128,
            page_size=16, dense_page_size=64)
        obj.kv_last_page_len.values = [2 + i for i in range(batch)]
        obj.dense_kv_last_page_len.values = [5 + i for i in range(batch)]
        for step in range(2):
            run_selected(blocks['plan'], {'self': obj}, torch=SimpleNamespace(bfloat16='bfloat16'))
            assert obj.decode_wrapper.calls[-1]['lengths'] == [3 + step + i for i in range(batch)]
            assert obj.dense_decode_wrapper.calls[-1]['lengths'] == [6 + step + i for i in range(batch)]
            for c in (obj.decode_wrapper.calls[-1], obj.dense_decode_wrapper.calls[-1]):
                assert c['heads'] == (8, 2) and c['dim'] == 128
                assert c['kwargs']['q_data_type'] == 'bfloat16'
            plan_calls += 2
        # Bounds rejected by CapturePlan, not by the archived plan().
        assert all(n <= obj.page_size for n in obj.kv_last_page_len.values)
        assert all(n <= obj.dense_page_size for n in obj.dense_kv_last_page_len.values)
    return dense_cases, plan_calls


def fp64_attention(entries: list[tuple[float, tuple[float, ...]]]) -> tuple[tuple[float, ...], float]:
    assert entries
    peak = max(s for s, _ in entries)
    weights = [math.exp(s - peak) for s, _ in entries]
    denom = sum(weights)
    result = tuple(sum(w * v[d] for w, (_, v) in zip(weights, entries)) / denom
                   for d in range(len(entries[0][1])))
    return result, (peak + math.log(denom)) / math.log(2.0)


def host_merge(a: list[tuple[float, ...]], alse: list[float],
               b: list[tuple[float, ...]], blse: list[float]) -> tuple[list[tuple[float, ...]], list[float]]:
    merged: list[tuple[float, ...]] = []
    lse: list[float] = []
    for x, p, y, q in zip(a, alse, b, blse):
        peak = max(p, q)
        wa, wb = 2.0 ** (p - peak), 2.0 ** (q - peak)
        merged.append(tuple((wa * u + wb * v) / (wa + wb) for u, v in zip(x, y)))
        lse.append(peak + math.log2(wa + wb))
    assert len(merged) == len(a) == len(b)
    return merged, lse


def merge_checks(blocks: dict[str, list[ast.stmt]]) -> tuple[int, int, float, int]:
    checks = 0
    dense_checks = 0
    max_error = 0.0
    sensitivity = {'omit_correction': False, 'natural_lse': False, 'stale_centers': False,
                   'uniform_probability': False}
    sampled_probabilities: set[float] = set()
    heterogeneous_checks = 0
    for group, batch, length, sampled in itertools.product((1, 4, 8), range(2), (3, 5, 8), (False, True)):
        head_results: list[tuple[tuple[float, ...], float, tuple[float, ...], float,
                                 tuple[float, ...], float]] = []
        for qhead in range(2 * group):
            kvhead = qhead // group
            query = (0.125 * (qhead + 1), -0.4 + 0.1 * batch)
            keys = [(0.31 * (t + 1) * (kvhead + 1) + 0.75 * batch,
                     (-1.0) ** t * (0.5 * t + kvhead)) for t in range(length + 1)]
            values = [((t + 1) * 0.2 + 0.7 * kvhead, (-1) ** t * (0.3 * t + batch))
                      for t in range(length + 1)]
            static = [0, length - 1, length]
            offloaded = list(range(1, length - 1))
            selected = offloaded[::2] if sampled else offloaded
            # Alternate over selected positions: sampled IDs are all odd, so
            # token-ID parity would silently omit the 0.65 probability cases.
            probabilities = {t: (0.4 if i % 2 == 0 else 0.65) if sampled else 1.0
                             for i, t in enumerate(selected)}
            if sampled:
                sampled_probabilities.update(probabilities.values())
            assert set(static).isdisjoint(selected)
            assert set(static + offloaded) == set(range(length + 1))
            mean = tuple(sum(keys[t][d] for t in offloaded) / len(offloaded) for d in range(2))
            def row(t: int, p: float, shift: bool = True) -> tuple[float, tuple[float, ...]]:
                k = [keys[t][d] - (mean[d] if shift else 0.0) for d in range(2)]
                score = sum(q * x for q, x in zip(query, k)) / math.sqrt(2.0) - math.log(p)
                return score, values[t]
            gpu_rows = [row(t, 1.0) for t in static]
            cpu_rows = [row(t, probabilities[t]) for t in selected]
            gpu, glse = fp64_attention(gpu_rows)
            cpu, clse = fp64_attention(cpu_rows)
            expected, el = fp64_attention(gpu_rows + cpu_rows)
            head_results.append((gpu, glse, cpu, clse, expected, el))
            if not sampled:
                dense, dl = fp64_attention([row(t, 1.0) for t in range(length + 1)])
                assert max(abs(a - b) for a, b in zip(expected, dense)) < 1e-12
                assert abs(el - dl) < 1e-12
                dense_checks += 1
            else:
                bad, _ = fp64_attention(gpu_rows + [row(t, 1.0) for t in selected])
                sensitivity['omit_correction'] |= max(abs(x - y) for x, y in zip(bad, expected)) > 1e-4
                # Different inclusion probabilities must also affect the
                # within-offload distribution, not just its total merge mass.
                uniform_cpu, _ = fp64_attention([row(t, 0.4) for t in selected])
                different = max(abs(x - y) for x, y in zip(uniform_cpu, cpu)) > 1e-4
                if len(set(probabilities.values())) > 1:
                    assert different, 'heterogeneous probabilities did not reweight offload output'
                    heterogeneous_checks += 1
                sensitivity['uniform_probability'] |= different
                # Shift only the GPU-side keys (stale/offload centering mismatch).
                bad_gpu = [row(t, 1.0, shift=False) for t in static]
                bad_center, _ = fp64_attention(bad_gpu + cpu_rows)
                sensitivity['stale_centers'] |= max(abs(x - y) for x, y in zip(bad_center, expected)) > 1e-4
        gpu = [h[0] for h in head_results]
        glse = [h[1] for h in head_results]
        cpu = [h[2] for h in head_results]
        clse = [h[3] for h in head_results]
        env: dict[str, Any] = {'gpu_hidden_states': gpu, 'gpu_lse': glse,
                               'cpu_hidden_states': cpu, 'cpu_lse': clse}
        run_selected(blocks['merge'], env, flashinfer=SimpleNamespace(merge_state=host_merge))
        merged, mlse = env['hidden_states'], env['_']
        assert len(merged) == len(head_results) == 2 * group
        for index, head in enumerate(head_results):
            err = max(max(abs(x-y) for x, y in zip(merged[index], head[4])), abs(mlse[index]-head[5]))
            max_error = max(max_error, err)
            assert err < 3e-12
            checks += 1
        # Feeding natural-log values to a base-2 merge must be observable.
        bad, _ = host_merge(gpu, [x * math.log(2) for x in glse], cpu, clse)
        sensitivity['natural_lse'] |= any(max(abs(x-y) for x, y in zip(a,b)) > 1e-4
                                          for a,b in zip(bad, merged))
    assert all(sensitivity.values()), sensitivity
    assert sampled_probabilities == {0.4, 0.65}, sampled_probabilities
    assert checks == 2 * sum((1,4,8)) * 2 * 3 * 2 and dense_checks == checks // 2
    assert heterogeneous_checks == 104, heterogeneous_checks
    return checks, dense_checks, max_error, heterogeneous_checks


def clear_checks(blocks: dict[str, list[ast.stmt]]) -> int:
    resets = 0
    for layers, batch in itertools.product((1, 2), (1, 2)):
        buffers = {k: Zeroable() for k in ('nnz', 'results_lsh_cpu', 'max_value_expsum',
                   'output_cuda', 'max_value_expsum_cuda', 'output', 'pinned_hashcode', 'pinned_query')}
        obj = SimpleNamespace(**buffers, num_layers=layers,
                              avg_k=[Zeroable() for _ in range(layers)],
                              flashinfer_kv_cache=[Zeroable() for _ in range(layers)],
                              kv_last_page_len=IndexedValues(batch),
                              dense_kv_last_page_len=IndexedValues(batch),
                              lsh_retriever=CountClear(), attn_server=CountClear())
        obj.kv_last_page_len.values = [2] * batch
        obj.dense_kv_last_page_len.values = [3] * batch
        run_selected(blocks['clear'], {'self': obj})
        for buf in (*buffers.values(), *obj.avg_k, *obj.flashinfer_kv_cache):
            assert buf.data == [0.0, 0.0]
            resets += 1
        assert obj.kv_last_page_len.values == [0] * batch
        assert obj.dense_kv_last_page_len.values == [0] * batch
        assert obj.lsh_retriever.count == obj.attn_server.count == 1
        # Refilled state can be reused, and subsequent clear must reset again.
        obj.kv_last_page_len.values = [4] * batch
        obj.nnz.data[0] = 9
        run_selected(blocks['clear'], {'self': obj})
        assert obj.kv_last_page_len.values == [0] * batch and obj.nnz.data[0] == 0
        assert obj.lsh_retriever.count == obj.attn_server.count == 2
    return resets


def bounded_state_checks(blocks: dict[str, list[ast.stmt]]) -> tuple[int, int, int, int]:
    capacity_checks = 0
    copy_rejections = 0
    for sparse, dense in ((3, 6), (4, 7), (4, 6), (3, 7)):
        obj = SimpleNamespace(
            kv_last_page_len=IndexedValues(1), dense_kv_last_page_len=IndexedValues(1),
            decode_wrapper=CapturePlan(), dense_decode_wrapper=CapturePlan(),
            kv_page_indptr=[0, 1], kv_page_indices=[0],
            dense_kv_page_indptr=[0, 1], dense_kv_page_indices=[0],
            num_attention_heads=2, num_key_value_heads=2, head_dim=2,
            page_size=4, dense_page_size=7)
        obj.kv_last_page_len.values = [sparse]
        obj.dense_kv_last_page_len.values = [dense]
        try:
            run_selected(blocks['plan'], {'self': obj}, torch=SimpleNamespace(bfloat16='bfloat16'))
        except ValueError as error:
            assert str(error) == 'host page capacity exceeded' and (sparse == 4 or dense == 7)
        else:
            assert (sparse, dense) == (3, 6)
            assert obj.kv_last_page_len.values == [4] and obj.dense_kv_last_page_len.values == [7]
        capacity_checks += 1
    for sparse in (False, True):
        for capacity in ((2, 5) if sparse else (3, 8)):
            slots = [[Slot(2, capacity, 2), Slot(2, capacity, 2)]]
            obj = SimpleNamespace(flashinfer_kv_cache=[slots], num_sink_tokens=1, num_local_tokens=1,
                                  kv_last_page_len=IndexedValues(1), dense_kv_last_page_len=IndexedValues(1))
            env = {'self': obj, 'layer_idx': 0, 'request_id': 0, 'seq_len': 2,
                   'key': Tensor([1.0] * 12, (2, 3, 2)),
                   'value': Tensor([2.0] * 12, (2, 3, 2)),
                   'key_cache': Tensor([1.0] * 8, (2, 2, 2)),
                   'value_cache': Tensor([2.0] * 8, (2, 2, 2))}
            try:
                run_selected(blocks['sparse_cache' if sparse else 'dense_fill'], env)
            except ValueError as error:
                assert str(error) == 'cache copy shape mismatch'
                assert slots[0][0].value.data == [0.0] * (4 * capacity)
                copy_rejections += 1
            else:
                raise AssertionError('nonbroadcastable cache copy accepted')

    class MeanTensor(Tensor):
        def __setitem__(self, request: int, mean: Tensor) -> None:
            assert mean.shape == self.shape[1:]
            width = math.prod(mean.shape)
            self.data[request * width:(request + 1) * width] = mean.data

        def zero_(self) -> None:
            self.data = [0.0] * len(self.data)

    class CacheLayer(list):
        def zero_(self) -> None:
            for slots in self:
                for slot in slots:
                    slot.zero_()

    buffers = {k: Zeroable() for k in ('nnz', 'results_lsh_cpu', 'max_value_expsum',
               'output_cuda', 'max_value_expsum_cuda', 'output', 'pinned_hashcode', 'pinned_query')}
    obj = SimpleNamespace(
        **buffers, num_layers=2, num_sink_tokens=1, num_local_tokens=1,
        avg_k=[MeanTensor([0.0] * 8, (2, 2, 1, 2)) for _ in range(2)],
        flashinfer_kv_cache=[CacheLayer([[Slot(2, capacity, 2), Slot(2, capacity, 2)]
                            for _ in range(2)]) for capacity in (4, 8)],
        kv_last_page_len=IndexedValues(2), dense_kv_last_page_len=IndexedValues(2),
        lsh_retriever=CountClear(), attn_server=CountClear(),
        decode_wrapper=CapturePlan(), dense_decode_wrapper=CapturePlan(),
        kv_page_indptr=[0, 1, 2], kv_page_indices=[0, 1],
        dense_kv_page_indptr=[0, 1, 2], dense_kv_page_indices=[0, 1],
        num_attention_heads=2, num_key_value_heads=2, head_dim=2,
        page_size=4, dense_page_size=8)
    steps = 0
    for cycle in range(2):
        lengths = [3 + cycle, 5 + cycle]
        expected_means = []
        for request, length in enumerate(lengths):
            key = Tensor([float(100 * cycle + 20 * request + t * 3 + h + d)
                          if t < length else 0.0
                          for t in range(8) for h in range(2) for d in range(2)], (8, 2, 2))
            value = Tensor([float(10 * cycle + 5 * request + t + h + d)
                            if t < length else 0.0
                            for t in range(8) for h in range(2) for d in range(2)], (8, 2, 2))
            env = {'self': obj, 'layer_idx': 0, 'request_id': request,
                   'seq_len': length, 'key_cache': key, 'value_cache': value}
            for name in ('partition', 'centering', 'center_store', 'sparse_cache'):
                run_selected(blocks[name], env)
            expected_means.extend([sum(key.index((t, h, d)) for t in range(1, length - 1)) / (length - 2)
                                   for h in range(2) for d in range(2)])
            run_selected(blocks['dense_fill'], {**env, 'layer_idx': 1})
            assert obj.flashinfer_kv_cache[1][request][0].value.data == key.transpose(0, 1).data
            assert obj.flashinfer_kv_cache[1][request][1].value.data == value.transpose(0, 1).data
            # Sparse layer must only copy its two resident tokens, leaving append space untouched.
            for slot, name in zip(obj.flashinfer_kv_cache[0][request], ('key', 'value')):
                assert slot.value[:, :2, :].data == env[name].data
                assert slot.value[:, 2:, :].data == [0.0] * 8
        assert obj.avg_k[0].data == expected_means
        assert obj.avg_k[1].data == [0.0] * 8
        for step in range(2):
            run_selected(blocks['plan'], {'self': obj}, torch=SimpleNamespace(bfloat16='bfloat16'))
            assert obj.kv_last_page_len.values == [3 + step] * 2
            assert obj.dense_kv_last_page_len.values == [n + step + 1 for n in lengths]
            raw = [float(200 + 100 * cycle + 20 * request + 4 * step + h + d)
                   for request in range(2) for h in range(2) for d in range(2)]
            new_values = Tensor([-x for x in raw], (2, 2, 1, 2))
            env = {'self': obj, 'layer_idx': 0, 'key_states': Tensor(raw, (2, 2, 1, 2))}
            run_selected(blocks['append_centering'], env)
            assert env['key_states'].data == [x - mean for x, mean in zip(raw, expected_means)]
            # A host append surrogate checks page offsets and ownership. The
            # FlashInfer append call and GPU scheduling remain unexecuted.
            for request in range(2):
                for layer, lengths_state in ((0, obj.kv_last_page_len), (1, obj.dense_kv_last_page_len)):
                    position = lengths_state.values[request] - 1
                    new_keys = env['key_states'] if layer == 0 else Tensor(raw, (2, 2, 1, 2))
                    for slot, data in zip(obj.flashinfer_kv_cache[layer][request],
                                          (new_keys[request], new_values[request])):
                        before = list(slot.value.data)
                        slot[..., position:position + 1, :].copy_(data)
                        for h in range(2):
                            for t in range(slot.value.shape[1]):
                                for d in range(2):
                                    index = (h * slot.value.shape[1] + t) * 2 + d
                                    assert slot.value.data[index] == (data.index((h, 0, d)) if t == position else before[index])
            assert obj.avg_k[0].data == expected_means
            steps += 1
        run_selected(blocks['clear'], {'self': obj})
        assert obj.kv_last_page_len.values == obj.dense_kv_last_page_len.values == [0, 0]
        assert obj.lsh_retriever.count == obj.attn_server.count == cycle + 1
        assert all(x == 0.0 for mean in obj.avg_k for x in mean.data)
        assert all(x == 0.0 for layer in obj.flashinfer_kv_cache for slots in layer
                   for slot in slots for x in slot.value.data)
        assert all(buf.data == [0.0, 0.0] for buf in buffers.values())
    assert capacity_checks == copy_rejections == steps == 4
    return capacity_checks, copy_rejections, steps, 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    path = args.root / SOURCE
    source = path.read_text()
    blocks = select(source)
    membership, centering, shift_error, short_unsupported = source_membership(blocks)
    dense, plans = dense_and_plan(blocks)
    merged, full_dense, merge_error, heterogeneous = merge_checks(blocks)
    resets = clear_checks(blocks)
    capacity, copy_rejections, lifecycle_steps, reuse_cycles = bounded_state_checks(blocks)
    statements = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.stmt)]
    hashes = {}
    for name, nodes in blocks.items():
        start, end = nodes[0].lineno, nodes[-1].end_lineno
        selected = sorted((n for n in statements if start <= n.lineno and n.end_lineno <= end),
                          key=lambda n: n.lineno)
        extracted = '\n'.join(ast.get_source_segment(source, n) or '' for n in selected)
        hashes[name] = {'start_line': start, 'end_line': end,
                        'sha256': hashlib.sha256(extracted.encode()).hexdigest()}
    result = {
        'id': TEST_ID, 'name': 'cache_merge', 'status': 'passed', 'source': SOURCE,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'selected_blocks': hashes, 'partition_cases': membership, 'centered_coordinates': centering,
        'centering_probability_max_error': shift_error, 'short_context_unsupported': short_unsupported,
        'dense_fill_cases': dense, 'plan_calls': plans, 'merge_head_checks': merged,
        'full_selection_dense_checks': full_dense, 'merge_fp64_max_error': merge_error,
        'clear_buffer_checks': resets, 'gpu_execution': False, 'torch_runtime': False,
        'capacity_boundary_checks': capacity, 'copy_shape_rejections': copy_rejections,
        'lifecycle_decode_steps': lifecycle_steps, 'lifecycle_reuse_cycles': reuse_cycles,
        'sampled_inclusion_probabilities': [0.4, 0.65],
        'heterogeneous_sampled_head_checks': heterogeneous,
        'flashinfer_execution': False, 'host_adapter': 'restricted CPU selected AST; FP64 list tensor and base-2 merger',
        'coverage_limit': 'nonempty offload; host copy/page guards and append surrogate; native capacity and FlashInfer append unexecuted',
    }
    print(json.dumps(result, sort_keys=True))
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
