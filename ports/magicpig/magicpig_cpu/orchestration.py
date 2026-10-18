"""Backend-independent complete CPU server control derived from archived attnserver.

TRACE_TEST_ID: MAGICPIG-SERVER-CPU. See SERVER_CPU_CONTRACT.md for changes,
source hashes, native/provided backend boundaries and ownership policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor


class AttentionBackend(Protocol):
    """Synchronous native CPU primitive API; caller owns all Tensor arguments."""
    def alloc(self, layers: int, heads: int, kvheads: int, dim: int, batch: int, capacity: int) -> None: ...
    def fill(self, layer: int, request: int, key: Tensor, value: Tensor, norm: Tensor) -> None: ...
    def attention_wrapper(self, layer: int, k: int, tables: int, output: Tensor, metadata: Tensor, query: Tensor, norm: Tensor, indices: Tensor, nnz: Tensor) -> None: ...
    def full_attention(self, layer: int, output: Tensor, metadata: Tensor, query: Tensor, nnz: Tensor) -> None: ...
    def clear(self) -> None: ...


class RetrievalBackend(Protocol):
    """Synchronous threshold-two table API; payload IDs refer to offload rows."""
    def alloc(self, k: int, tables: int, layers: int, heads: int, kvheads: int, batch: int, capacity: int) -> None: ...
    def fill(self, layer: int, request: int, codes: Tensor, indices: Tensor) -> None: ...
    def batch_retrieve(self, layer: int, query: Tensor, results: Tensor, nnz: Tensor) -> None: ...
    def clear(self) -> None: ...


@dataclass(frozen=True)
class ServerConfig:
    """Small explicit server capacity/configuration; no model loading or device IO."""
    layers: int
    heads: int
    kvheads: int
    dim: int
    batch: int
    capacity: int
    sink: int = 1
    local: int = 1
    generation: int = 2
    chunk: int = 2
    dense_layers: tuple[int, ...] = (0,)

    def validate(self) -> None:
        """Reject unsupported config before native allocation."""
        if min(self.layers, self.heads, self.kvheads, self.dim, self.batch, self.capacity, self.chunk) <= 0 or self.heads % self.kvheads or self.dim % 32:
            raise ValueError("invalid positive dimensions/head groups")
        if min(self.sink, self.local, self.generation) < 0 or not self.generation or self.sink + self.local >= self.capacity:
            raise ValueError("invalid resident/generation capacity")
        if len(set(self.dense_layers)) != len(self.dense_layers) or any(layer < 0 or layer >= self.layers for layer in self.dense_layers):
            raise ValueError("invalid dense layers")


def cpu_bf16(value: Tensor, shape: tuple[int, ...]) -> None:
    """Reject dtype/device/layout/shape and nonfinite operands before mutation."""
    if value.device.type != "cpu" or value.dtype != torch.bfloat16 or not value.is_contiguous() or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError("expected finite contiguous CPU BF16 tensor of configured shape")


class CPUAttentionServer:
    """Complete shared LSH/baseline orchestrator with real Torch resident math.

    K=0 selects baseline uncorrected full offload attention. K1..11 selects
    sampled offload attention; explicit projection avoids RNG/model dependencies.
    Constructor, fill outer chunk loop, sort/build, plan, decode and reset execute
    in full. This is maintained CPU control, not execution of archived CUDA code.
    """
    def __init__(self, config: ServerConfig, attention: AttentionBackend, retrieval: RetrievalBackend | None = None, *, k: int = 0, tables: int = 4, projection: Tensor | None = None) -> None:
        config.validate()
        if k < 0 or k > 11 or tables < 2:
            raise ValueError("K must be0..11 and tables>=2")
        if k and (retrieval is None or projection is None):
            raise ValueError("sampled backend/projection required")
        if not k and (config.dim != 128 or config.heads // config.kvheads not in (1, 4, 8)):
            raise ValueError("native baseline requires dim128/groups1,4,8")
        if projection is not None:
            cpu_bf16(projection, (config.dim, k * tables))
        self.config = config
        self.k = k
        self.tables = tables
        self.attention = attention
        self.retrieval = retrieval
        self.projection = projection.clone() if projection is not None else None
        self.attention.alloc(config.layers, config.heads, config.kvheads, config.dim, config.batch, config.capacity)
        if k:
            assert retrieval is not None
            retrieval.alloc(k, tables, config.layers, config.heads, config.kvheads, config.batch, config.capacity)
        self.resident_capacity = config.sink + config.local + config.generation
        self.resident = [torch.zeros((config.batch, 2, config.kvheads, config.capacity if layer in config.dense_layers else self.resident_capacity, config.dim), dtype=torch.bfloat16) for layer in range(config.layers)]
        self.average = [torch.zeros((config.batch, config.kvheads, 1, config.dim), dtype=torch.bfloat16) for _ in range(config.layers)]
        self.lengths = [[0] * config.batch for _ in range(config.layers)]
        self.offload_lengths = [[0] * config.batch for _ in range(config.layers)]
        self.filled: set[tuple[int, int]] = set()
        self.built: set[tuple[int, int]] = set()
        self.pending: set[int] = set()
        self.chunk_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.hashes: dict[tuple[int, int], Tensor] = {}
        self.sorted_payloads: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        self.alloc_buffer()
        self.last_branch = "unrun"
        self.last_resident_lse = torch.empty(0)
        self.last_offload_lse = torch.empty(0)

    def alloc_buffer(self) -> None:
        """Allocate owned contiguous CPU call buffers once; no pinned/CUDA storage."""
        if hasattr(self, "nnz"):
            raise ValueError("call buffers allocate once per owner")
        c = self.config
        self.nnz = torch.zeros(c.batch * c.heads, dtype=torch.int32)
        self.results = torch.zeros((c.batch * c.heads, c.capacity), dtype=torch.int32)
        self.output = torch.zeros((c.batch * c.heads, c.dim), dtype=torch.bfloat16)
        self.metadata = torch.zeros((2, c.batch * c.heads), dtype=torch.float32)

    def _route(self, layer: int, request: int) -> None:
        """Validate layer/request IDs without Python negative-index wraparound."""
        if layer < 0 or layer >= self.config.layers or request < 0 or request >= self.config.batch:
            raise ValueError("invalid layer/request")

    def _hash(self, operand: Tensor) -> Tensor:
        """Actual Torch BF16 projection, strict >0 ties and exact K1..11 packing."""
        assert self.projection is not None
        projected = operand @ self.projection
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("nonfinite hash projection products")
        signs = projected.gt(0).reshape(*operand.shape[:-1], self.tables, self.k)
        packing = torch.tensor([2 ** i for i in range(self.k)], dtype=torch.float16)
        return (signs.to(torch.float16) * packing).sum(-1).to(torch.int16)

    def fill(self, layer: int, request: int, key: Tensor, value: Tensor) -> None:
        """Partition prompt, center sparse keys, execute every nonempty hash chunk."""
        self._route(layer, request)
        c = self.config
        if self.pending or (layer, request) in self.filled:
            raise ValueError("clear required before prompt refill")
        length = key.shape[0] if key.ndim == 3 else -1
        cpu_bf16(key, (length, c.kvheads, c.dim)); cpu_bf16(value, (length, c.kvheads, c.dim))
        if length <= 0 or length >= c.capacity:
            raise ValueError("prompt must fit with generation room")
        if layer in c.dense_layers:
            self.resident[layer][request, 0, :, :length].copy_(key.transpose(0, 1))
            self.resident[layer][request, 1, :, :length].copy_(value.transpose(0, 1))
            self.lengths[layer][request] = length
            self.built.add((layer, request))
        else:
            end = length - c.local
            if end <= c.sink:
                raise ValueError("empty offload is unsupported; use dense layer")
            offkey = key[c.sink:end].transpose(0, 1).contiguous()
            offvalue = value[c.sink:end].transpose(0, 1).contiguous()
            average = offkey.mean(1, keepdim=True)
            offkey = (offkey - average).contiguous()
            norms = offkey.float().norm(dim=-1)
            if not bool(torch.isfinite(average).all()) or not bool(torch.isfinite(offkey).all()) or not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
                raise ValueError("nonfinite or zero-norm centered offload unsupported")
            resident_key = torch.cat((key[:c.sink], key[end:]), dim=0).transpose(0, 1) - average
            resident_value = torch.cat((value[:c.sink], value[end:]), dim=0).transpose(0, 1)
            if not bool(torch.isfinite(resident_key).all()):
                raise ValueError("nonfinite centered resident keys")
            ranges: list[tuple[int, int]] = []
            hashes = torch.empty((c.kvheads, self.tables, end-c.sink), dtype=torch.int16)
            if self.k:
                for start in range(0, offkey.shape[1], c.chunk):
                    stop = min(start+c.chunk, offkey.shape[1])
                    hashes[:, :, start:stop] = self._hash(offkey[:, start:stop]).transpose(1, 2)
                    ranges.append((start, stop))
            self.attention.fill(layer, request, offkey, offvalue, norms)
            self.average[layer][request].copy_(average)
            nresident = c.sink + c.local
            self.resident[layer][request, 0, :, :nresident].copy_(resident_key)
            self.resident[layer][request, 1, :, :nresident].copy_(resident_value)
            self.lengths[layer][request] = nresident
            self.offload_lengths[layer][request] = offkey.shape[1]
            self.chunk_ranges[layer, request] = ranges
            if self.k:
                self.hashes[layer, request] = hashes
            else:
                self.built.add((layer, request))
        self.filled.add((layer, request))

    def build_table(self, layer: int, request: int) -> None:
        """Sort each payload together with its codes; no stable tie order required."""
        self._route(layer, request)
        if (layer, request) not in self.filled or self.pending:
            raise ValueError("fill required before table construction")
        if self.k and layer not in self.config.dense_layers:
            codes, ids = self.hashes[layer, request].sort(dim=-1)
            indices = ids.to(torch.int32).contiguous()
            assert self.retrieval is not None
            self.retrieval.fill(layer, request, codes.contiguous(), indices)
            self.sorted_payloads[layer, request] = (codes.clone(), indices.clone())
        self.built.add((layer, request))

    def plan(self) -> None:
        """Reserve one decode per layer after all requests/tables are prepared."""
        c = self.config
        if self.pending or len(self.built) != c.layers * c.batch:
            raise ValueError("all layers/requests must be ready and previous step complete")
        for layer in range(c.layers):
            capacity = self.resident[layer].shape[-2]
            if any(length >= capacity for length in self.lengths[layer]):
                raise ValueError("generation capacity exhausted")
        self.pending = set(range(c.layers))

    def _resident_attention(self, layer: int, query: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        """Actual CPU Torch attention/LSE; replaces FlashInfer kernel explicitly."""
        c = self.config
        outputs: list[Tensor] = []; lses: list[Tensor] = []
        for request in range(c.batch):
            n = self.lengths[layer][request] + 1
            heads = torch.arange(c.heads) // (c.heads // c.kvheads)
            keys = cache[request, 0, heads, :n].float()
            values = cache[request, 1, heads, :n].float()
            logits = torch.einsum("hd,hnd->hn", query[request].float(), keys) / math.sqrt(c.dim)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite resident logits")
            outputs.append(torch.einsum("hn,hnd->hd", logits.softmax(-1), values))
            lses.append(logits.logsumexp(-1) / math.log(2))
        return torch.stack(outputs), torch.stack(lses)

    def decode(self, query: Tensor, key: Tensor, value: Tensor, layer: int) -> Tensor:
        """Execute dense or native sampled/full offload plus resident union merge."""
        self._route(layer, 0)
        c = self.config
        if layer not in self.pending:
            raise ValueError("plan required once per generation step/layer")
        cpu_bf16(query, (c.batch, c.heads, 1, c.dim))
        cpu_bf16(key, (c.batch, c.kvheads, 1, c.dim)); cpu_bf16(value, (c.batch, c.kvheads, 1, c.dim))
        flat = query.reshape(c.batch*c.heads, c.dim)
        query_norms = flat.float().norm(dim=-1)
        if not bool(torch.isfinite(query_norms).all()) or not bool((query_norms > 0).all()):
            raise ValueError("nonfinite or zero query norm unsupported")
        dense = layer in c.dense_layers
        if not dense:
            if self.k:
                assert self.retrieval is not None
                normalized = flat / flat.norm(dim=-1, keepdim=True)
                hashes = self._hash(normalized).to(torch.int32).contiguous()
                self.retrieval.batch_retrieve(layer, hashes, self.results, self.nnz)
                self.attention.attention_wrapper(layer, self.k, self.tables, self.output, self.metadata, flat, flat.float().norm(dim=-1), self.results, self.nnz)
                self.last_branch = "sampled"
            else:
                self.nnz.copy_(torch.tensor([self.offload_lengths[layer][request] for request in range(c.batch) for _ in range(c.heads)], dtype=torch.int32))
                self.attention.full_attention(layer, self.output, self.metadata, flat.float(), self.nnz)
                self.last_branch = "baseline_full"
        else:
            self.last_branch = "dense"
        if not dense:
            nonempty = self.nnz > 0
            if not bool(torch.isfinite(self.output).all()) or not bool(torch.isfinite(self.metadata[:, nonempty]).all()) or not bool(torch.isneginf(self.metadata[:, ~nonempty]).all()):
                raise ValueError("invalid native attention output/LSE")
        # Prepare append separately; native rejection cannot mutate resident cache.
        cache = self.resident[layer].clone()
        for request in range(c.batch):
            position = self.lengths[layer][request]
            centered = key[request, :, 0] if dense else key[request, :, 0] - self.average[layer][request, :, 0]
            cache[request, 0, :, position].copy_(centered)
            cache[request, 1, :, position].copy_(value[request, :, 0])
        resident, resident_lse = self._resident_attention(layer, query[:, :, 0], cache)
        self.last_resident_lse = resident_lse.clone()
        if dense:
            result = resident
            self.last_offload_lse = torch.full_like(resident_lse, -torch.inf)
        else:
            offload_lse = self.metadata[1].reshape(c.batch, c.heads)
            # CPU base2 merge; actual FlashInfer merge remains a CUDA boundary.
            total = torch.logaddexp(resident_lse * math.log(2), offload_lse * math.log(2)) / math.log(2)
            result = resident * torch.exp2(resident_lse-total).unsqueeze(-1) + self.output.reshape(c.batch, c.heads, c.dim).float() * torch.exp2(offload_lse-total).unsqueeze(-1)
            self.last_offload_lse = offload_lse.clone()
        if not bool(torch.isfinite(resident_lse).all()) or not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite derived attention/merge state")
        self.resident[layer].copy_(cache)
        self.lengths[layer] = [length+1 for length in self.lengths[layer]]
        self.pending.remove(layer)
        return result.reshape(c.batch, 1, c.heads*c.dim)

    def clear(self) -> None:
        """Reset all owned buffers/state and real native owners for prompt reuse."""
        for cache, average in zip(self.resident, self.average):
            cache.zero_(); average.zero_()
        for lengths in self.lengths + self.offload_lengths:
            lengths[:] = [0] * self.config.batch
        self.nnz.zero_(); self.results.zero_(); self.output.zero_(); self.metadata.zero_()
        self.filled.clear(); self.built.clear(); self.pending.clear()
        self.hashes.clear(); self.sorted_payloads.clear(); self.chunk_ranges.clear()
        self.last_branch = "clear"
        self.last_resident_lse = torch.empty(0); self.last_offload_lse = torch.empty(0)
        self.attention.clear()
        if self.k:
            assert self.retrieval is not None
            self.retrieval.clear()
