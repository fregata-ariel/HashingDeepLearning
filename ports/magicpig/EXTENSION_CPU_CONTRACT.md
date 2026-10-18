# Real CPU extension boundary (#117)

`tools/check_magicpig_extension_cpu.py` builds the actual maintained translation
units with real Torch pybind/Tensor APIs. The wheel supplies the C++ ABI (no
unconditional old ABI=0 override), allocator and exported FBGEMM implementation.
FBGEMM headers are the unchanged pinned archive. Runtime `FloatToBfloat16_simd`
and the server's AVX512 converters resolve to **wheel FBGEMM**, not substitutes
and not a claim to compile every archived FBGEMM source. Finite midpoint/tail
conversion is compared bitwise to the existing half-up model and records the
contrast with Torch's ties-even input quantization.

All accepted pointers are contiguous CPU storage. Sparse queries accept FP32
or BF16 in non-BF16 builds; BF16 entrypoints require BF16. Keys, values, outputs
are BF16, norms/metadata FP32, candidate IDs/counts int32. `fill` consumes
`[KV, length, dim]` and `[KV,length]`; requests/layers/capacity are checked before
writes. Sparse calls use `[batch*Q,dim]`, `[2,batch*Q]`,
`[batch*Q,capacity]`, and `[batch*Q]`. Grouped heads route to flattened
`head/groups`. Candidate multisets are permitted; unused padding is ignored.
Zero nnz writes zero output and negative infinity maximum/LSE. Invalid logical
counts/IDs/norm/correction domains reject before committing score/output/LSE.
No concurrent calls against one mutable server are supported.

LSH accepts K1..11, L>=2, positive/divisible heads, capacity/layers/batch.
Sorted fill accepts `[KV,L,length]` int16 codes and int32 permutation payloads;
fastfill accepts int32 codes. Sorted replacement clears absent buckets.
Retrieval uses `[batch*Q,L]` int32 codes and capacity-backed result/count outputs.
Layer/request, dtype/layout/device, shape and logical code domains reject before
pointer access. `copy` remains an unused archived no-op and is outside this API.

Getter Tensors borrow server memory: live mutations and clear are observable.
Direct Python getter results retain their owner via `keep_alive`. Callers must
keep the owner alive for derived aliases; use `clone()` for independent escape.
Tests never dereference a view after owner destruction. Allocation is once per
owner. Capacity is deliberately tiny; pathological allocation-size overflow and
concurrent mutation are outside this CPU validation contract.

Both BF16-disabled and enabled builds execute wrapper calls at flattened heads
32/d128 (scheduled forwarding),32/d32 (static),64/d32 (static),72/d32(dynamic),
with two batches/KV heads and varying GQA. Dispatch instrumentation records the
actual maintained wrapper/callee entries. Scheduled bodies remain forwarding
implementations. No unused WV variant or LSH::copy is enrolled.

Independent FP64 expectations use the exact already-BF16-quantized Q/K/V.
Probability <=.002, WV <=.008, base2 metadata <=.03 reuse the existing native
attention budgets. Existing probability and isolated BF16 QK oracle suites keep
their tighter stage budgets. Maintained extension compilation enables fatal UBSan. Existing selected-body ASan/UBSan and four-thread
OpenMP tests remain; this extension fixture also observes four actual OpenMP
workers. The Torch wheel itself is unsanitized; no whole-wheel sanitizer claim.

CPU/OS flags are checked before loading an AVX512 binary; unsupported AVX512 or
BF16 modes are emitted as skips, never native passes. The portable numerical
model remains the existing default CPU suite. CUDA/FlashInfer, streams,64-thread
NUMA performance and model generation remain #62 and unverified here.
