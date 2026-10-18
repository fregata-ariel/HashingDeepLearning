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

The optional local acceptance gate `--require-native` requires both native build modes:
either native build mode
skipping fails the command after its evidence file has been written. Default
exploratory runs preserve capability skips without treating them as execution.
Each completed build mode emits a bounded, percent/CR/LF-escaped Actions notice
with the tested commit and `EXECUTED_PASS` metrics or `SKIPPED` missing flags.
These log notices remain inspectable when the full evidence artifact cannot
be retrieved. They do not change the numerical or GPU scope.

## Required compatibility execution on CPUs without AVX512

`--execution-mode simde --require-execution` runs the same actual maintained
translation units and unchanged numerical fixture for both BF16 source branches.
This is **emulated intrinsic execution**, labeled `EMULATED_PASS`; it never
satisfies `--require-native` (the conflicting arguments reject). The default
`native` mode retains CPU/OS capability gating and native `EXECUTED_PASS`/`SKIPPED`
notices. Hosted native skips remain skips alongside required compatibility
execution; neither result establishes GPU behavior.

The forced-include [adapter](compat/simde_cpu.h) loads real Torch, immintrin and
archived FBGEMM declarations before enabling only AVX512-family SIMDe aliases.
It sets the existing BF16 source selector only after all dependency headers.
Compiler flags force baseline x86-64 with AVX/FMA disabled and `SIMDE_NO_NATIVE`.
The extension binary audit rejects AVX instructions, YMM/ZMM/mask registers and
unconditional FBGEMM AVX512 imports. Torch/FBGEMM shared libraries retain their
own safe runtime CPU dispatch; the audit does not describe their entire binaries.

The adapter explicitly redirects the maintained `*_avx512` converter callsites
to actual wheel FBGEMM `*_simd` runtime dispatch. This is an adapted callee
boundary, not execution of unconditional AVX512 converter machine code on a
weaker CPU. Actual Tensor ABI/conversion/aliasing/lifetime tests remain unchanged.
The same already-quantized operand FP64 oracles and budgets run in both modes.
SIMDe fallback BF16 results on these ordinary finite fixtures do not establish
arbitrary native BF16 DAZ/FTZ/special-value or performance equivalence. Inherited
constructor text naming AVX512 denotes the selected source branch, not hardware
execution; the explicit mode and notices are the execution evidence.

`compat/simde-source.json` pins official upstream snapshot
`2e5224981e9123b8fe89a9833a978556d636a565`, archive SHA256
`8bc6273b9ad2d44f1f3b4d00a440a76a2e1143b3be2133fb036208b9ea3aec4a`,
and MIT [license](compat/SIMDE_LICENSE). This snapshot includes reduce_add_ps;
no local arithmetic replacement or edited third-party header is used.
`tools/magicpig_simde.py` downloads that exact HTTPS archive, verifies its digest,
and validates every cached header against it before compiling. Modified/extra
cache files reject. Evidence records the actual include directory, archive/pin,
complete header-tree and avx512.h hashes, adapter/bootstrap and binary hashes.
Native and emulated evidence use separate build roots to preserve both results.
