# MagicPIG cache/merge CPU contract (#59, MP3-01)

This is a **restricted support test** of pinned v0.2 Python orchestration.
The immutable archive under `third_party/magicpig/` is unchanged. There is no
maintained Python server port. The `cache_merge` suite extracts nine **actual
AST statement ranges** from `LSHSparseAttnServer.fill`, `plan`, `decode` and
`clear`. It does not import the archived module, Torch, CUDA or FlashInfer.

- Fixture: `tests/traceability/test_magicpig_cache_merge_cpu.py`
- Descriptor: `tests/traceability/magicpig_python_suites/cache_merge.json`
- Ledger: `papers/traceability/magicpig-cache-merge.json`

The existing driver checks the approved source, class, block names, source
SHA256 and selected source-statement SHA256. The Python AST adapter is recorded
as `not_applicable_python_adapter` in sanitizer mode, **not** as a native
ASan/UBSan/LSan test.

## Actual selected source and CPU substitute

| Pinned source | Restricted substitute | Verified contract |
| --- | --- | --- |
| `fill` partition and centering | FP64 list Tensor | Sink/local vs offload membership and order; common offloaded-key mean |
| `fill` sparse/dense copy | Fixed-shape host copy slots and bounded views | Keys/values, destination slices and page-length assignments |
| `decode` appended-key centering | FP64 Tensor | Reuses fixed mean across requests |
| `plan` | Length buffers and plan-call spy | Two step increments and GQA/dimension argument routing |
| `decode` merge call | Base-2 FP64 host merge | Actual source call arguments, independent corrected-union attention oracle |
| `clear` | Zeroable buffers and server spies | Per-layer/request state reset and reuse |

Fixed, nonrandom CPU fixtures exercise 16 extracted partition cases
(lengths 3/4/5/8, two requests, two KV heads) and 312 query-head
merge compositions (GQA groups 1/4/8, two requests, lengths 3/5/8,
sampled or full selection). Every prefill token belongs exactly once to
sink/local or offload; the appended token is distinct.

For sampled offload IDs, the independent FP64 union oracle builds
logits as \(q\cdot k/\sqrt{d}-\ln(p)\) with inclusion probabilities
0.4 or 0.65, alternating by selected position; both values are asserted to
execute, including heterogeneous probabilities within a sampled subset.
Static/new tokens have \(p=1\). For full selection
every \(p=1\), and 156 head cases match independently computed dense
attention. The independent oracle computes the union using direct stable
natural exponentials, then converts total LSE to base 2.

The host `flashinfer.merge_state` substitute uses base-2 LSE values.
For \(m=\max(L_a,L_b)\), set \(w_a=2^{L_a-m}\), \(w_b=2^{L_b-m}\)
and return \((w_ao_a+w_bo_b)/(w_a+w_b)\), with combined LSE
\(m+\log_2(w_a+w_b)\). The independent direct-union oracle does not
reuse this merge equation. Sensitivity tests detect omission of sampled
correction, inconsistent/stale centering, and natural-log versus base-2
LSE confusion. A separate sensitivity check rejects replacing heterogeneous
sampled probabilities with one uniform probability, which would lose
within-offload value reweighting. These are host-oracle sensitivity comparisons;
they do not execute or mutate production probability-correction kernels.
All 104 heterogeneous-probability query heads must show the expected
within-offload reweighting compared with a uniform-probability substitute.
Selected-source centering also preserves independent
dense attention probabilities under common key translation.

The dense prefill caller in pinned `models/llama.py` allocates full
`max_length` key/value buffers and passes them together with logical `seq_len`.
The host fixture follows that padded-buffer contract: the selected dense fill
copies the entire fixed-size page and sets its logical length separately.
Short unpadded inputs are shape-error characterizations, not supported dense
prefill fixtures. The earlier shape-free copy slot accepted those inputs
incorrectly; this review correction changes the test adapter, not production.

Four host page-boundary cases include exact capacity and sparse/dense overflow;
four nonbroadcastable source-copy cases must reject incompatible shapes.
These guards belong to the host substitutes, not the archived implementation.
A shared two-layer/two-request state is filled, planned and appended for two
steps, cleared, and reused with different inputs for two more steps. Resident
prefixes, append offsets, the fixed centering reference, request/layer
isolation and reset are checked. Appended-key writes use a host position-copy
surrogate; the native FlashInfer append invocation is not selected or executed.

## Explicit restrictions and pending work

- Sparse fill with no offloaded token is **not supported**:
  `seq_len <= num_sink_tokens + num_local_tokens`. Two short-input
  characterizations expose the empty mean reduction; these are not passes.
- Host copy shape and cache capacity are checked with explicit negative
  cases by the fixture. Actual pinned source
  `plan()` does not establish safe native overflow, stream ordering or
  asynchronous transfers.
- FP64 dimension-two surrogate tests verify merge algebra only; #58
  supports separate actual selected kernels primarily with dimension 128,
  uniform candidate lengths and GQA groups 1/4/8.
- No true FlashInfer runtime, Torch ABI/BF16 conversion, FBGEMM dispatch,
  OpenMP/concurrent server, native GPU ownership or pretrained generation
  has been verified. No model weights or datasets are downloaded.
- Repairs #65–#68/#72–#77/#86 and MP5 are closed; deferred
  native-GPU task #62 remains separate. #60 is integrated and covers connected
  hash/retrieval/correction/attention/cache steps; #61 covers strict typing,
  final docs and CI integration.

## Reproduce

From repository root:

```bash
python3 tests/traceability/test_magicpig_cache_merge_cpu.py
python3 tools/check_magicpig_cpu.py --suite cache_merge
python3 tools/check_magicpig_cpu.py --suite cache_merge --sanitize
```

The default CPU driver and existing traceability workflow also enroll the
`cache_merge` Python suite. Hosted C++ sanitizer suites remain separate
evidence; Python AST execution does not inherit their runtime guarantees.
