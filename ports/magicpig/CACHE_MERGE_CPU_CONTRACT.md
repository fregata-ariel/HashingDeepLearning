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
| `fill` sparse/dense copy | Host copy slots | Keys/values and page-length assignments |
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
0.4 or 0.65; static/new tokens have \(p=1\). For full selection
every \(p=1\), and 156 head cases match independently computed dense
attention. The independent oracle computes the union using direct stable
natural exponentials, then converts total LSE to base 2.

The host `flashinfer.merge_state` substitute uses base-2 LSE values.
For \(m=\max(L_a,L_b)\), set \(w_a=2^{L_a-m}\), \(w_b=2^{L_b-m}\)
and return \((w_ao_a+w_bo_b)/(w_a+w_b)\), with combined LSE
\(m+\log_2(w_a+w_b)\). The independent direct-union oracle does not
reuse this merge equation. Sensitivity tests detect omission of sampled
correction, inconsistent/stale centering, and natural-log versus base-2
LSE confusion. Selected-source centering also preserves independent
dense attention probabilities under common key translation.

## Explicit restrictions and pending work

- Sparse fill with no offloaded token is **not supported**:
  `seq_len <= num_sink_tokens + num_local_tokens`. Two short-input
  characterizations expose the empty mean reduction; these are not passes.
- Host cache capacity is checked by the fixture. Actual pinned source
  `plan()` does not establish safe native overflow, stream ordering or
  asynchronous transfers.
- FP64 dimension-two surrogate tests verify merge algebra only; #58
  supports separate actual selected kernels primarily with dimension 128,
  uniform candidate lengths and GQA groups 1/4/8.
- No true FlashInfer runtime, Torch ABI/BF16 conversion, FBGEMM dispatch,
  OpenMP/concurrent server, native GPU ownership or pretrained generation
  has been verified. No model weights or datasets are downloaded.
- Existing repair Issues #65-#68 and #72-#77 remain open; deferred
  native-GPU milestone #62 is unchanged. #60 covers connected
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
