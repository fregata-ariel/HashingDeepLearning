# MagicPIG selected attention CPU contract (#58)

The `attention` suite executes the maintained `qk_kernel`, `qk_kernel_full`,
`transform_kernel`, `softmax_kernel`, `softmax_kernel_full`,
`softmax_kernel_optimized`, `wv_kernel`, `wv_kernel_dim128_full`, and actual
`SparseAttentionServer` constructor/destructor, `alloc`, `fill`, `clear`,
`attention`, `dynamic_attention`, `full_attention`, and four getters.
Production function bodies are extracted unchanged. This verifies selected CPU
composition and ownership support, with supplied candidate multisets.

The test dependency adapter is explicit. It models contiguous caller-owned
Tensor buffers and non-owning `from_blob` views. Queries are already FP32, and
model `Tensor.to(kFloat32)` returns the same buffer. It does not exercise Torch
allocation, dtype conversion, dispatch, shape validation or ABI. BF16 decode and
output conversion use a software model of the pinned FBGEMM finite-value policy:
add `0x8000` to FP32 bits then shift right 16. This is not round-to-nearest-even;
actual conversion intrinsics and FBGEMM linkage remain separate tests. Portable
FP32 lanes model multiplication, FP32 fused multiply-add and reduction; host exp
replaces the native polynomial. OpenMP pragmas execute serially, without OpenMP
runtime or production scheduling. Native AVX512F/FMA executes actual QK/WV/exp
arithmetic after the shared compiler/CPU/OS capability gate; conversion remains
the labelled model on both paths.

The independent FP64 oracle decodes BF16 by significand/exponent arithmetic,
computes dense dots/norms before supplied-index selection, enumerates the
at-least-two-hit Binomial(4,p) mass for K=3, retains the released `+1e-4`, and
computes probabilities, weighted values and base-2 maximum/LSE. Repeated row IDs
are a multiset: each supplied occurrence contributes, rather than being
implicitly deduplicated. Both actual same-data all-candidate sparse and dense
pipelines match their separate oracle; the distributions differ because sparse
inclusion correction remains even when all rows are supplied.

## Verified configurations

| Path | Conditions |
|---|---|
| Static/dynamic sparse server | 2 layers, 2 batches, 2 KV heads, 2 query groups; dim16/32/128; nnz1/15/16/17/31/32; unique unsorted or repeated supplied IDs |
| Full server | dim128; groups1/4/8; 2 batches, 2 KV heads; nnz16/32 identical across KV/query heads |
| Capacity | 32 valid initialized index/score slots per head, including rounded QK padding; output/metadata guards |
| Ownership | alloc once, fill copies source arrays, getter mutation aliases caches, clear updates existing views, server outlives views, destructor sanitizer checked |

The full server uses `nnz[i]` with both KV and query-head indexing; this suite
uses uniform values. A separate defined-memory reproducer expands KV lengths
[16,32] into per-query lengths [16,16,16,16,32,32,32,32] with four groups.
For zero logits and values of one, the second KV head outputs 0.5 instead of the
independent dense value 1.0: normalization uses 32 rows while WV sums only 16.
Independently varying GQA lengths remain unsupported pending a separate repair. Queries and keys/values are distinct across heads/batches/layers to
exercise stride routing.

There are 682 head compositions, 13,312 probability checks, 47,168 BF16 output
coordinate checks, 432 ownership checks, 32 corrected-score checks, and 11 boundary cases. Portable absolute
budgets are `3e-5` corrected score and probability, `0.004` output (including BF16 quantization), and
`5e-5` base-2 metadata. Native budgets are `0.002`, `0.008`, and `0.03`; the
metadata budget is the existing polynomial bootstrap budget. Local maxima were
portable `2.70259e-6 / 9.71851e-4 / 1.35911e-5`, native
`0.00123006 / 0.00215608 / 0.00695390`. These observed errors characterize fixed
moderate fixtures, not universal bounds on the approximation. The directly
checked pre-Softmax corrected-score maximum error is `2.51057e-6` on both paths;
this check also detects additive logit shifts that normalization would hide.

## Characterized restrictions

* QK rounds candidate loops to 16, reads padded indices and writes padded scores.
  Only valid `nnz` scores are cleared, so padded scores accumulate onto previous
  initialized values. Caller must supply capacity and valid row IDs through
  `round_up(nnz,16)`; this suite does not claim nnz-sized buffers are safe.
* Full Softmax scales and normalizes padded score slots. Valid probabilities are
  checked independently, with capacity canaries beyond the rounded region.
* QK/WV process `HEAD_DIM/16` blocks. A dim17 QK fixture with a nonzero final
  coordinate yields zero instead of one, and dim17 WV leaves its last output
  untouched. General sparse support requires dimension divisible by16.
* Full WV processes a fixed128 coordinates and accepts groups1/4/8. Group2
  silently returns without output writes. Full tests enforce dim128 and allowed
  groups; arbitrary dimensions/groups are not supported claims.
* Empty candidates are not called in this suite. The separate Softmax suite
  covers empty inputs and exact score-allocation tail diagnostics; QK exact
  nnz allocation is not tested here. Invalid layer/request IDs, counts,
  capacities, head divisibility, repeated allocation, expired views and OpenMP
  races remain unsupported.

This work changes no maintained arithmetic. It is not whole-extension build
success, LSH-to-attention end-to-end validation, model generation, accuracy or
performance reproduction. GPU tests remain postponed.
