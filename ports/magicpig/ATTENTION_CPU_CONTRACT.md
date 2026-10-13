# MagicPIG selected attention CPU contract (#58, repairs #75/#76/#77)

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
| Full server | dim128; groups1/4/8; 2 batches, 2 KV heads; nnz16/32 uniform and varying across KV and individual query heads; two layers |
| Capacity | 32 valid initialized index/score slots per head, QK uses logical nnz only; output/metadata guards |
| Ownership | alloc once, fill copies source arrays, getter mutation aliases caches, clear updates existing views, server outlives views, destructor sanitizer checked |

Repair #77 defines `nnz` as one count per flattened batch/query head:
`[batch_size * num_attention_heads]`. QK, Softmax and WV all read `nnz[i]`,
and shared key/value cache routing is `i / num_attention_groups`.
Queries within one GQA group may have distinct lengths. The original
[16,16,16,16,32,32,32,32] all-one-value regression now produces dense1.0
for the second KV group. Additional FP64 fixtures vary lengths within groups,
between KV heads, between batches and between two layers; keys/values and
queries differ across these axes. Shape, layer and every count are validated
before any score/output/metadata write. Negative and over-capacity counts and
a KV-sized count vector are rejected. Empty full-server counts are explicitly
rejected until repair #73 is composed; empty QK remains supported separately.

There are 786 head compositions, 15,808 probability checks, 60,480 BF16 output
coordinate checks, 432 ownership checks, 32 corrected-score checks, and 60 boundary cases including dimension/group rejection. Portable absolute
budgets are `3e-5` corrected score and probability, `0.004` output (including BF16 quantization), and
`5e-5` base-2 metadata. Native budgets are `0.002`, `0.008`, and `0.03`; the
metadata budget is the existing polynomial bootstrap budget. Local maxima were
portable `2.70259e-6 / 9.71851e-4 / 1.35911e-5`, native
`0.00123006 / 0.00215608 / 0.00695390`. These observed errors characterize fixed
moderate fixtures, not universal bounds on the approximation. The directly
checked pre-Softmax corrected-score maximum error is `2.51057e-6` on both paths;
this check also detects additive logit shifts that normalization would hide.

## Characterized restrictions

* Repair #75 QK reads, clears and writes exactly logical `nnz` candidates,
  without rounding the final candidate tile. Exact-sized indices/scores and
  guarded scores pass nnz0/1/15/16/17/31/32 against same-quantized FP64 dots.
  Empty QK returns before buffer access, permitting null buffers, and negative
  nnz rejects before writes. Native BF16 repeats exact/guarded allocation tests
  behind the compiler/CPU/OS capability gate.
* Full Softmax scales and normalizes padded score slots. Valid probabilities are
  checked independently, with capacity canaries beyond the rounded region.
* Repair #76 rejects nonpositive or nonmultiple-of-16 sparse FP32 QK/WV
  dimensions before buffer access. Native BF16 QK requires positive multiples
  of32; its server methods reject before OpenMP workers. Fixed WV kernels
  require dim128, and the scheduled path requires dim128 before dispatch.
* Full WV and full server require dim128 and groups1/4/8. Unsupported groups
  and dimensions throw `std::invalid_argument` before score, output or metadata
  writes; fixtures exercise dimensions near16/32/128 and unsupported group2.
  General low-level full QK accepts positive groups and dimension multiples16.
* `alloc` rejects invalid/nondivisible dimensions and repeated allocation before
  mutating server state. Sparse groups2 remain supported.
* Empty candidates are not called in this suite. The separate Softmax suite
  covers empty inputs and exact score-allocation tail diagnostics; QK exact logical
  allocation and empty-buffer safety are tested; empty server awaits #73. Invalid layer/request IDs, counts,
  capacities, head divisibility, repeated allocation, expired views and OpenMP
  races remain unsupported.

Repairs #75/#76/#77 enforce logical candidate bounds, dimension/group policy
and consistent query-head length routing. It is not whole-extension build
success, LSH-to-attention end-to-end validation, model generation, accuracy or
performance reproduction. GPU tests remain postponed.
