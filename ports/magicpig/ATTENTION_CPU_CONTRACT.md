# MagicPIG selected attention CPU contract (#58, repairs #73/#75/#76/#77)

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
a KV-sized count vector are rejected. Repair #73 accepts zero counts, writes zero BF16 output and negative-infinity
base2 maximum/LSE, and leaves the empty head's score row unchanged. Sparse
empty heads bypass correction, so undefined empty norms are never read.

There are 832 nonempty head compositions, 16,544 probability checks, 66,368 BF16 output
coordinate checks, 432 ownership checks, 32 corrected-score checks, and 175 boundary cases (including 90 empty server heads and26 invalid caller cases) including dimension/group rejection. Portable absolute
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
* Repair #73 checks empty and mixed-empty heads in static/dynamic sparse
  server and full server across two layers and batches. Full groups1/4/8 and
  sparse group2 execute actual selected server methods; empty score rows and
  output/metadata canaries remain unchanged. Native BF16/scheduled source paths
  apply the same empty policy, but those server dispatches are not executed by
  this std-only fixture. Invalid request IDs, other shapes/ABI/dtypes,
  reallocation lifetime beyond alloc rejection and OpenMP races remain outside
  these support claims.


Repairs #75/#76/#77 enforce logical candidate bounds, dimension/group policy
and consistent query-head length routing. It is not whole-extension build
success, LSH-to-attention end-to-end validation, model generation, accuracy or
performance reproduction. GPU tests remain postponed.

## Repair #67 caller validation

Sparse static/dynamic entrypoints validate tensor extents, logical counts,
candidate IDs and K/L before OpenMP dispatch. A shared free helper stages the
actual QK and correction results for all heads in temporary capacity-backed
storage and invokes the maintained transform domain checks on that storage.
Only after every head passes are logical score rows committed and Softmax/WV
workers started. A bad norm, nonfinite dot or inconsistent cosine in a later
head therefore leaves all exposed scores, output and metadata unchanged.
Empty heads retain #73's no-norm-access policy. This helper avoids a second,
potentially divergent cosine/probability formula.

The staging pass is serial and uses `batch * query_heads * max_length` extra
floats; this CPU correctness repair does not claim performance reproduction.
Scheduled entrypoints delegate to the staged implementation after their
fixed128/32-head policy check. Wrappers select that path only for dim128.
BF16 entrypoints use the same staging helper with actual native BF16 QK when
compiled; native server dispatch remains unexecuted on a CPU without that ISA.

Twenty-six static/dynamic regressions put invalid norms, nonfinite queries,
out-of-domain cosine, counts or IDs in later heads and invalid K/L in the
caller. Every rejected call preserves all score rows, output and metadata.
These require the transform #67/#68 fixes; ordinary caller shape validation
is still a selected dependency-model check, not whole Torch ABI coverage.

The combined caller/#67/#68 selected fixture also passes ASan/UBSan/LSan with
actual OpenMP enabled and `OMP_THREAD_LIMIT=4` on the portable arithmetic
backend. This verifies rejected domains are handled before workers and that
the supported small composition runs through the real OpenMP runtime; it
does not establish native SIMD or unrestricted production thread scheduling.

## Repair #86 local SIMD reduction storage

`softmax_kernel` explicitly aligns its16-float local reduction array to64
bytes. The existing nnz1 actual-attention fixture exercises this local store
with no complete16-candidate score tile; summation order, numerical budgets
and logical score bounds are unchanged. The change follows an actual Hosted
GCC13 sanitizer fault at this local `_mm512_storeu_ps` call, while the same
native plain fixture passed. The failing instruction and exact compiler cause
must be established from the original executable's disassembly; source-level
stack alignment alone does not establish a GCC defect. Repair native execution
and complete sanitizer results require the dedicated Hosted validation run.
