# MagicPIG LSH selected-body CPU contract

Issue #56 executes the maintained LSH constructor/destructor, `alloc`, `fill`,
`retrieve`, `batch_retrieve`, `clear` and `get_mask` bodies without linking Torch.
The executable includes exact selected bodies from `library/lsh/lsh.cc`; the
historical source snapshot and production arithmetic remain unchanged.

`magicpig_lsh_adapter.hpp` supplies a visibly separate non-owning contiguous
Tensor view, TensorOptions/from_blob substitutes and the maintained class layout.
Only access to `retrieve` is exposed for testing. The adapter's checked shape
access is not production Torch's dtype/device/contiguity validation, allocator,
dispatcher, ABI or ownership. Caller buffers must have enough storage for every
shape. Portable runs explicitly replace `fast_memcpy_avx512` with `std::copy_n`.
Native runs may execute the actual copy helper only after the CPU/OS capability
probe approves AVX512F (the common driver may conservatively also require FMA).
Compilation is serial, without `-fopenmp`; threaded execution is unverified.

## Valid-input sorted-fill lifecycle

The caller allocates once per instance, with positive/divisible GQA dimensions,
small bounded `K`, valid layer/request/head IDs and lengths no greater than
capacity. A second `alloc` is outside this tested contract: upstream appends
storage vectors and replaces `mask` instead of resetting ownership.

`fill` receives contiguous int16 bucket hashes sorted within each KV-head/table,
with values in `[0, 2^K)`, and contiguous int32 indices that are permutations of
valid token IDs within each table. This uniqueness condition matters: repeated
IDs within one table would let `retrieve` count repeated membership as two hits.
These are caller preconditions, not checks newly added to production code.
Query storage is contiguous int32 with valid bucket IDs. All output storage is
caller-owned, independent of inputs and sufficiently large. Input buffers are
mutated/deallocated after fill to establish that retrieval uses copied indices
and table boundaries, rather than borrowing caller input storage.

The independent oracle examines each token's original literal vector once and
counts equality across distinct table coordinates. It has no bucket boundaries,
masks, sorting or retrieval control flow in common with production. Tests cover
0/1/2/3/4 collisions and deduplication, empty buckets, all collisions, lengths
0/1/2/15/16/17/31/32/35, 2 layers, batch 2, GQA 4-to-2, repeated query masks,
clear/refill and three allocation/destruction lifetimes. There are 2688 batched
head comparisons plus direct retrieval and edge cases. Bounds are verified with
row-tail sentinels, outer canaries and fatal ASan/UBSan instrumentation.

`get_mask` is a non-owning view. Shape and current mask values are inspected only
while the LSH owner lives; clear is checked through the live view. No dangling
view is dereferenced after teardown. This does not establish safe retained Torch
views after owner destruction. `get_table_start`, `get_table_end` and `get_table`
are declared in `lsh.h` but lack source definitions/bindings; no tests or claims
cover them.

## Isolated fastfill defect diagnostic

The same fixture's `--fastfill-probe` executes the exact alternate `fastfill`
body in a separate process. Four literal tokens all collide in four tables; the
independent expected set is `{0,1,2,3}`, but retrieval returns `{0}`. The body
updates bucket counts/prefixes and never writes token indices to `table`, which
remains zero from allocation. It also allocates `mo_ij` once per table and never
frees it (64 bytes across four allocations for this fixture).

The ordinary passing suite compiles/hashes this body but never executes its
leaking path. The explicit diagnostic must retain the exact defect marker and
wrong-result check. Hosted sanitizer validation additionally requires a positive
LeakSanitizer allocation-site report; an unrelated sanitizer/process failure is
not evidence of the expected leak. Use an unoptimized dedicated leak build when
necessary to prevent the unused allocation being optimized away. An explicit
local-only leak-disable option may work around sandbox `/proc` restrictions,
but that outcome proves neither leak freedom nor a successful leak diagnostic.

`fastfill` remains unsupported; [repair #66](https://github.com/fregata-ariel/HashingDeepLearning/issues/66) tracks it. A focused repair needs both table population and
temporary ownership, with valid-input equivalence to sorted `fill` and a clean
sanitizer run before it can join the ordinary passing contract. No GPU execution,
Torch extension build, model generation, sampling recall or performance claim is
made by this CPU contract.
