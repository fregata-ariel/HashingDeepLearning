# MagicPIG LSH selected-body CPU contract

Issue #56 executes the maintained LSH constructor/destructor, `alloc`, `fill`,
`retrieve`, `batch_retrieve`, `clear` and `get_mask` bodies without linking Torch.
The executable includes exact selected bodies from `library/lsh/lsh.cc`; the
historical source snapshot remains unchanged. Issue66 deliberately repairs the
maintained `fastfill` implementation; sorted-fill retrieval arithmetic is retained.

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

## Historical fastfill defect and maintained regression

Issue56/PR70 recorded the archived alternate path returning `{0}` instead of
the independently expected `{0,1,2,3}`: prefixes were filled without token IDs,
and one temporary per table leaked64bytes/four allocations. Those historical
commit/CI records and the immutable archive retain the characterization.

Issue66 deliberately repairs maintained table population and uses scoped cursor
ownership. The same fixture's `--fastfill-probe` now runs the repaired body and
requires exactly `{0,1,2,3}`, a normal zero exit, the exact repaired marker and no
stderr diagnostics. Any leak, sanitizer error, wrong IDs/marker or fake success
fails even when the process exits zero. Hosted sanitizer execution records
`leak_status=clean`; ordinary and explicitly leak-disabled runs record
`not_checked`. The driver records `fastfill_regression`, not a reproduced defect.

The dedicated [repair contract](FASTFILL_REPAIR_CPU_CONTRACT.md) defines int32
CPU/layout/range checks before mutation, targeted repeat-fill replacement and
independent15360head parity/lifecycle checks plus14boundary rejections. The
metadata shim is not real Torch validation/ABI evidence. No GPU execution,
Torch extension build, model generation, sampling recall or performance claim
is made by this CPU contract.
