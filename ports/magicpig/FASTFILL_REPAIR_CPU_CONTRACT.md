# Maintained fastfill repair (#66)

`LSH::fastfill` previously populated bucket prefixes without token contents and
leaked one cursor allocation per table. Issue66 repairs that maintained path;
the immutable v0.2 archive preserves the original defect.

The owner must have been allocated once with valid bounded K/L/head/capacity
configuration. The call validates owner state, layer/request, rank3, CPU,
contiguity, int32 dtype, `[KV heads,L,length]`, capacity and every bucket code
before writing any table state. Accepted codes lie in `[0,2^K)`. Repeated calls
replace only the chosen layer/request, including shorter and zero-length input.
Input hash storage is borrowed only during the call; owner tables retain copied
token IDs. Stable bucket placement preserves increasing token order within each
bucket. Temporary cursors use scoped `std::vector` ownership.

`MAGICPIG-FASTFILL-REPAIR-CPU` executes selected maintained bodies using a narrow
Torch metadata/buffer shim. Ten lengths from zero through capacity35, three code
patterns, two owner lifetimes, four replacement generations, two layers/two
requests/two KV heads/four query heads produce15360 independently checked head
results. Every original token is accepted iff at least two distinct table
coordinates equal the query. Sorted-fill comparison supplies additional parity.
Canaries, duplicate rejection, copied input destruction, clear/empty/refill and
14 invalid-input rejections also run. Late invalid codes must leave the previous
valid state intact. Malformed shapes and metadata are rejected before indexing.

Run the enrolled suite with:

```
python3 tools/check_magicpig_cpu.py --suite lsh_fastfill
python3 tools/check_magicpig_cpu.py --suite lsh_fastfill --sanitize
```

LeakSanitizer must be enabled for hosted evidence; expected-defect leak output
is no longer a success criterion. Existing lsh_retrieval's dedicated four-token
probe must now return `{0,1,2,3}` with clean teardown; the shared driver and its
diagnostic rejection tests must be updated alongside this deliberate repair.

This is a maintained variant of the archived alternate table construction,
supporting MagicPIG's Section4.3 candidate-selection rule. Real Torch ABI,
dispatcher behavior, OpenMP scheduling, complete extension linkage and GPU/model
execution remain unverified. The shim exposes synthetic dtype/device/layout
metadata; it does not claim real Torch integration. Native AVX copy remains a
separate capability-gated lsh_retrieval check.
