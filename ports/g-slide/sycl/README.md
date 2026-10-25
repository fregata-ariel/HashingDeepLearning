# G-SLIDE shared SYCL device boundary (#115)

The current component structure and feature/validation scope are documented in
[the library design](../../../docs/GSLIDE_LIBRARY_DESIGN.md). The checked component API is in `component_api.hpp`; named configuration and
SDK-free views are in `configuration.hpp` and `views.hpp`. See the
[paper/reference semantic review](../../../docs/GSLIDE_SEMANTICS.md). Lower-level
raw-pointer entries and `kernels.hpp` remain compatibility interfaces.
CTest now registers fifteen host/component/integration tests separately.

This maintained implementation submits the same C++ kernels to a SYCL queue
selected by the caller. The routine fixture requires an actual CPU device;
NVIDIA execution is separately recorded in the GPU checkpoints. CUDA source
execution, performance and CUDA lock progress are distinct contracts (#34).
It supplements the existing five CUDA-body CPU extraction
suites rather than reclassifying their scalar adapters as parallel execution.

Configure `tests/gslide_sycl` with AdaptiveCpp and run CTest. Toolchain activation
is described by `docs/SYCL_CPU_BACKEND.md`. The test prints device metadata
and a check count, throws on failed comparisons, and waits for asynchronous
errors at each boundary. No datasets, GPU or paid resource are needed.

All arguments are borrowed USM in the supplied queue context. Calls complete
before return. The caller clears frequencies and bucket sizes before rebuild or
reuse; owns allocations through completion; supplies monotone offsets, valid
IDs and finite weights; and provisions every stated capacity. Element counts
and offsets use signed 32-bit integers. Empty batches are accepted. Local size
must fit the device, and hash local size must be divisible by tile lanes.

| Original definition (baseline `097b795`) | Shared entry / accepted contract |
| --- | --- |
| kernel.cu:21 `warp_reduce`, :49 `warp_max` | `reductions`: actual SYCL subgroup collective, actual subgroup width |
| kernel.cu:27 `block_reduce`, :55 `block_max` | `reductions`: local-memory tree across all work-items, padded identities |
| kernel.cu:77 `get_dense_acts_knl` | `dense_ids`: `[batch,node]` IDs and `batch+1` offsets, including zero/one-node extents |
| lshKnl.cu:18 unsigned `gen_rand_keys_knl` | `random_keys(packed=true)`: minstd 48271, prefix `id/previous<<16`; preserves released `1<<15` mask (operator precedence), not a full 16-bit random suffix |
| lshKnl.cu:34 signed `gen_rand_keys_knl` | `random_keys(packed=false)`: positive minstd values, same fixed seed contract |
| lshKnl.cu:47 `init_hash_tt_knl` | `build_hash(per_thread=tile)`: local bins and rows, bounded atomic bucket ring |
| lshKnl.cu:114 `init_hash_knl` | `build_hash(per_thread<tile)`: multiple lanes per node, corrected node-based staging and grid |
| GPUMultiLinkedHashTable.cu:178 random `d_activate_labels_seq` | `activate_labels(permutation!=nullptr)`: sequential per-sample labels plus cyclic minimum padding |
| GPUMultiLinkedHashTable.cu:290 `block_reduce_cnt_knl` | `count_candidates`: one group per sample, gathered boundaries `offset[sample*L]` |
| GPUMultiLinkedHashTable.cu:301 label wrapper | `activate_labels(permutation=nullptr)`: one work-item per sample, partial guard |
| GPUMultiLinkedHashTable.cu:313 random wrapper | `activate_labels(permutation!=nullptr)`: connects the random overload and seed/sample |

These are translated implementations, not execution of the original CUDA
bodies. The four reductions intentionally replace warp-32 shuffle assumptions
with subgroup/local-memory algorithms. Direct-address frequency storage
`[batch,nodes]` replaces the native linked pool: capacity is `batch*nodes`,
without a spinning lock. Sequential per-sample padding is the source's intended
small-label algorithm, not a claim of intra-sample parallel padding. Frequency
counting and hash insertion do execute multiple work-items with real atomics.
Random padding requires a permutation of all node IDs; invalid permutations are
outside the borrowed-input contract. It makes at most one complete cycle.

WTA uses the released natural-log packing variant, first-position ties and
negative values. The fixture compares bucket sets and counts with a separately
implemented scalar oracle, never insertion order. Over-capacity rings specify
only bounded retained IDs and full insertion count, not a stable survivor set.
Atomic entry stores avoid a C++ data race on overwritten slots.

Source inspection found two original hazards; the SYCL contract corrects them
without changing archived CUDA: dense offset emission is nested inside the
node extent (missing the terminal offset for one/zero nodes); alternate hash
computes `node=tid/lanes` while loading/reading rows by `tid` and launches only
`ceil(nodes/local)` blocks. The checked-in source-extraction regression records
concrete original-body behavior separately; neither correction is evidence of
native CUDA execution or repair. #43 and #34 remain open.

Tests cover counts 1/31/33/65/129, negative maxima and tails, 32/64/128 local
sizes and actual subgroup partitions; zero/one/partial dense extents and guards;
fixed RNG seeds, sequence/range/nonconstant/reuse and packed variant; both tiled
hash modes, partial tiles, ties, negative rows, query agreement and bucket ring;
65 samples with L-offset mapping, duplicate labels, empty tables, floors 0/5/7,
full cyclic padding, reset/reuse and capacity rejection. Source identities and
exact commands are captured by `tools/check_gslide_sycl.py`.
