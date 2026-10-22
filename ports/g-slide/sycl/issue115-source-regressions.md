Issue #115 source regression evidence

The shared SYCL implementation corrects two maintained CUDA source hazards.
At the #115 checkpoint both the maintained CUDA files and immutable archive
were unchanged. The 2026-10-22 native checkpoint repairs maintained CUDA and
keeps the exact pre-repair bodies in the immutable archive as negative controls.
See [native acceptance](../../../docs/NVIDIA_GPU_CHECKPOINT.md).

`python3 tools/check_gslide_source_regressions.py --build-dir /tmp/gslide-source-regressions`
extracts exact archived bodies from `third_party/g-slide` (archive-verified) and compiles with GCC. Dense IDs execute
serially. Hash executes the exact body with 32 ordinary CPU threads per block,
a real `std::barrier` and `atomic_ref`, using a legal original host-dispatch
block size32. This is CPU compatibility execution, not CUDA scheduling or GPU
evidence. Evidence JSON includes exact extracted body hashes.

1. `kernel.cu:77 get_dense_acts_knl`, nodes=1, batch=2, capacity=2;
   initialize offsets `[-71,-71,-71]`, execute original body at global IDs 0,1.
   Independent contract: IDs `[0,0]`, offsets `[0,1,2]`. Actual original offsets
   `[0,1,-71]`: terminal write is guarded by `tid<node_num*batch_size`.
   Shared `dense_ids` submits an independent offsets extent, including empty
   and one-node batches. It is compared with integer expected offsets.

2. `lshKnl.cu:114 init_hash_knl`, previous=2, nodes=64, L=2, K=1,
   bin=2, tile=2, per_thread=1 (two lanes/node), buckets/table=4,
   bucket capacity=64, bins `[0,1,0,1]`. Even rows are `[2,1]`, odd rows
   `[1,2]`. Execute the source host grid formula: two blocks,32 items each.
   Expected each table:64 nodes,32 at address0 and32 at address1. Original
   inserts32 nodes/table: all at table0/address0 and table1/address1; nodes
   32..63 are absent. All staged weight rows used by this fixture are loaded,
   so the mismatch does not depend on reading uninitialized local storage.
   This exposes both `tid/lanes` versus staged `tid` row mismatch and grid
   undercoverage. The new `build_hash` uses global node groups and node-based
   local staging; its real SYCL CPU fixture uses local64 with node counts
   1/31/33/65, tile tails and lanes1/2/4.

The negative-control results above remain the pre-repair observations. The
native checkpoint verifies the repaired maintained CUDA and actual callers
separately; no old failing result is redefined as a correct Golden. The source
contract retains pre-repair hashes alongside repaired source hashes.
