Issue #115 review evidence (draft; not posted)

The shared SYCL implementation corrects two maintained CUDA source hazards.
The maintained CUDA files and immutable `third_party/g-slide` archive remain
unchanged; neither correction closes native GPU acceptance (#34/#43).

`python3 tools/check_gslide_source_regressions.py --build-dir /tmp/gslide-source-regressions`
extracts the current exact bodies and compiles with GCC. The fixture uses one
thread per block, serial atomic addition and a barrier that asserts block size
one. These launches are for source index logic only, not valid evidence for
CUDA parallel behavior. Evidence JSON includes exact extracted body hashes.

1. `kernel.cu:77 get_dense_acts_knl`, nodes=1, batch=2, capacity=2;
   initialize offsets `[-71,-71,-71]`, execute original body at global IDs 0,1.
   Independent contract: IDs `[0,0]`, offsets `[0,1,2]`. Actual original offsets
   `[0,1,-71]`: terminal write is guarded by `tid<node_num*batch_size`.
   Shared `dense_ids` submits an independent offsets extent, including empty
   and one-node batches. It is compared with integer expected offsets.

2. `lshKnl.cu:114 init_hash_knl`, previous=2, nodes=2, L=2, K=1,
   bin=2, tile=2, per_thread=1 (two lanes/node), buckets/table=4,
   bucket capacity=2, bins `[0,1,0,1]`, rows `[2,1]` and `[1,2]`.
   Execute the source's grid formula: two blocks, one thread each. Expected
   each table: node0 at address0, node1 at address1. Original inserts node0
   at table0/address0 and table1/address1; node1 is absent. This small scalar
   launch exposes the source's `tid/lanes` versus staged `tid` row mismatch
   and undercoverage independently of scheduling. The new `build_hash` uses
   global node groups and node-based local staging; its CPU SYCL fixture uses
   local size64 with node counts1/31/33/65, tile tails and lanes1/2/4.

The native source defects remain known limitations. Native repairs need their
own legal CUDA launch verification. These translated corrections are not new
original-body execution coverage and do not redefine the old Golden results.
