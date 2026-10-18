# Shared production control on a SYCL CPU (#116)

`include/TrainingControl.h::train_step` now owns the maintained
`Network::train` order: input/label transfer; hidden and selected output forward;
full-capacity hidden-delta reset; output and reverse hidden backward; all Adam
updates; optional rebuild/reshuffle; completion. Both `src/Network.cu` and
`sycl/network.hpp::Network::train` invoke that controller. The CPU fixture calls
the latter public caller for two updates, a partial update, empty input and
invalid capacity. It does not replace the training control with a test loop of
individual kernel calls. A separate callback-order check covers two hidden
layers and the reshuffle-only branch.

The SYCL numerical backend currently supports one hidden ReLU layer and a
selected output layer. Its dense hidden representation is an explicit backend
choice; sparse input transfer accumulates duplicate IDs into a dense batch.
The native production caller still supports its original layer vector. This
work does not execute all original 54 host definitions or the native CUDA runtime.
A separate source-extraction adapter test executes the exact maintained
`Network::train` host body with explicitly mocked layer/runtime operations.
It checks input/label wiring, two-hidden-layer order, full-capacity reset,
within-step accumulation and rebuild/reshuffle/no-rebuild branches. This narrow
adapter evidence is distinct from the real numerical SYCL execution and does
not execute CUDA allocation or device kernels. Native CUDA build, BLAS/Thrust interoperability, ownership, stream
ordering, GPU races/performance and reset acceptance remain #34/#43 work.
No parser, dataset driver or checkpoint feature is added.

| Boundary | Ownership, shape and ordering |
| --- | --- |
| Allocation/copy/clear | Queue-context USM RAII arrays; explicit capacities; synchronous transfer/clear with `wait_and_throw`; all arrays live through submissions |
| Sparse input/labels | Validated monotone CSC offsets, endpoints, finite values and in-range IDs; partial batches accepted; training labels required per nonempty sample |
| Dense hidden IDs/forward | Actual #115 dense-ID kernel and SYCL dense product; hidden weights column major `[input,hidden]`; batch values row major |
| Index/query | Actual #115 tiled hash rebuild, independent per-query address, device lengths scan/gather and parallel frequency counts; direct-address frequency capacity `[max_batch,outputs]` |
| Activation/filter | Labels and optional minimum padding through #115; device size scan then device frequency filter; output CSC capacity `max_batch*outputs` |
| Scan/filter/sort | Deterministic reference device algorithms; scan is per-output prefix summation, key sort is rank-based with tie index only for unique write ownership; tests compare correspondence and sorted keys |
| Selected Softmax | Row-major output weights `[output,hidden]`; stable maximum normalization; label-minus-probability delta divided by actual batch |
| Backward/reset | Full owned hidden delta capacity cleared immediately before each backward; true atomic accumulation from multiple active outputs; ReLU gates preserved; gradients accumulate within a step |
| Optimizer | Zero initial gradients/moments/velocities; positive signed Adam update; caller supplies bias-corrected rate as original `main.cu` does; every gradient accumulator consumed and cleared |
| Rebuild cadence | Constructor builds once; controller rebuilds after updates when rebuild or reshuffle is requested; buckets and counts reset and buffers reused |
| Dense evaluation | Public `evaluate_dense` accepts unlabeled samples and includes learned output bias (an explicit numerical variant: native `SoftmaxLayer::forward_dense` is SGEMM-only and omits output bias); executes both hidden and row-major output products on the device; explicit tests also compare row/column layouts, bias, empty extent and partial batch |

All backend calls complete before returning and propagate asynchronous errors.
The algorithms are reference SYCL implementations shared by CPU and GPU, with
no CPU scalar substitution for a claimed device operation. The naive prefix
scan and rank sort are intentionally simple, quadratic algorithms; neither
claims Thrust performance. `dense_product` is a reference matrix product and
does not claim cuBLAS execution. Native optimized libraries are retained in the
CUDA adapter. Dense evaluation therefore validates the documented biased SYCL model, not
bitwise/numerical equivalence to that native SGEMM-only evaluation caller.
Numeric inputs are finite; dimensions/capacities are checked before
allocation, including signed integer multiplication limits. Sparse labels are
counted as the released representation does; callers should supply distinct
labels when their target probabilities should sum to one.

`tests/gslide_sycl/test_network.cpp` has an independent scalar FP64 model. It
checks sparse-to-dense transfer effects, every hidden activation, selected
probability/output delta, hidden delta, parameter, moment, velocity and cleared
gradient. The first genuine Adam update crosses the known near-tie WTA winner;
six table bucket memberships change. The second input/label batch changes
active sets. Every post-update bucket set/count is compared to the independent
index; a stale-index run changes the actual queried active set. Replaying the
same production controller while deliberately skipping reset yields hidden
delta discrepancy about `0.201329` and parameter discrepancy about `0.007291`.
That diagnostic is a negative fixture, never a Golden for correct training.

A partial-batch third update is checked against the same FP64 model. Poisoning
all hidden delta slots first proves that the real caller clears the inactive
capacity tail, and closed ReLU gates remain zero. Dense evaluation after the
update is independently checked; an empty training batch is a no-op; invalid
input capacity fails before transfer. Reshuffling executes actual key generation
and device key sorting and preserves each bin permutation. All normal
allocations are freed by RAII; this does not validate CUDA ownership under a
host sanitizer.

The native adapter now contains the same full-capacity reset as a source-level
repair. This is distinct from native GPU confirmation, so #43 stays open.
Historical extraction evidence and its stale-delta negative fixture remain
unchanged. `tools/check_gslide_sycl.py` requires both real CPU test reports,
records both binaries, all caller/kernel/fixture/source hashes, compiler command
lines, runtime/device output and numeric tolerances, and removes old success
reports before each invocation. Run it after activating the shared toolchain:

```bash
python3 tools/check_gslide_sycl.py --build-dir /tmp/gslide-sycl
python3 tools/check_gslide_train_adapter.py --build-dir /tmp/gslide-native-adapter
python3 tools/check_gslide_cpu.py --build-dir /tmp/gslide-extracted
python3 tools/validate_traceability.py
python3 tools/audit_traceability_docs.py
```
