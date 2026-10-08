# G-SLIDE selected-body training composition

Run `python3 tools/check_gslide_cpu.py --suite training --build-dir build`.
This compiles eleven freshly extracted maintained CUDA/device definitions with
the restricted CPU adapter. It executes a connected two-sample, two-step fixture
twice. Sparse input vectors feed four hidden ReLU nodes; three WTA tables retrieve
a sparse subset of four output nodes, linked counting applies threshold two, and
absent labels are promoted. Selected Softmax produces signed, batch-normalized
deltas, row-major backward propagates hidden deltas and output gradients,
first-layer backward accumulates input gradients, and Adam updates both layers'
weights and biases. Output node 1 starts with a near tie (`0.50` versus `0.49`)
between hidden coordinates 0 and 2. Its connected first Adam update swaps the
winner, changing all three WTA addresses and the following queries. The second
step asserts that rebuilt buckets and sizes differ from the pre-update index
and verifies them against the independent oracle. A local mutation that skips
the second rebuild and reuses the old index fails this assertion.

A separate dense FP64 scalar oracle checks hidden activations, WTA addresses and
bucket contents, gathered candidates, counts and active sets, probabilities,
both layers' deltas/gradients, and all optimizer states and parameters. The
fixture checks finite actual loss, parameter changes, consumed accumulators,
closed ReLU gates and repeatable active sets. Owned CSC/table RAII invokes normal
host teardown; allocation lifetime requires hosted sanitizer validation because
local LeakSanitizer cannot inspect process tasks in this sandbox. Arithmetic
uses absolute tolerance `3e-6`; parameters use `8e-6`. The observed maximum
absolute comparison error was `8.83802084e-8`; actual loss changed from
`1.14521666` to `0.890611627`. These two losses are fixture observations, not a
convergence claim.

Host substitutions are explicit: owned allocations, dense hidden IDs/offsets,
deterministic seed-40 permutation rotations, prefix scans, linked traversal and
storage-order filtering, per-step hidden-delta resets, every-step index rebuild,
and the `main.cu` bias-corrected learning rate. The kernel coefficients and both
source JSON configurations were checked as `BETA1=0.9`, `BETA2=0.999`; the fixture
uses base learning rate `0.01`. CUDA/Thrust RNG is not executed. Historical
natural-log WTA packing is unchanged.

The reset substitution matters. `Layer::Layer` zeroes compressed deltas only at
construction; reviewed `ReluLayer::forward` and `Network::train` do not reset them
each step. `bp_rowmajor_knl` adds an existing previous delta whenever the ReLU
activation is positive. The second-step characterization replays that actual
body with the preceding step's buffer and matches fresh backward plus retained
delta, observing maximum stale discrepancy `0.201329470`. The gated node remains
zero. This records a source/selected-body lifetime gap, without silently fixing
production code. Native GPU confirmation and a focused reset-wiring follow-up
are pending.

Block size one, serial atomics and scalar reductions validate arithmetic and
serial state only. `Network::train`, native CUDA scheduling, parallel reductions,
lock contention, Thrust, cuBLAS, production allocation/teardown, launch selection,
and native per-step delta lifetime remain unexecuted. The immutable upstream
archive is verified by the driver.
