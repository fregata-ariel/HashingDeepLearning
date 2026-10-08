# G-SLIDE: CPU-first validation checkpoint

G-SLIDE can follow the repository's standard research integration process.
The first milestone freezes the reference source and verifies selected kernel
arithmetic and serial state on CPU. Full CUDA build/train/save/teardown remains
pending. CPU emulation is the primary routine test; GPU environments are used
only when the user makes them available, at milestones.

## Provenance and maintained changes

- Paper: [TPDS record](https://ieeexplore.ieee.org/document/9635657), DOI
  `10.1109/TPDS.2021.3132493`, volume 33(11), pages 3015–3027 (2022).
- Mechanism review: [author-hosted manuscript](https://panzaifeng.github.io/assets/pdf/tpds22gslide.pdf).
  This is an author version, not a locally vendored publisher PDF.
- Repository: [PanZaifeng/G-SLIDE](https://github.com/PanZaifeng/G-SLIDE).
- Pin: `d93c2f6d0bbf1dd7b96d9c2340ed629c29f4f902`.
- Upstream tree: `7271376329b09ce2b86e40f3b190b387d8abe376`.
- License: MIT. All 27 upstream files, including the license, are stored
  unchanged under `third_party/g-slide/`; the pin manifest verifies exact
  Git blob identities and the complete archive file set.
- Maintained `ports/g-slide/` starts from that snapshot. CUDA-source comments
  document tested contracts. The sole arithmetic change is
  `MAX_INIT: 0.0 -> -FLT_MAX` in `src/kernel.cu`.

The Softmax regression reproduced an upstream failure: shifting the two test
logits by -1000 yields probability 0 instead of approximately 0.970688 for
the first active output. Starting the maximum below finite logits restores
stable normalization. The identity also affects padded lanes in `block_max`;
actual warp execution still needs GPU validation.

WTA packing remains an explicit variant. The source uses `floor(ln(bin_size))`
while Section 4.3 / Figure 6 describes `ceil(log2(bin_size))` bit widths.
With bin size 8 and K=2, tuples (0,4) and (1,0) both address bucket 4.
The CPU test characterizes the released behavior; changing packing would
require coordinated address/allocation changes and a separate decision.

## Routine CPU checks

From the repository root, Python 3.10+ and GCC with C++14 are sufficient.
No CUDA toolkit, GPU, cuBLAS, Thrust, datasets or Python packages are required.

```bash
python3 tools/check_gslide_cpu.py --build-dir /tmp/gslide-cpu/plain
python3 tools/check_gslide_cpu.py --sanitize --build-dir /tmp/gslide-cpu/sanitized
python3 tools/audit_traceability_docs.py
python3 tools/validate_traceability.py
```

The baseline driver extracts ten definitions from maintained CUDA sources at run time
and uses the actual data-structure headers, loop macros and optimizer
constants. Independent oracles live in
`tests/traceability/test_gslide_cpu_emulation.cpp`. Generated translation units
and `gslide_cpu_evidence.json` record the selected definition hashes, compiler
command/version, sanitizer mode and explicit `gpu_validated: false`.

Independent task suites are enrolled by
`tests/traceability/gslide_suites/<name>.json` and run by default, in separate
translation units. Select one with `--suite <name>`. Each descriptor declares
`name`, `test_file` and extra `definitions` by source; the baseline definitions
remain available. Evidence lives under `<build-dir>/<name>/`; the root summary
lists every executed suite. Experimental ledger fragments under
`papers/traceability/*.json` join the baseline ledger before validation,
documentation audit and artifact generation. Duplicate IDs fail validation.
Every `g-slide-2022` experiment whose ID ends in `-CPU` must map to an enrolled
suite file, and each suite must have a CPU ledger mapping. A descriptor omission
therefore fails before compilation. The exact resolved test file is included
and hashed; a same-basename file cannot be substituted. Old success summaries
are invalidated before each invocation, including failed/interrupted runs.

| Traceability ID | CPU evidence | Remaining boundary |
| --- | --- | --- |
| `GSLIDE-WTA-LSH-CPU` | Bin initialization, fixed-permutation WTA, insert/query/gather, partial tile, negative ties, packing alias | RNG, alternate hash kernels, bucket overflow/order and parallel insertion |
| `GSLIDE-CANDIDATE-COUNT-CPU` | Linked collisions, duplicate frequencies, threshold crossings, labels, empty second table | Lock contention, memory fences, random padding, pool exhaustion and Thrust filtering |
| `GSLIDE-SPARSE-FORWARD-CPU` | CSC active sets, column-major dot products, bias and ReLU across two samples | Adaptive kernel selection, cuBLAS dense path, shared-memory parallelism |
| `GSLIDE-SOFTMAX-CPU` | Row-major dot products, probabilities, multi-label batch deltas and negative-shift regression | CUDA warp/block reductions and other Softmax variants |
| `GSLIDE-SPARSE-GRADIENT-CPU` | Column-major first-layer gradients and bias accumulation | Deeper-layer deltas, parallel atomics and complete backward wiring |
| `GSLIDE-ADAM-CPU` | Two updates, moments, velocity, signed parameter change, reset and tail bounds | Host bias correction, concurrent writers and complete optimizer wiring |

This is a restricted serial adapter, not a general CUDA simulator. Shared
memory kernels run with one thread per block; barriers assert that restriction.
Atomics are serial read/modify/write operations, `block_max`/`block_reduce` are
scalar adapters, allocations use host arrays, and host `exp`/division replace
CUDA intrinsics. Thrust scan boundaries are supplied by the fixture. The
original warp bodies, allocation APIs, launch syntax and scheduling are not
executed. Passing this test does not prove full G-SLIDE runs on CPU or GPU,
race freedom, leak-free CUDA ownership, paper accuracy or GPU speedup.

The existing `traceability.yml` runs plain and ASan/UBSan CPU checks and uploads
their source/compiler evidence. Its documentation audit now includes `.cu`
and `.cuh`. There is no recurring GPU job and no extra diagnostic workflow.
G-SLIDE has no maintained Python sources; the 31-file strict ports scope stays
unchanged. ASan/UBSan cover the emulated host allocations and selected bodies,
not the untouched CUDA allocation/free paths. Local managed environments may
block LeakSanitizer process inspection; the hosted CI run is the sanitizer
completion gate.

## Next milestones

Continue CPU coverage with independently expected values for remaining sparse
backward and kernel variants, then compose a tiny deterministic training
fixture. Keep the full production CUDA baseline pending until GPU access is
provided; a CPU fixture must not be presented as `Network::train` execution.

When an infrequent Hugging Face or Colab session is available, collect the
exact commit, `nvidia-smi`, `nvcc --version`, architecture and build log first.
The upstream CMake requires CUDA, cuBLAS/Thrust and JsonCpp (FetchContent
1.9.5); the README's historical environment uses nvcc 11.1 and RTX 2080 Ti.
Modern toolkit compatibility is unverified.

1. Configure/build the maintained CUDA target and capture toolchain changes.
2. Run fixed-permutation selected kernels on GPU against these CPU oracles;
   compare sets when parallel insertion order is unspecified.
3. Exercise multi-warp blocks, tail lanes, adaptive shared-memory dispatch,
   contention, bounded bucket overflow and capacity limits.
4. Run one deterministic train/update/rebuild/save/teardown fixture with
   observable parameter changes; use Compute Sanitizer for device memory,
   race and synchronization diagnostics where supported.
5. Investigate CUDA ownership, including the apparent omission of
   `d_rand_node_keys`/`d_rand_nodes` from `LSH::~LSH`; this is source inspection,
   not a confirmed runtime leak. Record fixes only with evidence.

Save the milestone's logs, fixture, seed, expected values and exact commit so
the next session can continue without depending on another GPU allocation.
Large-dataset accuracy/performance reproduction is a later separate milestone.
