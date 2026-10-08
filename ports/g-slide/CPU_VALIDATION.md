# G-SLIDE: CPU-first validation checkpoint

G-SLIDE can follow the repository's standard research integration process.
G1 freezes the reference source; G2 extends selected kernel arithmetic and serial
state checks into a composed two-update CPU training fixture. Full CUDA build/train/save/teardown remains
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
| `GSLIDE-WTA-LSH-CPU` | Bin initialization, fixed-permutation WTA, insert/query/gather, partial tile, negative ties, packing alias | RNG, alternate hash kernels and parallel insertion; serial capacity/rebuild cases are covered separately below |
| `GSLIDE-CANDIDATE-COUNT-CPU` | Linked collisions, duplicate frequencies, threshold crossings, labels, empty second table | Lock contention, memory fences, random padding, pool exhaustion and Thrust filtering |
| `GSLIDE-SPARSE-FORWARD-CPU` | CSC active sets, column-major dot products, bias and ReLU across two samples | Adaptive kernel selection, cuBLAS dense path, shared-memory parallelism |
| `GSLIDE-SOFTMAX-CPU` | Row-major dot products, probabilities, multi-label batch deltas and negative-shift regression | CUDA warp/block reductions; two additional variants are covered below |
| `GSLIDE-SPARSE-GRADIENT-CPU` | Column-major first-layer gradients and bias accumulation | Parallel atomics and production backward wiring; deeper variants are covered below |
| `GSLIDE-ADAM-CPU` | Two updates, moments, velocity, signed parameter change, reset and tail bounds | Concurrent writers and production optimizer wiring; host bias correction is checked in the composition below |

## G2 combined coverage

The default command runs five suites. Definition counts include the ten common
bodies, so they overlap: sixteen distinct maintained CUDA definitions are selected.

| Suite | Definitions | Independent CPU evidence |
| --- | ---: | --- |
| `baseline` | 10 | Six G1 arithmetic/state contracts listed above |
| `backward` | 14 | Four deeper backward variants, both layouts, sparse IDs, activation gating, accumulation and two updates against dense oracles |
| `softmax` | 12 | Three variants, 45 fixtures / 72 sample checks, empty inputs, bias, labels, batches and large negative shifts against FP64 expectations |
| `candidates` | 10 | Collision counts, threshold/label filtering, sample isolation, bounded ring overwrite and stale versus rebuilt memberships |
| `training` | 11 | Two connected ReLU/selection/Softmax/backward/Adam/rebuild updates, repeated deterministically against a dense FP64 oracle |

Task details: [Softmax](../../docs/g-slide/softmax.md),
[candidates](../../docs/g-slide/candidates.md), and
[training](../../docs/g-slide/training.md). Eleven new fragment records join the
six baseline G-SLIDE records. The combined repository ledger has 48 experiments;
the native documentation audit covers 38 entries.

The training fixture observes loss `1.14521666 -> 0.890611627` and maximum
absolute arithmetic error `8.83802084e-8`. Its first actual Adam update flips a
near-tie WTA winner, changing three addresses. Removing the rebuild fails the
fixture. These are fixture observations, not paper accuracy/performance results.
Removing the maintained negative-logit maximum fix also fails the Softmax checks.

The composition explicitly resets hidden deltas on the host. Replaying the
actual backward body with stale deltas produces maximum discrepancy
`0.20132947`; production per-step reset wiring needs native confirmation and
repair in [Issue #43](https://github.com/fregata-ariel/HashingDeepLearning/issues/43).
Within-step accumulation remains intentional. Host scans, table construction,
scalar reductions and RAII teardown do not establish CUDA scheduling or ownership.

Combined five-suite plain and ASan/UBSan gates, archive identities, ledger and
code contracts passed in [PR #48 CI](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37708385020).

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

G2's selected CPU coverage and composed training fixture are complete. The next
milestone is G3 when GPU access is provided. Alternate hash/shared-memory bodies,
adaptive dispatch and production CUDA execution remain pending; the CPU fixture
does not execute `Network::train`. The durable task/PR checkpoints are in
[the orchestration record](../../docs/GSLIDE_ORCHESTRATION.md).

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
   Confirm and repair the per-step hidden-delta reset tracked in #43, preserving
   accumulation within a step and handling changed active sets.
5. Investigate CUDA ownership, including the apparent omission of
   `d_rand_node_keys`/`d_rand_nodes` from `LSH::~LSH`; this is source inspection,
   not a confirmed runtime leak. Record fixes only with evidence.

Save the milestone's logs, fixture, seed, expected values and exact commit so
the next session can continue without depending on another GPU allocation.
Large-dataset accuracy/performance reproduction is a later separate milestone.
