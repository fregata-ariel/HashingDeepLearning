# Shared SYCL CPU device validation

Issue #114 establishes an execution backend for maintained shared kernels. It
runs SYCL kernels on AdaptiveCpp's actual OpenMP CPU device. Existing ordinary
C++ oracles and sequential CUDA-body adapters remain separate evidence. Passing
this backend contract does not accept a paper algorithm, GPU execution, CUDA
streams, native CUDA libraries, throughput or training accuracy.

## Pinned toolchain and supported scope

- AdaptiveCpp commit: `31db2e83903c1db9f50b03b620eb0a09224cd0eb`
  (`25.10.0+git.31db2e83.20261016`). The SHA-256 of `git archive` at this
  commit is `1f0816f8c6272334ee4c5ba8315828178c28a2bf35d82c82c07f3e3118be661a`.
- Clang/LLVM: Debian `19.1.7-3+b1`; compilation flow: `generic`; full compiler
  feature profile, LLVM SSCP and CPU acceleration enabled; runtime: OpenMP.
- Target platform: Debian 13, x86-64, distribution libstdc++/glibc. The installer
  rejects other distributions and architectures. Ubuntu runtime compatibility
  is not inferred from a successful Debian run.
- [Dependency artifact manifest](../tools/sycl-cpu-packages.tsv): 48 exact HTTPS
  URLs and SHA-256 values, derived from Debian's signed trixie package index.
  The initial index was verified with the Debian archive keyring (stable key
  fingerprint `41587F7DB8C774BCCF131416762F67A0B2C39DE4`). Every downloaded
  artifact is checked before extraction; a missing historical artifact or hash
  mismatch fails installation. CMake `4.2.3` and Ninja `1.13.0` come from PyPI.
- CPU bootstrap disables CUDA, ROCm, OpenCL and Level Zero runtime backends.
  The shared queue/kernel API can accept GPU queues, but this installation
  cannot validate one. Build an appropriate separately validated runtime before
  selecting a GPU; an unavailable GPU must raise an error.

Upstream documents [supported compilation flows](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/31db2e83903c1db9f50b03b620eb0a09224cd0eb/doc/installing.md)
and [performance limits](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/31db2e83903c1db9f50b03b620eb0a09224cd0eb/doc/performance.md).
This pin supports official LLVM 15–21 for the generic flow; that does not imply
that this repository has tested each version. The `omp.library-only` flow is a
different supported configuration and is not the required CI flow. PCUDA and
CuPBoP were not executed here; no original CUDA-source compatibility is claimed.
The backend suite does not measure subgroup collectives: this CPU reports
subgroup size 1. Its local-memory size report is a backend sentinel rather than
a measured amount of physical shared memory.

## Reproduce

On Debian 13 amd64, provide the signed distribution prerequisites used by CI:

```sh
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git python3 python3-venv g++ libc6-dev libedit2 libffi8 libxml2 libz3-4 libpfm4 libtinfo6 zlib1g libzstd1 time util-linux xz-utils
```

From the checkout:

```bash
tools/bootstrap_sycl_cpu.sh
source /workspace/.sycl-toolchain/env.sh
tools/check_sycl_cpu.sh --build-dir /tmp/sycl-cpu-evidence
```

Set `ACPP_TOOLCHAIN_ROOT` before bootstrap to choose another writable prefix,
then source that prefix's `env.sh`. Installed compiler configuration refers to
that absolute prefix; rebuilding is required after relocation. The installer
extracts verified dependency packages locally and builds the pinned compiler.
It does not replace system packages. It uses two build workers by default
(`SYCL_BUILD_JOBS` overrides this). Existing caches must pass the pinned source
identity and installed compiler/runtime checksum checks.

The activation file supplies `SYCL_TOOLCHAIN_PREFIX` and `ACPP_INSTALL_DIR`
(the final `install` directory), `SYCL_CXX`, required library paths and
`ACPP_TARGETS=generic`. A writable `ACPP_APPDB_DIR` is required for runtime JIT
cache initialization. The driver creates a writable cache in its evidence
directory and records `ACPP_VISIBILITY_MASK=omp`, `OMP_NUM_THREADS` and
`OMP_THREAD_LIMIT` (default 4). Missing compiler, runtime, CPU device, shared USM,
required work-group size or required tests fails the run. No CPU fallback or
capability skip changes such a failure into a pass.

A maintained paper suite can consume the same environment explicitly:

```bash
tools/check_sycl_cpu.sh --source-dir tests/EXPLICIT_SUITE --build-dir /tmp/paper-evidence
```

The source directory must belong to the selected checkout and contain an
AdaptiveCpp CMake project. Missing suites and zero-test CTest runs fail. Paper
acceptance fixtures and algorithms belong to their own suites; this command
does not discover them or silently skip missing targets.

## Shared interface and tested contract

`ports/sycl-common` exports the portable CMake interface target
`hashing_sycl::common`; compiler integration remains with the executable's
`find_package(AdaptiveCpp CONFIG REQUIRED)` and `add_sycl_to_target` calls.
`hashing::sycl_backend::make_queue` supports explicit CPU/GPU and default device
selection. The helper's validation policy uses an ordered queue and enables
profiling when the selected device supports it. Numerical kernels accept the
caller's queue, allowing other ordering and profiling policies later.
`require_cpu` rejects non-CPU devices. Queue metadata includes platform/runtime,
backend enum, vendor/driver, CPU/GPU type, subgroup sizes, shared USM, work-group
limits and actual queue properties. The async handler propagates exceptions;
owned execution boundaries must call `wait_and_throw()` before reporting success
and before destroying allocations.

The feature contract enforces the exact extent sequence
`0,1,31,32,33,127,128,129,1025`, with work-group width 32. Positive extents submit
multiple real work-items, including padded boundary lanes and 33 groups in the
largest case. Kernels use local scratch, local barriers, device-scope atomic
increments and a separate SYCL reduction. A scalar host loop independently
checks every output, each group sum, the global sum, visit count and allocation
guards. Integer arithmetic has exact comparisons. Zero-only and malformed
fixture mutations must be rejected, so removing device execution from the
fixture cannot pass. These are three CTests: one positive contract and two
expected rejection contracts.

## Evidence and CPU budget

Each driver run creates a fresh build directory, measures a clean and a cached
build, then measures cold and warm CTest execution. GNU time records wall time
and peak RSS. Evidence includes the checkout/tree and local patch, maintained
source/fixture SHA-256 values, dependency artifacts, compiler/runtime manifest,
compile commands/CMake cache, binary SHA-256 values, actual device/queue logs,
JUnit results, resource logs and final `result.txt`. Source hashes include all
maintained ports to bind included paper kernels as well as the explicit suite.
A failed command preserves its exit status and records `status=FAIL`; old
binaries are never used to stand in for a failed new build. Git reads trust only
the explicit checkout path through per-command configuration, covering container
checkout ownership differences without changing global trust or `HOME`. Git
failures retain full stderr in `git-stderr.log` and emit a bounded, escaped CI
error annotation. Checkout identity failures stop before compilation. Other
failures also emit a bounded, escaped error annotation identifying the first
failed stage and original exit code, with the tail of its configure/build/test
log and retained stderr. Full stderr remains in `driver-stderr.log`; the bounded
plain-text excerpt is saved in `failure-summary.txt`. `result.txt` includes the
stage and exit code even when metadata collection fails.

A local initial backend run on 2026-10-18 measured clean build 7.22 s / 303004 KiB,
cached build 0.14 s / 16932 KiB, cold tests 0.94 s / 68996 KiB, warm tests
0.15 s / 54120 KiB, and runtime initialization about 11 ms. These observations
calibrate #111's small backend contract budget; runner measurements remain
necessary. Toolchain bootstrap is a separate cold cost and cached compiler
restoration does not eliminate clean per-suite compilation. CPU sanitizers in
other jobs do not establish coverage of all generated/JIT device code.

The existing `traceability.yml` contains the required `sycl-cpu` job, with an
immutable Debian image digest, pinned artifact cache key, failing CPU contract
and evidence upload even after failure. Existing workflow count remains seven.
GPU/native CUDA/PCUDA validation is unrun and remains separate from this CPU
success. Additional paper suites are enrolled by explicit steps in this same
job as their acceptance work lands.
