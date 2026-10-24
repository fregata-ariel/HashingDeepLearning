# Native NVIDIA GPU checkpoint

For the later Docker CUDA 12.6, MONGOOSE/Scatterbrain/MagicPIG component and shared
SYCL NVIDIA execution, see [the 2026-10-25 lineage checkpoint](GPU_LINEAGE_CHECKPOINT.md).
The CUDA 12.0 record below remains a distinct preserved profile.

The user resumed real GPU work on 2026-10-22 and selected G-SLIDE and GPU CI.
The local RTX 2080 Ti checkpoint passes the bounded #124/#125/#43 fixtures,
including all four required device diagnostics. This is a local execution;
GitHub runner registration and a hosted/manual Actions run have not occurred.

## Issues and accepted scope

[Parent #113](https://github.com/fregata-ariel/HashingDeepLearning/issues/113)
tracks native GPU work separately from completed CPU acceptance. The priority
is [G-SLIDE #34](https://github.com/fregata-ariel/HashingDeepLearning/issues/34),
with [dense offsets #124](https://github.com/fregata-ariel/HashingDeepLearning/issues/124),
[alternate hash #125](https://github.com/fregata-ariel/HashingDeepLearning/issues/125)
and [hidden-delta reset #43](https://github.com/fregata-ariel/HashingDeepLearning/issues/43).
[Budget #111](https://github.com/fregata-ariel/HashingDeepLearning/issues/111)
remains separate. Local checkpoint evidence does not close an Issue or replace
independent review, applicable PR checks and integration-tree verification.

| Fixture | Actual CUDA execution and independent expectations |
| --- | --- |
| Dense, #124 | 48 combinations: nodes 0/1/3/33, batch 0/1/2/33, blocks 32/64/128; exact IDs, all offsets, canaries and capacity/invalid-input rejection. Both production ReLU overloads and empty batch run. |
| Hash, #125 | 135 combinations: nodes 1/31/33/64/65, lanes 1/2/4, blocks 32/64/128, positive/negative/tied weights; two rebuilds, partial table tiles, exact unordered bucket membership/counts. Actual LSH::build selects init_hash_knl, logs its grid/shared bytes and runs production queries. |
| Reset/training, #43 | Actual Network::train, two deterministic updates; FP64 hidden activations, selected probabilities, hidden deltas, all weights/biases/Adam moments/velocities and consumed gradients. Spare hidden capacity is poisoned before each step and must reset. A separate native backward fixture confirms retained positive deltas and changed compressed IDs/gates after reset. |
| Native evaluation | cuBLAS logits and independent argmax, including exact ties choosing ID zero. Released dense evaluation omits output biases; this behavior is retained and explicit. |
| Original negative controls | Archive-verified CUDA bodies, renamed symbols only: dense terminal remains -71 versus expected 2; original hash emits 32 versus expected 64 nodes/table and omits nodes 32–63. These are failures of the archived implementation, not correct Golden values. |

Dense IDs and offsets now have independent extents. The caller validates capacity
before launch and permits zero nodes/batch. Hash staging now loads one row per
node group; the caller counts nodes*lanes work-items and matching shared bytes,
checks actual-kernel occupancy and records dispatch. Supported lane counts are
powers of two up to 128; the host rejects other layouts before launch.
The LSH destructor also frees the two owned random-node allocations.

The old CUDA 12.0 Thrust max_element path reported two initcheck reads during
native evaluation, in its tuple<float, ptrdiff_t> result-copy path. The maintained
evaluation now uses a fully initialized eight-byte value/index transform-reduce,
with the original first-index tie rule. All four diagnostics pass after this
change; no sanitizer suppression or kernel exclusion is used.

## Reproduce locally or on an explicitly provisioned runner

Prerequisites are Linux x86-64, an accessible NVIDIA GPU/driver, Python 3, CMake
3.24+, CUDA nvcc/headers/cuBLAS/Thrust, a supported CUDA host compiler, curl,
xz/tar, sha256sum, /usr/bin/time, uname and lscpu. The measured profile is Ubuntu
24.04, nvcc 12.0.140, GCC 12.4.0, CMake 3.28.3, driver 595.99.02 and SM75.
The driver API reports 13.2 compatibility; the compiled CUDA runtime is 12.0.
Neither value is substituted for the installed toolkit version.

```bash
# Both destinations must be new writable paths. Prefer paths outside the checkout.
sanitizer_root=/tmp/hashing-sanitizer-session
evidence_root=/tmp/hashing-gpu-session
trap 'rm -rf -- "$sanitizer_root"' EXIT
bash tools/bootstrap_nvidia_sanitizer.sh "$sanitizer_root"
python3 tools/check_nvidia_gpu.py --build-dir "$evidence_root" \
  --cuda-arch 75 --cuda-host-compiler g++-12 --jobs 2 \
  --compute-sanitizer "$sanitizer_root/bin/compute-sanitizer"
```

The bootstrap extracts NVIDIA's cuda_sanitizer_api 12.9.79 locally, after checking
SHA-256 `e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf`.
It retains the package license, removes the downloaded archive and cleans a failed
installation. The caller removes the extracted tools after the checkpoint.
[NVIDIA's redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)
supplies the artifact identity. This installs no system packages. The included
Compute Sanitizer reports 2025.2.1.0; the distribution's 2022.4.1 launcher could
not instrument the target on this host and was not accepted as a diagnostic pass.

The driver creates a fresh build, fails on missing tools/GPU, checks native
completion markers, and requires memcheck with full leak checking, racecheck
with zero errors/warnings, initcheck and synccheck. Nonzero exits and absent
clean summaries fail the aggregate result even after numerical success.
Six CPU-only reporting regressions guard missing tools, stale directories,
missing execution, diagnostic errors and incomplete diagnostics. They run in
the existing traceability workflow without requiring a GPU.

Evidence records the checkout/tree, tracked patch, untracked input hashes,
source/fixture/binary hashes, compiler flags/cache, actual device and dispatch,
command logs/exit codes, JUnit and build/test wall time/host peak RSS. Before/after
nvidia-smi snapshots establish context and process cleanup; they do not measure
peak GPU memory or algorithm throughput. Each command has a timeout, the native
CTest has a 120-second timeout, and the Actions job has a 20-minute limit.

## CI definition and execution record

[The GPU workflow](../.github/workflows/nvidia-gpu.yml) supports workflow_dispatch
only, with read-only repository permissions, one concurrent checkpoint, required
runner labels self-hosted/linux/x64/nvidia-gpu, retained artifacts even on failure,
and final cleanup of downloaded tools. It downloads the pinned sanitizer per run.
It has no push, pull-request or scheduled GPU trigger. Provisioning/registration
of a runner is an operational step still required before an Actions execution.
The script runs the same fixture locally and in that job.

[Machine-readable local evidence](evidence/nvidia-2026-10-22.json) retains the
final source identities and numerical/diagnostic output. Full build/log evidence
is also at /tmp/hashing-nvidia-final on this host. The source was an uncommitted
patch on feat/nvidia-gpu-validation, based on c64e5526eca284640a8f04c2e8478677a0314f90;
per-file hashes bind the tested patch. The evidence additionally records a final
workflow-only context correction after the native run; compiled source, fixture
and driver hashes are unchanged. Runner temporary paths are initialized in a
shell step using RUNNER_TEMP, following
[GitHub context availability](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).
At the time of this execution record no PR, merge or Issue closure had occurred.
Subsequent publication status is recorded in the session handoff and GitHub PR.

Final numerical result: 9226 checks, maximum absolute error 3.8112e-8, tolerance
8e-6. memcheck reports zero errors and zero leaked allocations; racecheck reports
zero hazards/errors/warnings; initcheck and synccheck report zero errors. Existing
five CPU oracle suites also passed both plain and ASan/UBSan/LSan modes; the exact
host caller adapter and source-contract/doc/ledger checks remain distinct evidence.
The pinned SYCL CPU runtime was not rerun in this Ubuntu GPU session.

## Remaining boundaries

#34 remains open: broad lock/pool/overflow stress, all adaptive/fallback branches,
nondefault streams, model checkpoint serialization and large training/quality
are beyond this bounded checkpoint. Hash coverage here fixes K=1/bin=2; released
natural-log packing and general overflow survivors are separate contracts.
The connected training fixture uses all output candidates; changed compressed
IDs are exercised separately in the backward fixture.

Shared SYCL NVIDIA compilation, same-input CUDA/SYCL comparison and a predeclared
performance-degradation threshold still need their own profile. No performance
equivalence or paper-level accuracy is inferred. MagicPIG #62, Scatterbrain #101,
MONGOOSE #121 and paid-run/TTL calibration #111 remain unaccepted here. The local
wall/RSS measurements calibrate this tiny fixture only, not those larger budgets.
