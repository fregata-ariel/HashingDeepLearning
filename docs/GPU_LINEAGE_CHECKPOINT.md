# GPU lineage checkpoint — 2026-10-25

All enumerated bounded profiles passed on the local RTX 2080 Ti (SM75), driver
595.99.02. The aggregate is deliberately `PARTIAL`: original failures,
unsupported hardware paths and outstanding research acceptance remain explicit.
This is local Docker execution, not a GitHub GPU workflow run.

The [Dockerfile and runner](../docker/gpu/README.md) implement the requested test
environment. Runtime uses one GPU, bounded resources, the caller's nonroot UID,
a read-only filesystem/checkout, no network, no capabilities and a single output
bind. Docker/tool approval remains external; the image does not grant access.
The build context contains only the required Docker inputs and sanitizer script.

[Permanent evidence](evidence/gpu-lineages-2026-10-25.json) retains each report,
numerical/diagnostic logs, source and binary hashes, dependency versions, compiler
commands and image identity. The tested SYCL image is
`sha256:5c700b0d9fb9ab5a93c3ae393a7df74426301a9a0538b9e0457fcbd635ae414d`.
CUDA is 12.6.3, Torch 2.6.0+cu126, CuPy 13.6.0, PyNVRTC 9.2, FlashInfer 0.2.5,
GCC 12.4.0, LLVM 18.1.3 and AdaptiveCpp commit
`31db2e83903c1db9f50b03b620eb0a09224cd0eb`. These are separate from the
[successful host CUDA 12.0 rerun](evidence/nvidia-2026-10-23.json).
After execution, one trailing blank line in the new maintained CUDA file was
removed for the staged whitespace check. Evidence retains both hashes and the
exact LF-only reconstruction; no executable token or binary was changed.

| Profile | Actual accepted execution |
| --- | --- |
| G-SLIDE maintained native CUDA | 9,226 checks; maximum absolute error 3.8112e-8, tolerance 8e-6; existing archived dense/hash negative controls preserved. All four diagnostics clean, including full memcheck leak checking. |
| MONGOOSE original CUDA | Exact archived CUDA string compiled/loaded/launched through CuPy/NVRTC. Eight packing cases across two streams; k=1/2/31/32, signed zero and bit31, output guards. |
| MONGOOSE production wrapper | Actual CuPy/PyNVRTC wrapper with raw pointer/scalar ABI, projection/pack/transpose and a nondefault Torch stream. Separate from the passing Torch GPU fallback. |
| MONGOOSE CUDA RNG replay | Reversible dropout forward/backward matches ordinary autograd for input and parameter gradients; restores CUDA RNG state. One device, no DDP/NCCL claim. |
| Scatterbrain maintained CUDA | All four local operations, prefixes and windows 1/2/3/4/9; detached CPU FP64/scalar oracle. Maximum error 6.1201e-7. Nondefault stream, original causal primitive, actual noncausal Torch hybrid, two SGD updates, save/reload. All four diagnostics clean. |
| MagicPIG GPU dependency fixture | Tiny FP32/FP16/BF16 projection arithmetic, pinned D2H/H2D with stream/events, FP16 FlashInfer HND paged append/plan/decode/LSE merge. Decode error 2.4220e-4 against FP64, bound 2e-3. Original BF16 server/model not executed. |
| Shared SYCL NVIDIA: G-SLIDE | 11,749 kernel checks and 754 connected-production checks; training error 5.69841e-8. Required GPU queue, NVIDIA vendor, shared USM, subgroup32, local memory and completion recorded. |
| Shared SYCL NVIDIA: MONGOOSE | Five projection cases, literal packing/transpose, bit31, k rejection and real native CPU LSH consuming GPU-produced int32 keys. |
| Shared SYCL NVIDIA: Scatterbrain | Five four-operation cases, ten hybrid cases including two full-support cases, twenty mutation detections, four lifecycle cases, eight extra scalar cases and four foreign threads. Local operations run on GPU; Torch hybrid/ABI inputs remain CPU. |

The SYCL queue reports an actual NVIDIA GPU, `is_cpu=0`, `is_gpu=1`, shared USM,
subgroup32 and 49,152 bytes local memory. `HASHING_SYCL_REQUIRE_NVIDIA=ON` is an
explicit build option; the permanent default CPU profile is retained. A CPU
fallback cannot satisfy this GPU profile.

## Scatterbrain source failures and maintained repairs

The pinned original local CUDA requires equal physical sequence extents; the
first unequal-length probe failed its reshape. The maintained port rejects that
unsupported layout explicitly rather than claiming the broader CPU contract.

Two independent original backward failures remain executable negative controls:

| Contract | Original actual | Independent expected |
| --- | --- | --- |
| Padding cotangent, window1/prefix2 | Q gradient `[4, 10, 18]` | `[4, 10, 0]` |
| Reversed key window, window3 | K gradient `[14, 38, 32]` | `[10, 34, 36]` |

[The maintained CUDA copy](../ports/scatterbrain/cuda/local_product_cuda.cu)
masks invalid cotangents, reverses key-gradient indexing and uses the current
Torch stream. It accepts strided/expanded autograd cotangents and validates its
bounded dtype/layout contract. The pinned original is compiled separately;
`third_party/` is unchanged. MIT license, commit/blob identities and modifications
are retained beside the maintained source. The fixture explicitly loads this
variant under the original autograd wrapper; no external installation is patched.

Four Scatterbrain diagnostics reuse only the preceding native build after
checking fixture, source and binary hashes. They include the original controls
and causal primitive as well as the maintained hybrid. No suppressions or kernel
exclusions are used. This Torch fixture makes no allocator leak-freedom claim;
G-SLIDE's separate memcheck additionally requires zero leaked allocations.

## Limits and calibration

MagicPIG's successful BF16 arithmetic is not native BF16 support on SM75. Its
original BF16/tensor-core server remains unsupported here. FlashInfer 0.2.5's
append API requires explicit batch indices/positions; this component fixture
adapts that API and does not establish compatibility with the archived server.
AVX-512 CPU extension execution, full generation and model downloads are absent.

The pinned original causal primitive passed independently; Scatterbrain's causal
hybrid, later CPU-only lineage and general unequal-length CUDA remain separate.
SLIDE's CPU-only ports gain no GPU claim. G-SLIDE broad stress, sparse connected
selection, general packing, nondefault native streams and model serialization
retain the boundaries in the earlier checkpoint. Multi-device DDP/NCCL, paper
quality/scale, same-input CUDA/SYCL parity and performance remain outstanding.

Measured whole-profile wall times in the final image were 31.08s G-SLIDE,
1.5–1.9s per MONGOOSE profile, 138.54s Scatterbrain including five extension
compilations, 47.36s FlashInfer cold component/JIT and 33.58s all three SYCL suites.
Verified-binary Scatterbrain diagnostics took 3.6–6.8s each. Torch peak allocated
memory was about 25.4 MiB for Scatterbrain and 32.1 MiB for MagicPIG; those counters
exclude driver/JIT/native allocations. These are small-fixture costs, not warm
kernel throughput or a full-model budget. No performance ratio or degradation
threshold is inferred.

## Publication, CI and cleanup

[Draft PR #127](https://github.com/fregata-ariel/HashingDeepLearning/pull/127)
targets `research/lsh-lineage-vendor`, also the verified remote default branch.
Its incoming head `f5812af` passed all three jobs in
[run 39813806042](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/39813806042),
which tested merge candidate `680c7ff`. New commits' checks are recorded in the
PR; CPU CI success remains separate from the local GPU evidence. Independent
review and research acceptance remain pending, so the PR stays Draft. No Issue
is closed and no manual GitHub GPU job or runner setup is performed.

Superseded host venv/cache/source/tool downloads were removed; the
[cleanup record](evidence/gpu-host-setup-cleanup-2026-10-25.json) lists exact task
paths. Containers use `--rm`. Docker images are retained deliberately as the
requested reusable test environment. Raw final evidence remains at
`/tmp/hashing-docker-final-20261025`; permanent JSON does not depend on its lifetime.
