# GPU test environment

Builds and execution use the existing manual NVIDIA workflow or the same local
entry point. No scheduled GPU job, model download or runner registration is added.

```bash
# New absolute directory outside the checkout. Default target is sycl.
bash tools/run_gpu_container.sh /tmp/hashing-gpu-run-001 sycl
# Native CUDA/Torch/FlashInfer only, without compiling AdaptiveCpp:
bash tools/run_gpu_container.sh /tmp/hashing-gpu-run-002 components
# Build/cache the environment without starting GPU tests:
bash tools/run_gpu_container.sh /tmp/hashing-gpu-build-001 sycl build-only
# Retry one failed profile, retaining NOT_RUN for the other profiles:
bash tools/run_gpu_container.sh /tmp/hashing-gpu-retry-001 sycl run scatterbrain
```

The runner builds `docker/gpu/Dockerfile`, records its image ID/configuration,
then runs one NVIDIA device with four CPUs, 16 GiB host RAM, 512 processes and
an 8 GiB temporary filesystem. The process uses the caller's UID/GID, a read-only
root filesystem and checkout, no capabilities, no privilege escalation and no
network. Only the named evidence directory is a writable host bind. JIT/compiler
work goes there or into the container's temporary filesystem. `--rm` removes the
container on exit. The image remains available for subsequent tests; remove only
these task images when they are no longer needed, never prune shared Docker data.

The host must already provide Docker and a working NVIDIA Container Toolkit.
Docker daemon access and GPU permissions remain subject to host/tool approval;
the Dockerfile does not alter those policies. See the
[NVIDIA container documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

The AMD64 CUDA 12.6.3 base and Dockerfile frontend are pinned by digest. Torch,
CuPy, PyNVRTC and FlashInfer have explicit versions. Apt and transitive Python
resolutions are recorded in `/opt/os-packages.txt` and `/opt/python-packages.txt`
and copied into evidence. They are not a complete historical package mirror:
rebuilding later can resolve different transitive packages; image identity and
the recorded manifests identify the tested environment. No host Python or driver
is changed. CUDA 12.6 is a different compiler profile from the previous host's
CUDA 12.0 checkpoint.

The image retains licenses and the exact pinned Idiap source. Its original
local and causal CUDA blobs are checked before use. AdaptiveCpp uses the same
commit and Git archive SHA-256 as the required CPU environment, in a separate
prefix with the CUDA backend enabled. Compute Sanitizer is the existing verified
12.9.79 component. Downloads and intermediate build trees are removed in their
build layer where possible; installed test dependencies remain part of the image.

`result.json` enumerates each attempted profile and failures remain failures.
`PARTIAL` with exit zero means the bounded component checks succeeded, with
research acceptance still pending. The `components` image explicitly records
SYCL as `NOT_RUN`. Per-profile JSON/logs, source and binary hashes, JUnit, image
identity, dependency manifests, wall time and Torch peak allocation/reservation
remain in the output directory. Torch memory counters exclude driver/JIT/native
allocations; wall time includes import/build/JIT and is not algorithm throughput.

Profiles are distinct:

- G-SLIDE native CUDA, archived negative controls and all four required device
  diagnostics use the existing driver.
- MONGOOSE runs the unchanged archived CUDA string through CuPy/NVRTC, the
  actual CuPy/PyNVRTC production wrapper, a separate Torch GPU fallback, and
  reversible dropout backward with CUDA RNG replay. Packing includes signed
  zero, bit 31 and a nondefault stream. Single-device success says nothing about DDP/NCCL.
- Scatterbrain builds the pinned original local/causal native extensions. Local
  backward's archived padding and key-gradient failures are retained as negative
  controls. The maintained CUDA copy masks invalid cotangents, reverses the key
  window and launches on the caller's stream. Its four operations, the separate
  original causal primitive, noncausal hybrid and tiny train/save/reload have
  independent dense/FP64 expectations. The causal hybrid remains a separate boundary. The pinned
  CUDA local product requires equal physical Q/K lengths; its original unequal-
  length shape error is recorded separately from the more general CPU contract.
  All four Compute Sanitizer tools run against the numerical fixture after
  verifying that sources, fixture and binaries match its preceding plain build.
  This avoids recompilation per diagnostic; changed identities fail. G-SLIDE
  additionally requires full leak checking; the Torch-based Scatterbrain
  diagnostic profile does not make a framework/allocator leak-freedom claim.
- MagicPIG checks small dtype projections, pinned D2H/H2D transfers with explicit
  stream/events, FP16 FlashInfer paged append/plan/decode and LSE merge. This is a
  dependency fixture, not execution of the original BF16 server or model. SM75
  has no native BF16 support; successful emulated arithmetic cannot change that.
  FlashInfer 0.2.5 uses explicit batch indices/positions for append; this fixture
  adapts that dependency call and does not claim compatibility with the archived
  server's older indptr-based append call.
- Shared SYCL builds explicitly require a real NVIDIA GPU and shared USM. Existing
  G-SLIDE and MONGOOSE oracles run there. Scatterbrain's synchronous host-tensor
  ABI moves its four local kernels to that GPU; the composed Torch hybrid still
  runs on CPU. This is distinct from an all-GPU Torch bridge or original CUDA.

No quality, long-context memory budget, performance parity, broad stress or full
model-generation acceptance is inferred from these small fixtures. CUDA/SYCL
performance comparisons require matching inputs and a threshold declared before
measurement.
