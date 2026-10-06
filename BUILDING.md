# Build, training, and correctness status

This branch separates provenance from maintenance:

- `third_party/` contains byte-for-byte imported historical snapshots.
- `ports/` contains the annotated and maintained working copies.
- compatibility fixes are never silently folded back into `third_party/`.

The current maintained SLIDE ports build, execute a real training step, save
weights, and complete teardown. The optimized port has also been exercised on
AVX-512 and AVX512-BF16 hardware.

## Current status

| Component | Check | Result |
| --- | --- | --- |
| Original SLIDE archive | modern GCC/CMake build | **PASS** |
| Original SLIDE port | tiny one-batch train → eval → save → teardown | **PASS** |
| Original SLIDE | verified parameter update | **PASS — L1 delta 0.0319997** |
| Original SLIDE port | ASan + UBSan one-batch teardown | **PASS** |
| Optimized archive | untouched historical snapshot / modern GCC | **FAIL as archived; historical defects are preserved** |
| Optimized port | modern GCC scalar FP32 train → save → teardown | **PASS** |
| Optimized port | whitespace-only config line | **PASS** |
| Optimized port | ASan + UBSan scalar one-batch teardown | **PASS** |
| Optimized port | Intel Classic 2021.10 scalar FP32 | **PASS** |
| Optimized port | AVX-512 FP32, deterministic 128-output comparison | **PASS** |
| Optimized port | scalar vs AVX saved-weight comparison | **PASS — max abs diff 7.45e-09** |
| Optimized port | AVX-512 FP32 with 4-output tail | **PASS** |
| Optimized port | AVX512-BF16 mode 1: BF16 activation + FP32 master weight | **PASS** |
| Optimized port | AVX512-BF16 mode 2: BF16 activation + BF16 weight | **PASS** |
| MONGOOSE | Python syntax + native Cython/C++ LSH extension | **PASS** |
| MONGOOSE | learnable-hash backward + optimizer step | **PASS — weight delta observed** |
| MONGOOSE Reformer full historical entrypoint | CUDA/APEX/CuPy/NVRTC stack | **BLOCKED — Issue #4** |

These are correctness smoke tests, not reproduction of the papers' benchmark
accuracy or throughput.

## Original SLIDE

A synthetic SVM-format fixture uses two training records, forty evaluation
records, 32 input features, four classes, batch size two, and one training
batch.

The historical code already built on a current GCC toolchain. A strengthened
probe confirmed that the batch changes parameters rather than merely traversing
the training loop:

```text
SMOKE_WEIGHT_DELTA_L1 0.0319997
```

The maintained port additionally fixes teardown ownership. Its current CI
passes train → evaluation → NPZ save → destructors → exit 0.

ASan/UBSan subsequently exposed a first-layer backpropagation OOB read caused by
evaluating `_hiddenlayers[j - 1]` when `j == 0`. That was fixed in Issue #10,
and the sanitizer rerun is green.

## Optimized SLIDE: historical archive vs maintained port

The exact RUSH-LAB-pinned archive is intentionally preserved with its historical
defects. The maintained port fixes them as separate, reviewable commits.

Resolved maintained-port issues include:

- #1 invalid `DataLayerOpt` constructor and undefined `MAX_BUFFER_SIZE`;
- #2 metadata vectors indexed after `reserve()` instead of `resize()`;
- #3 ownership / teardown / duplicate and interior-pointer frees;
- #6 uninitialized gradient and Adam-state buffers;
- #7 missing AVX dense-forward output-tail handling;
- #8 missing `<cstring>` for `memset`;
- #9 config `trim()` failure on whitespace-only lines;
- #10 first-layer `_hiddenlayers[-1]` access;
- #11 non-conforming `aligned_alloc` byte counts.

All of those Issues are now closed after CI validation. The archive remains
unchanged.

## Intel compiler environment

The current official image tested was:

```text
intel/oneapi:2026.1.0-devel-ubuntu22.04
```

It contains `icx/icpx 2026.1`, not the old Classic compiler.

For historical optimized-SLIDE reproduction we use:

```text
intel/oneapi-hpckit:2023.2-devel-ubuntu22.04
icpc (ICC) 2021.10.0 20230609
```

The old image does not ship CMake, so CI installs CMake and Make before the
build. This compiler family matches the repository's original Intel build
expectation and accepts the Intel/SVML-specific vector-math intrinsics used by
the optimized source.

## Deterministic scalar ↔ AVX-512 correctness comparison

Actions run:
https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37440311849

The comparison uses:

- 128 outputs, satisfying the historical vector block shape;
- the same deterministic weight initialization seed;
- explicit zero initialization of gradient and Adam state;
- identical tiny training/evaluation data;
- Intel Classic 2021.10;
- one scalar FP32 build and one AVX-512 FP32 build.

Sample 1 produced:

```text
scalar:
SMOKE_NONFINITE_WEIGHTS 0
SMOKE_WEIGHT_DELTA_L1 1.02369
SMOKE_WEIGHT_SUM -0.599894

AVX-512:
SMOKE_NONFINITE_WEIGHTS 0
SMOKE_WEIGHT_DELTA_L1 1.0237
SMOKE_WEIGHT_SUM -0.599901
```

The saved FP32 weights compare as:

```text
SCALAR_AVX_MAX_ABS_DIFF 7.450580596923828e-09
SCALAR_AVX_MEAN_ABS_DIFF 1.6711458883378327e-09
SCALAR_AVX_L1_DIFF 6.845013558631763e-06
SCALAR_AVX_FP32_NUMERICAL_COMPARISON_PASS
```

The same workflow passed all three sampled jobs.

This result explains the earlier AVX NaN smoke: it combined uninitialized
optimizer/gradient state (#6) with a four-output fixture that violated the
historical 128-output vector-loop assumption (#7). Under initialized,
shape-valid conditions there is no remaining evidence from this test that the
Intel AVX arithmetic itself is unstable.

## AVX output-tail fix and BF16 runtime

The maintained port preserves the original 8x16 unrolled kernel for complete
128-output blocks and adds masked AVX-512 handling for remaining output lanes.

Dedicated run:
https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37442222359

A hosted runner exposing both `avx512f` and `avx512_bf16` executed the
maintained four-class / four-output fixture:

```text
PORT_AVX_FP32_4_OUTPUT_TAIL_PASS
PORT_BF16_MODE_1_4_OUTPUT_TAIL_PASS
PORT_BF16_MODE_2_4_OUTPUT_TAIL_PASS
```

All three modes completed training and NPZ weight save without a NaN in the
logged output. This validates the maintained tail path for the small fixture
that originally exposed Issue #7.

## Maintained-port smoke and sanitizers

Normal maintained-port smoke:
https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37441726370

- Original SLIDE: build, train, save, destructors, exit 0.
- optimized GCC scalar: build, whitespace-config parse, train, save, destructors,
  exit 0.

Sanitizer rerun:
https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37443082986

Both Original and optimized scalar ports pass one-batch training and teardown
under ASan + UBSan with halt-on-error enabled. Leak detection is intentionally
disabled in this first sanitizer gate; invalid memory access and undefined
behavior are fatal.

## MONGOOSE

The native `lsh_lib` Cython/C++ extension builds and imports on a modern
Python toolchain.

The learnable hash projection in
`mongoose_slide/slide_lib/triplet_network.py` has also completed a real
autograd update:

```text
loss 0.8027675747871399
weight_delta_l1 0.0774054229259491
MONGOOSE_TRIPLET_TRAINING_STEP_COMPLETED
```

The full Reformer training entrypoint remains the only open compatibility item:
Issue #4. It assumes the historical CUDA/APEX stack, calls `.cuda()`
unconditionally, and routes scheduler hashing through CuPy/NVRTC.

## CI files

- `.github/workflows/build-smoke.yml` — archival build/import probes.
- `.github/workflows/training-smoke.yml` — archival/diagnostic training probes.
- `.github/workflows/optimized-correctness.yml` — deterministic scalar/AVX and
  BF16 correctness matrix.
- `.github/workflows/ports-smoke.yml` — maintained-port build/train/teardown.
- `.github/workflows/ports-avx-probe.yml` — samples hosted runners and executes
  maintained AVX/BF16 paths only when the CPU exposes the needed features.
- `.github/workflows/ports-sanitizers.yml` — ASan/UBSan maintained-port gate.

Older diagnostic workflows are retained because they document how the AVX/BF16
failures were isolated.

## Next work

The C++ SLIDE correctness/maintenance baseline is now strong enough to proceed
with paper-linked documentation and larger reproduction work without mixing
archival defects into the analysis.

The main unresolved engineering item is MONGOOSE Issue #4. The next practical
choice is either to reproduce its historical CUDA/APEX environment or add a
maintained CPU/reference scheduler-hash backend plus modern optional AMP/device
handling, then run a complete Reformer training batch.
