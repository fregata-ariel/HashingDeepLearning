# Build and training smoke tests

This research branch keeps the imported source trees as archival snapshots and
tests them through CI. Compatibility fixes needed to probe old code are applied
only to the CI working copy unless/until a separate maintained port is created.

The workflows are:

- `.github/workflows/build-smoke.yml` — compile/import checks.
- `.github/workflows/training-smoke.yml` — tiny-data optimizer/training checks.

Automatic runs are limited to changes under `third_party/**`; both workflows
can also be launched manually.

## High-level status

| Component | Test | Result |
| --- | --- | --- |
| Original SLIDE | Ubuntu 22.04 / GCC / CMake build | **PASS** |
| Original SLIDE | tiny synthetic 1-batch training + verified parameter update | **PASS — L1 weight delta 0.0319997** |
| MONGOOSE | Python 3.10 `compileall` | **PASS** |
| MONGOOSE | Cython/C++ `lsh_lib` build + `import clsh` | **PASS** |
| MONGOOSE | learnable-hash `TripletNet`: forward, backward, SGD step, verified parameter change | **PASS** |
| Optimized SLIDE | untouched historical snapshot / modern GCC | **FAIL (known source regressions)** |
| Optimized SLIDE | CI compatibility fixes / generic GCC / 1-batch train + cleanup | **PASS** |
| Optimized SLIDE | Intel Classic 2021.10 / generic FP32 / 1-batch train + cleanup + parameter update | **PASS — L1 weight delta 0.0319998** |
| Optimized SLIDE | Intel Classic 2021.10 / AVX-512 compile | **PASS** |
| Optimized SLIDE | Intel Classic 2021.10 / AVX-512 strict finite-parameter smoke | **FAIL on 4-class fixture; NaN update, and fixture violates the kernel's 128-output block assumption (#5, #7)** |
| Optimized SLIDE | Intel Classic 2021.10 / AVX-512 BF16 compile | **PASS** |
| Optimized SLIDE | AVX-512 BF16 runtime training | **PROBED on BF16-capable runners; not yet passing strict finite-parameter checks** |
| MONGOOSE Reformer | full historical CUDA/APEX training entrypoint | **BLOCKED; see Issue #4** |

These are smoke tests, not reproduction of the paper's accuracy/performance
numbers.

## Training results

### Original SLIDE

A tiny SVM-format dataset is generated in CI:

- 2 training records,
- 40 evaluation records,
- 32 input features,
- 4 output classes,
- batch size 2,
- one epoch / one training batch.

The historical binary successfully reaches network construction, the training
loop, final evaluation, NPZ weight save, and exit code 0.

Representative output:

```text
Network Initialization takes 0.411 milliseconds
...
over all 0.25
save for layer 0
TRAINING_SMOKE_COMPLETED
```

Important teardown caveat: the original `main()` does not delete the allocated
`Network`, so this proves normal process exit but does not exercise the model
destructors. The original layer destructor also contains an unsafe repeated
delete for the softmax normalization buffer. See Issue #3.

### Optimized SLIDE: generic modern GCC

The exact archived snapshot does not build unmodified, so CI applies a small
compatibility patch only in the checkout workspace:

1. repair the invalid `DataLayerOpt` constructor introduced by the later
   static-analysis commit;
2. remove its undefined `MAX_BUFFER_SIZE` guard;
3. add the missing `<cstring>` include for `memset`;
4. change four `reserve(numRecords_)` calls to `resize(numRecords_)` before
   indexed writes;
5. remove the duplicate `delete sizesOfLayers;` in `main()`.

With those changes, generic FP32 optimized SLIDE builds on Ubuntu 22.04 / GCC,
loads the tiny dataset, completes one batch, evaluates, writes the NPZ weights,
executes `delete _mynet`, and returns exit code 0.

Representative output:

```text
Precision: FP32
Network Initialization takes 0.268 milliseconds
Data loading takes 0.098 milliseconds
...
over all 0.25
save for layer 0
OPTIMIZED_SLIDE_GCC_TRAINING_SMOKE_COMPLETED
```

This is the strongest current evidence that the optimized training path itself
is still executable once the known snapshot regressions are isolated.

## Intel compiler environment

Two official Intel container generations have been tested.

### Current image

`intel/oneapi:2026.1.0-devel-ubuntu22.04`

Contains:

- `icx`
- `icpx`

Observed compiler version:

```text
Intel(R) oneAPI DPC++/C++ Compiler 2026.1.0
```

It no longer contains `icc/icpc`.

### Historical image matching the code's intended compiler family

`intel/oneapi-hpckit:2023.2-devel-ubuntu22.04`

Contains both compiler families:

- `icx/icpx` 2023.2 era
- `icc/icpc` Classic

Observed Classic compiler:

```text
icpc (ICC) 2021.10.0 20230609
```

The image does not include CMake by default, so CI installs `cmake` and
`make` before building.

This historical image is a good reproducibility container for the optimized
SLIDE source because the repository's own `README.Intel.md` expects ICC.

## Optimized SLIDE under Intel Classic

Using the same CI-only compatibility fixes described above, the historical
Intel image successfully completed the generic FP32 training smoke:

```text
Precision: FP32
Network Initialization takes 1.637 milliseconds
Data loading takes 0.071 milliseconds
...
over all 0.175
save for layer 0
OPTIMIZED_SLIDE_INTEL_TRAINING_SMOKE_COMPLETED
```

### AVX-512

The same source then builds with:

```text
OPT_IA=ON
OPT_AVX512=ON
OPT_AVX512_BF16=OFF
```

under `icpc 2021.10`.

The hosted runner used by the successful test exposed `AVX-512F`, so this was
not compile-only: the AVX-512 binary was executed against the same tiny dataset
and completed training, evaluation, weight save, cleanup, and exit 0.

Representative output:

```text
INTEL_CLASSIC_AVX512_BUILD_COMPLETED
Runner exposes AVX-512F; executing AVX-512 training binary
Precision: FP32
Network Initialization takes 1.555 milliseconds
Data loading takes 0.088 milliseconds
...
over all 0.25
save for layer 0
INTEL_CLASSIC_AVX512_TRAINING_SMOKE_COMPLETED
```

This also confirms that the Intel/SVML-specific `_mm512_mask_exp_ps` path,
which fails to compile with GCC, is accepted and executable with the historical
Intel compiler.

### AVX-512 BF16

The source also builds successfully with:

```text
OPT_IA=ON
OPT_AVX512=ON
OPT_AVX512_BF16=ON
```

under the same `icpc 2021.10` environment:

```text
INTEL_CLASSIC_AVX512_BF16_BUILD_COMPLETED
```

The BF16 binary has not yet been executed. A runtime smoke should only be run on
a runner that explicitly exposes the required AVX-512 BF16 CPU feature.

## MONGOOSE

### Buildable native LSH library

On modern Python/Cython tooling the `lsh_lib` Cython/C++ extension builds and
imports successfully.

### Learnable-hash optimizer step

The repository's `mongoose_slide/slide_lib/triplet_network.py` has been
executed with real autograd and SGD:

- finite forward loss,
- `loss.backward()`,
- finite weight gradients,
- `optimizer.step()`,
- explicit check that the learned hash weight tensor changed.

Observed smoke-test values:

```text
loss 0.8027675747871399
weight_delta_l1 0.0774054229259491
MONGOOSE_TRIPLET_TRAINING_STEP_COMPLETED
```

This validates a real learnable-hash training step, not merely an import.

The full MONGOOSE/Reformer training entrypoint remains tied to its historical
CUDA/APEX environment: it raises without APEX, calls `.cuda()` unconditionally,
and its scheduler's SimHash implementation uses CuPy/NVRTC. That work is tracked
separately in Issue #4.

## Known source issues

- Issue #1 — invalid optimized-SLIDE constructor and undefined
  `MAX_BUFFER_SIZE` introduced in the later pinned history:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/1
- Issue #2 — `DataLayerOpt::loadData` writes by index after `reserve()`
  instead of `resize()`:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/2
- Issue #3 — original/optimized SLIDE teardown ownership and double-delete
  problems:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/3
- Issue #4 — full MONGOOSE/Reformer entrypoint hard-requires historical
  CUDA/APEX stack:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/4
- Issue #5 — AVX-512 one-step smoke produces non-finite parameters under the
  original tiny fixture:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/5
- Issue #6 — optimized SLIDE allocates gradient/Adam-state buffers without
  explicit initialization:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/6
- Issue #7 — AVX dense-forward kernel has no output-tail handling below or
  beyond 128-output blocks:
  https://github.com/fregata-ariel/HashingDeepLearning/issues/7

## What remains

The next useful runtime checks are:

1. repair the aligned-output AVX diagnostic fixture and compare AVX/scalar
   updates on a shape the historical vector kernel actually supports;
2. isolate BF16 non-finite behavior after the AVX shape and initialization
   issues are controlled;
3. add sanitizer-backed cleanup tests in `ports/` as compatibility fixes land;
4. build a CPU/reference backend for MONGOOSE scheduler hashing, or reproduce
   the historical CUDA environment, then run a complete Reformer batch;
5. keep paper-level performance reproduction separate from smoke correctness.
