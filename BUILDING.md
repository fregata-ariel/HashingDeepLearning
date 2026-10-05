# Build smoke tests

The research branch includes `.github/workflows/build-smoke.yml` to check whether
the vendored historical code still builds on a modern Linux toolchain.

These are **smoke tests**, not reproduction of the paper experiments. Dataset,
huge-page, accelerator, accuracy, and performance reproduction are separate
tasks.

## Current results

Tested on GitHub Actions `ubuntu-22.04` runners.

| Component | Probe | Result |
| --- | --- | --- |
| Original SLIDE | CMake + GCC build | **PASS** |
| MONGOOSE | Python 3.10 `compileall` | **PASS** |
| MONGOOSE | Cython/C++ `lsh_lib` build + `import clsh` | **PASS** |
| Optimized SLIDE | untouched generic GCC build | **FAIL** |
| Optimized SLIDE | untouched AVX-512 GCC build | **FAIL** |
| Optimized SLIDE | untouched AVX-512+BF16 GCC build | **FAIL** |
| Optimized SLIDE | compatibility probe, generic GCC | **FAIL later** |
| Optimized SLIDE | compatibility probe, AVX-512 GCC | **FAIL later** |
| Optimized SLIDE | compatibility probe, AVX-512+BF16 GCC | **FAIL later** |

The successful MONGOOSE extension probe installs modern build tooling with
`Cython<3` and NumPy, runs `python setup.py build_ext --inplace`, and imports
the produced `clsh` module.

## Optimized SLIDE findings

### 1. A source regression exists in the historical pinned tree

The exact RUSH-LAB-pinned tree contains:

```cpp
DataLayerOpt() numRecords_{0}, numFeatures_ {0}, numLabels_ {0} {};
```

This is invalid C++. Yong Wu's earlier lineage anchor
`85f758c631fd2e0e9f2d33f5fefb897370e4e911` contains the valid original:

```cpp
DataLayerOpt() {}
```

The invalid change was introduced by
`b6c92c5f63ebfefde95c884867f106af38b3581b` ("Static code analysis changes").
The compatibility-probe job repairs only the CI workspace; the vendored source
snapshot remains byte-for-byte unchanged.

### 2. Generic GCC gets substantially further after repairing that regression

The next blockers are ordinary source portability issues:

- `SLIDE/main.cpp`: `MAX_BUFFER_SIZE` is not declared.
- `SLIDE/srp.cpp`: `memset` is used without including `<cstring>`.

This indicates the generic code path is close to building with a modern GCC
once several small historical-source issues are repaired.

### 3. AVX-512 exposes an Intel-compiler-specific dependency

After the constructor regression is repaired, the AVX-512 path reaches
`Layer.cpp` and fails on:

```cpp
_mm512_mask_exp_ps(...)
```

Modern GCC does not provide this as a normal AVX-512 intrinsic. This is an
Intel SVML/compiler intrinsic assumption, consistent with the repository's
`README.Intel.md`, which specifies ICC >= 19.

A modern port should either:

- build this historical path with a compatible Intel compiler/runtime, or
- replace the SVML-specific exponential with a portable/vector-math
  implementation while documenting the numerical/performance implications.

### 4. AVX-512 BF16 has an additional modern-GCC type mismatch

The untouched BF16 build also fails because modern GCC types
`_mm512_cvtneps_pbh` as returning `__m256bh`, while the historical wrapper
returns `__m256i`. Adding `-flax-vector-conversions` is enough for the probe
to move past this mismatch, after which it reaches the same
`_mm512_mask_exp_ps` blocker.

This should not automatically become the permanent fix; a modern port should
use explicit, type-correct BF16 handling.

## What has not been tested yet

- execution of Original SLIDE against the Amazon-670K dataset;
- huge-page behavior and performance;
- actual execution of optimized AVX-512/BF16 instructions on a guaranteed
  AVX-512/BF16 runner;
- Intel ICC / oneAPI `icpx` builds;
- full MONGOOSE/Reformer training, which depends on historical CUDA/APEX and
  old PyTorch-era packages;
- numerical equivalence, accuracy, throughput, and memory benchmarks.

The first three source trees remain preserved snapshots. Compatibility changes
should be carried as separate patches or a separate maintained port, rather
than silently editing the archival copies.
