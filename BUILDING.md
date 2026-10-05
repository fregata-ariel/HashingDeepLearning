# Build smoke tests

The research branch includes `.github/workflows/build-smoke.yml` to check whether
the vendored historical code still builds on a modern Linux toolchain.

The matrix intentionally separates:

- Original SLIDE: ordinary GCC/CMake build.
- Optimized SLIDE: generic, AVX-512, and AVX-512+BF16 configurations.
- MONGOOSE: Python syntax compilation and the Cython/C++ `lsh_lib` extension.

These are smoke tests, not reproduction of the paper experiments. Dataset- and
GPU-dependent training runs are out of scope for this first pass.

Expected follow-up is to document any failures as compatibility gaps between
the historical environment and current compilers/libraries, then patch them in
separate commits without modifying the preserved source snapshots.
