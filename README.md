# LSH / SLIDE lineage research workspace

This branch vendors three related LSH-for-neural-training codebases as ordinary
source files. They are **not Git submodules**, and their upstream Git metadata is
not embedded.

## Layout

- `third_party/slide-original/` — original SLIDE implementation (MLSys 2020)
- `third_party/slide-optimized-avx512/` — modern-CPU optimized SLIDE (MLSys 2021)
- `third_party/mongoose/` — MONGOOSE learnable-LSH framework (ICLR 2021)
- `papers/SOURCES.md` — paper/repository provenance and pinned commits
- `papers/slide-2020-existing-notes/` — pre-existing SLIDE paper notes/images,
  relocated unchanged from this repository
- `SOURCE_MANIFEST.tsv` — machine-readable source manifest

The previous root layout is intentionally replaced on this branch so that this
workspace can later be transplanted into a separate repository.

## Intended next pass

Map paper sections to implementation units and annotate the code. Use Doxygen
blocks for C/C++ and Python docstrings for Python. Each annotation should keep
three things separate: the algorithmic role, the concrete implementation
mechanism, and any speed/memory/accuracy effect actually reported by the paper.
Code-derived inference should be labeled as inference rather than a paper claim.
