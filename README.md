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
- `ports/` — maintained/annotated working copies; raw snapshots remain untouched
- `papers/CODE_PAPER_MAP.md` — paper section → implementation index
- `SESSION_HANDOFF_2026-10-06.md` — historical first-session intent, commit-SHA map, and validation handoff
- `docs/RESEARCH_INTEGRATION_PLAYBOOK.md` — reusable procedure for adding future papers/reference implementations
- `docs/CI_ARCHITECTURE.md` — permanent CI responsibilities and temporary-diagnostic cleanup policy

The previous root layout is intentionally replaced on this branch so that this
workspace can later be transplanted into a separate repository.

## Annotation and maintenance policy

`third_party/` remains the immutable provenance snapshot. Paper annotations and
future compatibility fixes live in `ports/`, with the mapping indexed from
`papers/CODE_PAPER_MAP.md`. Doxygen/docstring annotations keep paper claims,
code-level interpretation, and measured effects separate.


## Adding another paper or reference implementation

Start with [the research integration playbook](docs/RESEARCH_INTEGRATION_PLAYBOOK.md). It captures the process
used on this branch from provenance recovery through independent-oracle tests,
traceability, maintained fixes, code contracts, strict typing, and final CI
cleanup.

Permanent CI is intentionally small. See [the CI architecture guide](docs/CI_ARCHITECTURE.md) before
adding a new workflow; diagnostics should normally be temporary and removed
after their findings become maintained regression tests.
