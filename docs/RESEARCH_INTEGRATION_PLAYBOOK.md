# Research integration playbook

This document defines the maintenance process for adding another research paper
and its reference implementation to this repository.

It is distilled from the work performed on
`research/lsh-lineage-vendor`, which started from merge base
`6f8558da001d7f177b51f9917551d7374fcfd39a` and integrated Original SLIDE
(MLSys 2020), Optimized SLIDE (MLSys 2021), and MONGOOSE (ICLR 2021).

The objective is not merely to make historical code compile. The objective is
to leave an auditable chain:

> paper claim / prerequisite → reference source → maintained implementation →
> independent test → code contract → CI evidence

That chain should be reproducible for every future paper added here.

## 1. Repository invariants

### 1.1 `third_party/` is the provenance archive

A vendored upstream snapshot belongs under `third_party/<project>/`.

Rules:

- pin an upstream commit or otherwise identify the exact recovered tree;
- record repository URL, commit/tree identity, license, and lineage notes;
- do not apply portability fixes, commentary, formatting, or refactors;
- do not make CI success on `third_party/` a permanent maintenance goal;
- use it to answer: "what did the selected reference snapshot contain?"

When the canonical repository is unavailable, document the recovery chain. Do
not silently substitute an arbitrary fork HEAD.

### 1.2 `ports/` is the maintained layer

Create `ports/<project>/` as the maintained copy.

This is where we allow:

- portability and correctness fixes;
- deterministic test hooks and small pure helpers;
- Doxygen/docstring contracts;
- strict Python typing;
- native boundary stubs;
- sanitizer fixes;
- CPU/reference paths;
- hardware-safe fallbacks.

A behavior-changing fix should be paired with a regression test whenever
practical.

### 1.3 Paper files and provenance live under `papers/`

Maintain:

- `papers/SOURCES.md` — human-readable provenance;
- local PDFs when legally/technically obtainable;
- `papers/traceability.json` — machine-readable paper/code/test ledger;
- `papers/CODE_PAPER_MAP.md` — higher-level human map.

When a paper PDF is local, CI records `sha256sum`. If a primary source cannot
be downloaded non-interactively, record that fact; do not replace it with
slides, reconstructed text, or another version without saying so.

## 2. The integration lifecycle

The recommended order matters. Most wasted effort in historical research code
comes from annotating or refactoring before behavior is understood.

### Phase A — identify lineage and freeze sources

1. Identify the paper version to study.
2. Identify the canonical code repository and commit.
3. If the canonical repository is gone, reconstruct lineage using surviving
   mirrors, aggregator pins, Git ancestry, or exact tree hashes.
4. Vendor the source under `third_party/`.
5. Record provenance in `papers/SOURCES.md` and the source manifest.
6. Copy the source into `ports/` before maintenance work starts.

The first SLIDE/MONGOOSE structural import on this branch was commit
`7213080521332220208cccd9ef84de997512c49e`; the explicit archive/port
separation was established by
`7513e846ff478baf7db1189b006857f038e07a13`.

### Phase B — establish a real execution baseline

Do not stop at "it compiles".

Create tiny deterministic fixtures that can prove:

- configure/build;
- one forward path;
- one backward/optimizer step where the paper is about training;
- observable parameter change;
- save/serialization if the implementation normally saves;
- normal teardown;
- sanitizer-clean teardown for native code.

For historical SIMD/GPU code, distinguish:

1. source defect;
2. fixture assumption;
3. compiler/toolchain requirement;
4. hardware requirement.

During the SLIDE work, apparent AVX NaNs were decomposed into uninitialized
optimizer state and a 128-output-tail assumption before scalar-vs-AVX
arithmetic was judged.

### Phase C — use temporary diagnostics aggressively, but temporarily

A diagnostic workflow is allowed to:

- patch a throwaway workspace;
- instrument historical source;
- compare scalar and vector paths;
- sample hardware capabilities;
- test a zero-initialization hypothesis;
- probe an old compiler/container.

It is not a permanent gate.

Once the question is answered:

1. move any real fix into `ports/`;
2. add a maintained regression test;
3. record the useful Action run/commit in a handoff or Issue;
4. delete the diagnostic workflow.

Historical examples on this branch included BF16 runner probes, aligned AVX
A/B comparisons, zero-init diagnostics, and CI-only compatibility patches.

## 3. Organize verification by concept coverage

Before writing many tests, classify what must be understood.

Use four coverage classes:

- **direct** — mechanism explicitly described/proposed by the paper;
- **prerequisite** — algorithm/math needed for the paper to make sense
  (LSH family, Softmax, Adam, BF16 representation, Reformer hashing, etc.);
- **variant / approximation** — released code differs from the displayed paper
  algorithm or uses a practical proxy;
- **support** — FIFO storage, CLI wiring, Cython ownership, parsing, optimizer
  infrastructure, build adapters, etc.

Issues should be organized around conceptual coverage, not files.

For the SLIDE/MONGOOSE pass, this became milestones for provenance,
Original SLIDE, Optimized SLIDE, MONGOOSE, and final contracts/typing.

## 4. Traceability ledger

Each experimentally verified relationship gets a stable ID in
`papers/traceability.json`.

A record should contain at least:

```json
{
  "id": "PAPER-MECHANISM-ID",
  "paper": "paper-key",
  "paper_location": "Section X; Equation Y; Algorithm Z",
  "mechanism": "What is being verified",
  "relation": "direct",
  "implementation": {
    "file": "ports/project/path/file.cpp",
    "symbol": "Class::method"
  },
  "test": {
    "file": "tests/traceability/test_name.cpp",
    "id": "PAPER-MECHANISM-ID"
  },
  "conditions": {
    "oracle": "independent expected-value description"
  },
  "note": "Important implementation difference or scope statement"
}
```

The relation and the test result are deliberately separate:

- a test can pass for an implementation **variant**;
- passing a test does not upgrade that variant into a direct paper
  implementation;
- unsupported theoretical claims or performance claims should remain explicit,
  not be forced into unit-test success.

Use `tools/validate_traceability.py` to catch stale symbols/test IDs.

## 5. Test design

### 5.1 Prefer independent oracles

Strong tests compute expected results independently of the implementation under
test.

Examples:

- explicit bucket sets rather than "implementation A equals implementation B";
- dense scalar `W*x+b` versus sparse/vector implementation;
- hand-computed Adam moment/state update;
- scalar BF16 round-to-nearest-even reference versus native instruction;
- explicit set model versus native LSH insert/query/remove;
- AST/config propagation checks for CLI wiring.

Scalar-vs-AVX agreement is useful, but is weaker if both paths share the same
bug. Whenever possible, compare both to a third oracle.

### 5.2 Separate layers of behavior

Do not create one giant test for "SLIDE works".

Prefer separate tests for:

1. hash component semantics;
2. hash composition / table addressing;
3. table insert/query;
4. candidate aggregation;
5. sparse arithmetic on a fixed active set;
6. optimizer state;
7. maintenance/rebuild;
8. full Layer/Network integration.

This localizes failures and makes code comments defensible.

### 5.3 Characterization versus correctness

Sometimes the released snapshot behaves differently from the paper.

In that case:

- first write a characterization test for the snapshot;
- mark the relation `variant` or `approximation`;
- only then decide whether the maintained `ports/` copy should be corrected;
- pair the correction with a regression test.

Never make a test silently redefine the paper.

### 5.4 Hardware-dependent tests

Do not pretend GitHub-hosted runner hardware is fixed.

Instead:

- print `uname -a`, `lscpu`, CPU flags, and `GITHUB_SHA`;
- gate AVX-512/BF16 execution on runtime capability;
- treat a capability-based skip as a skip, not a pass;
- keep scalar/reference tests hardware-independent;
- use a separate maintained-port smoke when historical compiler behavior must
  still be exercised.

## 6. When to modify the maintained code

Change `ports/` only after a failing/characterizing test makes the contract
clear.

A good fix sequence is:

1. minimal failing fixture;
2. independent oracle;
3. smallest implementation fix;
4. regression run;
5. sanitizer/native-boundary run when relevant;
6. traceability entry;
7. comment/type contract.

Refactoring comes after behavior is known. If several implementations look
similar, first prove behavioral equivalence with cross-implementation tests.

## 7. Code documentation conventions

Comments must keep four things separate:

1. paper claim;
2. reference/maintained implementation behavior;
3. our inference or maintained-port deviation;
4. our test/reproduction evidence.

### 7.1 C/C++ Doxygen structure

Use only sections that apply, but prefer this shape:

```cpp
/**
 * @brief One-sentence implementation responsibility.
 *
 * @par Paper mapping
 * Paper name, section/equation/algorithm.
 *
 * @par Ownership / preconditions
 * Borrowed vs owned pointers, valid lengths/shapes, lifetime constraints,
 * required hardware/layout/compiler assumptions.
 *
 * @par Implementation note
 * Snapshot-specific behavior or maintained-port design.
 *
 * @par Traceability relation
 * Direct / Variant / Approximation / Support.
 * TRACE_TEST_ID: PAPER-MECHANISM-ID.
 *
 * @warning Any known limitation that materially changes interpretation.
 */
```

For raw pointers, always answer:

- who allocates it?
- who frees it?
- can the callee retain it?
- what operation invalidates it?

### 7.2 Python docstring structure

State:

- responsibility;
- Tensor/NumPy shape and dtype when typing cannot express them;
- ownership/mutation for arrays/native buffers;
- paper mapping;
- direct/variant/approximation/support distinction;
- traceability ID.

Use type hints for Python-level shape-independent contracts and docstrings/tests
for Tensor shape/dtype/device/gradient ownership.

### 7.3 Do not attach paper speedups to functions

Paper-level throughput or accuracy improvements belong to paper experiments.
Do not write "this function gives 3x speedup" unless the paper measured that
function in isolation.

## 8. Strict Python typing strategy

Type code after its behavioral role is understood.

Rules used here:

- `mypy.ini` uses `strict = True`;
- do not use repository-wide `ignore_missing_imports`;
- do not spray `Any` through dynamic/native boundaries;
- model boundaries with small `.pyi` stubs, `Protocol`, `TypeAlias`,
  `TypedDict`, overloads, and local casts;
- keep dynamic casts at the boundary, not throughout business logic;
- audit every maintained `def`, including nested functions;
- auto-discover maintained Python files so new files cannot silently escape
  strict typing.

The final gate is `tools/audit_python_annotations.py` plus all-ports mypy in
`.github/workflows/python-typing.yml`.

## 9. Native/Cython boundaries

For every native call, document and test:

- dtype;
- shape;
- contiguity;
- ownership/lifetime;
- whether the callee copies or borrows;
- return container ownership;
- mutation of caller-owned buffers.

Keep `.pyx` and `.pyi` synchronized. If a boundary test proves that the
implementation contradicts the stub, fix one or the other explicitly; do not
weaken typing to hide it.

## 10. Issue and milestone workflow

A practical hierarchy is:

1. provenance / traceability;
2. paper A conceptual coverage;
3. paper B conceptual coverage;
4. paper C conceptual coverage;
5. documentation / typing / final gates.

For large typing/documentation milestones, split work so one Issue can normally
be completed in one session. The M5 split into Issues #23–#31 is a useful
pattern.

An Issue should include:

- coverage class;
- paper location;
- implementation scope;
- oracle/test requirement;
- "done when" checklist;
- completion evidence with Action run and commit SHA.

Close an Issue only after its CI evidence is green.

## 11. Permanent CI architecture

See `docs/CI_ARCHITECTURE.md`.

The permanent rule is:

> CI protects maintained contracts. Diagnostics investigate questions.

If a workflow is only answering a historical question and its result has been
converted into a maintained test, remove it.

## 12. Branch evolution: what happened here

This branch is a useful case study, not a template to copy commit-for-commit.

### A. Vendoring and provenance

- `7213080521332220208cccd9ef84de997512c49e` — vendor the three source
  lineages.
- source provenance and `SOURCE_MANIFEST.tsv` established.

### B. Build/training and transient diagnostics

Early workflows proved real training and decomposed AVX/BF16 failures:

- build smoke;
- one-step training smoke;
- historical Intel Classic probes;
- BF16 runner sampling;
- zero-init hypothesis;
- aligned scalar/AVX A/B;
- deterministic scalar/AVX comparison.

Useful selected commits include
`af7fff92d590a158d0fa4cef8c9294a22342972c`,
`35eaf96b1388f1542cc21335a6fd11e687ee4946`,
`62692a2b69df47653f6fc8b8becd46765421029e`, and
`1eb5da25889d8a57cd7ca671c7238010beca9637`.

These diagnostics should not be the final repository shape.

### C. Archive/port split

- `7513e846ff478baf7db1189b006857f038e07a13` created the explicit
  provenance-vs-maintenance separation.

### D. Maintained correctness fixes

Issues #1–#11 converted historical defects into maintained fixes and smoke /
sanitizer evidence.

### E. Paper-linked verification

The next stage added:

- local paper sources where obtainable;
- `papers/traceability.json`;
- independent numerical/structural oracles;
- concept-coverage Issues and milestones;
- Doxygen/docstrings linked to test IDs.

### F. Strict contracts

Finally:

- native boundary stubs;
- all maintained Python modules under strict mypy;
- automated annotation audit;
- automated C/C++/Cython documentation audit;
- traceability validation inside the final gate.

### G. Cleanup

Temporary diagnostic workflows were removed after their findings were captured
by maintained tests. Git history and recorded Action run IDs preserve the
investigation without making every historical experiment a permanent CI cost.

For the detailed first-session history, see
`SESSION_HANDOFF_2026-10-06.md`.

## 13. Checklist for the next paper

Before declaring another paper/reference implementation integrated:

- [ ] primary paper/version identified;
- [ ] exact source snapshot and license recorded;
- [ ] `third_party/` provenance copy present;
- [ ] `ports/` maintained copy present;
- [ ] build baseline established;
- [ ] real execution/training path established;
- [ ] sanitizer/native boundary checked where applicable;
- [ ] direct/prerequisite/variant/support coverage outlined;
- [ ] traceability records use stable test IDs;
- [ ] independent-oracle tests exist for central mechanisms;
- [ ] hardware-dependent paths have capability-gated tests;
- [ ] fixes are limited to `ports/` and regression-tested;
- [ ] Doxygen/docstrings document ownership and relation strength;
- [ ] Python code is enrolled in strict typing;
- [ ] native stubs match runtime dtype/shape/ownership;
- [ ] temporary diagnostic workflows are removed;
- [ ] permanent gates remain green;
- [ ] Issue/Milestone completion evidence records commit/run IDs.
