# Session handoff — 2026-10-06

This document records the intent, provenance, validation results, and commit
history of the research/porting session performed on
`research/lsh-lineage-vendor`.

It is meant to be the starting point for the next phase: deeper paper reading,
paper-linked source comments, and careful extraction of common libraries.

## 1. Session objective

The session started with three related code/research lines:

1. **Original SLIDE** — the MLSys 2020 work using LSH to select active neurons.
2. **Optimized SLIDE** — the MLSys 2021 modern-CPU implementation adding
   memory-layout changes, AVX-512 vectorization, and BF16 support.
3. **MONGOOSE** — the ICLR 2021 learnable-LSH framework, sharing authors with
   SLIDE and applying learnable/scheduled LSH to SLIDE/Reformer-style training.

The immediate goals were:

- identify the paper/repository lineage;
- vendor stable source snapshots into one branch without submodules;
- preserve provenance while allowing the old repository structure to be
  replaced;
- establish whether each codebase can still build and perform a real training
  step;
- isolate historical source defects from modern-toolchain defects;
- prepare the tree for source-level paper citations and later library
  refactoring.

By the end of the session, those preparation goals were substantially met.

## 2. Important repository invariant

The most important design decision is the split between `third_party/` and
`ports/`.

### `third_party/`: provenance archive

The three imported source trees under `third_party/` are intended to remain
historical snapshots.

Do **not** apply portability fixes, paper commentary, cleanup, or refactors
directly to these trees.

They are the baseline for answering:

> “What did the selected upstream snapshot actually contain?”

### `ports/`: maintained and annotated copies

`ports/` began as Git-blob-identical copies of the three `third_party/`
snapshots. It is the place for:

- Doxygen comments and Python docstrings tied to paper sections;
- portability/correctness fixes;
- sanitizer fixes;
- CPU/reference implementations;
- common-library extraction in the next phase.

The separation was introduced in commit
`7513e846ff478baf7db1189b006857f038e07a13`.

The next phase should preserve this distinction.

## 3. Source provenance

### Original SLIDE

Paper:

> *SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for
> Large-Scale Deep Learning Systems* — MLSys 2020.

Imported upstream:

- repository: `keroro824/HashingDeepLearning`
- pinned commit:
  `c9283490ffe34ba97005ec6e11800fdcdf165d79`
- local archive: `third_party/slide-original/`
- maintained copy: `ports/slide-original/`

### Optimized SLIDE

Paper:

> *Accelerating SLIDE Deep Learning on Modern CPUs: Vectorization,
> Quantizations, Memory Optimizations, and More* — MLSys 2021.

Historical canonical repository:

- `IntelLabs/SLIDE_opt_ia` — now unavailable.

Lineage evidence:

- coauthor Yong Wu's surviving mirror:
  `uyongw/SLIDE_opt_ia`
- useful lineage anchor:
  `85f758c631fd2e0e9f2d33f5fefb897370e4e911`
- `RUSH-LAB/SLIDE` records optimized SLIDE as a submodule and pins:
  `29e40b45d89d62d50bc4a86df5b804b0594ce514`
- that exact descendant tree was recovered from a surviving fork.

Git ancestry comparison showed that the RUSH-LAB-pinned snapshot is six
commits ahead and zero behind the Yong Wu anchor.

Local paths:

- archive: `third_party/slide-optimized-avx512/`
- maintained copy: `ports/slide-optimized-avx512/`

### MONGOOSE

Paper:

> *MONGOOSE: A Learnable LSH Framework for Efficient Neural Network Training*
> — ICLR 2021.

Imported upstream:

- repository: `HazyResearch/mongoose`
- pinned commit:
  `890043b39b59a93b8e91a30bc79f4b8125febb78`
- archive: `third_party/mongoose/`
- maintained copy: `ports/mongoose/`

Beidi Chen and Anshumali Shrivastava are shared authors between original SLIDE
and MONGOOSE.

Detailed provenance is in `papers/SOURCES.md`.

## 4. Initial vendoring and research layout

The first structural commit was:

- `7213080521332220208cccd9ef84de997512c49e` —
  **Vendor SLIDE lineage research snapshots**

It replaced the previous root layout on the research branch and created:

- `third_party/slide-original/`
- `third_party/slide-optimized-avx512/`
- `third_party/mongoose/`
- `papers/SOURCES.md`
- `SOURCE_MANIFEST.tsv`

No Git submodules are used.

The existing SLIDE paper notes that had already been added to this fork were
retained under `papers/slide-2020-existing-notes/`.

## 5. Build and training validation

The important lesson from this session is that “builds” and “actually trains”
must be tested separately.

### Original SLIDE

Original SLIDE:

- builds on a current GCC/CMake environment;
- performs one synthetic training batch;
- performs evaluation;
- saves NPZ weights;
- shows a real parameter update.

Strengthened measurement:

```text
SMOKE_WEIGHT_DELTA_L1 0.0319997
```

The parameter-update check was introduced around:

- `e225bad46bbb807f0bfb6d0b59d29bccdf6deb7f`
- `38da9d2acae8c73db9c66d2aafcde287287708fe`
- `9ab371f5ec6817ad16a0eda5c4da03ac46227b13`

The maintained Original port later passed ASan/UBSan train/save/teardown.

### Optimized SLIDE: compiler environment

The historical source expects Intel compiler behavior.

Two Intel container generations were checked:

Current:

```text
intel/oneapi:2026.1.0-devel-ubuntu22.04
```

This provides `icx/icpx`, but not Classic `icc/icpc`.

Historical environment used for reproduction:

```text
intel/oneapi-hpckit:2023.2-devel-ubuntu22.04
icpc (ICC) 2021.10.0 20230609
```

This compiler accepts the Intel/SVML-specific vector intrinsics used by the
optimized source, including paths that ordinary GCC does not accept directly.

Key setup/history commits include:

- `9e9dc6d8a2c4da9b56664034eff1c59fc079a649`
- `11abbbda28d2814574d581a196c6903c31b22826`
- `9729df5a7dd07ee3e50cb6c42ca38dabb07c2d17`

### Optimized SLIDE: deterministic scalar vs AVX-512

Early smoke tests initially suggested AVX-512 produced NaNs. Strengthened
diagnostics showed that this was not evidence that AVX arithmetic itself was
broken.

Two historical implementation problems contaminated the tiny fixture:

1. optimizer/gradient state was not initialized;
2. the dense AVX kernel assumed complete 128-output blocks and had no tail
   handling.

After controlling those issues and using deterministic initialization, scalar
FP32 and AVX-512 FP32 produced nearly identical saved weights.

Successful correctness run:

https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37440311849

Representative result:

```text
SCALAR_AVX_MAX_ABS_DIFF 7.450580596923828e-09
SCALAR_AVX_MEAN_ABS_DIFF 1.6711458883378327e-09
SCALAR_AVX_L1_DIFF 6.845013558631763e-06
SCALAR_AVX_FP32_NUMERICAL_COMPARISON_PASS
```

Key diagnostic commits:

- `af7fff92d590a158d0fa4cef8c9294a22342972c` — BF16 runtime runner probe
- `35eaf96b1388f1542cc21335a6fd11e687ee4946` — zero-init diagnostic
- `62692a2b69df47653f6fc8b8becd46765421029e` — aligned AVX A/B diagnostic
- `1eb5da25889d8a57cd7ca671c7238010beca9637` — deterministic scalar/AVX matrix
- `9d3a2e1eb633d4eeae330f28d37ec676b383c6c6` — deterministic seed fix

### Optimized SLIDE: AVX tail and BF16

The maintained port adds masked AVX-512 handling for output tails after full
128-output blocks.

Dedicated run:

https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37442222359

On a runner exposing both `avx512f` and `avx512_bf16`, the maintained
four-output fixture passed:

```text
PORT_AVX_FP32_4_OUTPUT_TAIL_PASS
PORT_BF16_MODE_1_4_OUTPUT_TAIL_PASS
PORT_BF16_MODE_2_4_OUTPUT_TAIL_PASS
```

Thus the maintained port has runtime evidence for:

- AVX-512 FP32;
- BF16 activations + FP32 master weights;
- BF16 activations + BF16 weights.

This is a correctness smoke baseline, not a reproduction of the paper's
throughput numbers.

### Sanitizers

ASan/UBSan were added for maintained SLIDE ports.

Final successful sanitizer run:

https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37443082986

This uncovered and helped fix first-layer backpropagation and allocation issues
that ordinary smoke tests did not reveal.

## 6. Issues discovered and maintained-port fixes

All Issues #1–#11 created during this session are now closed as completed.

| Issue | Problem | Main maintained-port fix commit |
| --- | --- | --- |
| #1 | invalid optimized-SLIDE constructor / undefined `MAX_BUFFER_SIZE` | `bf5ca853cb2e79c8cdca32673b2bfdcd55dced9b` |
| #2 | `DataLayerOpt` indexed writes after `reserve()` | `bdb75c5a8868a5b8ede4f311ed719e8035f69ae7` |
| #3 | teardown ownership, duplicate/interior frees | `aee726141bb725eafdda9f88ab9a5efd1c01fa5d`, `5562243557aade7df4fa65a963559ed3fb1f2359` |
| #4 | MONGOOSE hard CUDA/APEX path | `0490c9ec2b204f027d8a489da56fa75b718d8369`, `6776834d2d682ad1cc2da52bf4b86bd8273db1be` |
| #5 | early AVX NaN diagnostic | closed after #6/#7 + deterministic correctness matrix |
| #6 | uninitialized optimized gradient / Adam state | `6fe7ceefe667660a1e4ea2c498f6f15e42d81c60` |
| #7 | AVX dense-forward output tails omitted | `b0e9b7b26c7cbdfe0c06548733c2464cbf916231` |
| #8 | missing `<cstring>` around `memset` | `634e655681161beffd20e040f6d35e4eafaa2c54` |
| #9 | whitespace-only config line crashes trim helper | `13997e91dc66082b0320892994b051a9ec8a5c6e` |
| #10 | first-layer backprop reads `hiddenlayers[-1]` | `f73db551c7cd7fa7ac8af5cb5555accf6049abfd` |
| #11 | `aligned_alloc` sizes not rounded to alignment | `c56b3d96f673054b0bd3777d3ba8dfc8f8f313b0` |

The important rule is that these fixes live in `ports/`, not
`third_party/`.

## 7. MONGOOSE maintained CPU/reference path

MONGOOSE began with a working standalone learnable-hash optimizer smoke:

```text
loss 0.8027675747871399
weight_delta_l1 0.0774054229259491
MONGOOSE_TRIPLET_TRAINING_STEP_COMPLETED
```

The original Reformer entrypoint, however, assumed:

- NVIDIA APEX;
- unconditional CUDA placement;
- CuPy/NVRTC SimHash fingerprinting;
- script-style Python import layout.

The maintained path was added in:

- `0490c9ec2b204f027d8a489da56fa75b718d8369` —
  CPU/reference Reformer support;
- `4c415f7202be00bbf57a610ac0372deb32056672` —
  CPU one-batch entrypoint smoke;
- `b95e8bc7858c6d757843b88dd64ef487bdd2bafa` —
  strengthened main/hash parameter-update verification;
- `3fb34679a513879aa25c8de97d4b00abf8d034ac` and
  `6776834d2d682ad1cc2da52bf4b86bd8273db1be` —
  package-relative import fixes;
- `6f90b1bc84a99567c7b36503bfbbbc208890baf7` —
  final documentation of Issue #4 resolution.

Successful final run:

https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37452616474

Result:

```text
training device: cpu
APEX available: False
| end of epoch   0 | ... | valid loss 2.79 | valid ppl 16.30
MONGOOSE_REFORMER_CPU_ENTRYPOINT_PASS

main_loss 2.716763734817505
triplet_loss 9.328916549682617
main_parameter_delta_l1 6.317929459735751
rotation_parameter_delta_l1 0.03199991211295128
MONGOOSE_REFORMER_MAIN_AND_HASH_UPDATE_PASS
```

This is sufficient for a maintained CPU correctness baseline.

It is **not** a reproduction of the paper's historical CUDA/APEX throughput
environment.

## 8. Paper-to-code preparation

The paper-reading phase was prepared, but intentionally not finished.

The map is:

- `papers/CODE_PAPER_MAP.md`

The first source annotations were added in:

- `f300e911c82973848352e67a3ab57ae68f04876c` —
  Original SLIDE Doxygen mappings;
- `f4fde70e80a281682aaabc19cdb316bcf7a71b06` —
  optimized SLIDE Doxygen mappings;
- `5cd4d294f92ee4636f1f2f4f5b63f4a4fe0f2378` —
  MONGOOSE scheduler / learnable-LSH docstrings.

The annotation convention established in this session is:

### C/C++

Use Doxygen blocks with distinct sections such as:

- `@par Paper mapping`
- `@par Implementation note`
- `@par Reported effect`
- `@warning` for known implementation constraints

### Python

Use docstrings with the same conceptual separation.

### Critical rule

Do not blur these categories:

1. **paper claim**
2. **what this source snapshot implements**
3. **our inference from code**
4. **our reproduction/smoke result**

In particular, paper speedups should not be attached to a single function
unless the paper itself reports a function-level measurement.

## 9. Commit timeline by phase

This is a compact map of the session-created history.

### A. Vendoring, build probes, and first training smoke

| SHA | Intent |
| --- | --- |
| `7213080521332220208cccd9ef84de997512c49e` | vendor all three research snapshots and provenance |
| `71138956b18717d67cfa82043b93d5aed54c1aab` | add build-smoke workflow |
| `e286a5255e17742976135bcca4a716a653097673` | document build matrix |
| `d1cb7f6a4fcffd7f345bf7472ed784895cb902d0` | probe optimized build past first historical regression |
| `12d32b300dc927d45346ae6a3a0237a07b44f919` | record build blockers |
| `9e9dc6d8a2c4da9b56664034eff1c59fc079a649` | add one-step training + Intel compiler smoke |
| `11abbbda28d2814574d581a196c6903c31b22826` | test Intel Classic optimized training |
| `629342ae8dff438197b34a2f65da68c718d6c2cb` | add generic optimized training smoke |
| `9729df5a7dd07ee3e50cb6c42ca38dabb07c2d17` | exercise Intel Classic AVX-512 builds and runtime |
| `d5eef51ebeb3e6ad28c077f1f08455be231fd4bd` | document successful initial training/Intel results |

### B. Stronger update checks and AVX/BF16 diagnosis

| SHA | Intent |
| --- | --- |
| `e225bad46bbb807f0bfb6d0b59d29bccdf6deb7f` | verify actual SLIDE parameter updates; probe BF16 |
| `38da9d2acae8c73db9c66d2aafcde287287708fe` | strengthened training smoke |
| `9ab371f5ec6817ad16a0eda5c4da03ac46227b13` | Original SLIDE parameter-update check |
| `3ec89a7d0c96bdb617ecedf70bc6ebf3287d3dc8` | continue BF16 probing after strict AVX failure |
| `af7fff92d590a158d0fa4cef8c9294a22342972c` | sample BF16-capable hosted runners |
| `35eaf96b1388f1542cc21335a6fd11e687ee4946` | test zero-init hypothesis |
| `62692a2b69df47653f6fc8b8becd46765421029e` | aligned-output AVX A/B comparison |
| `1eb5da25889d8a57cd7ca671c7238010beca9637` | deterministic scalar-vs-AVX correctness matrix |
| `9d3a2e1eb633d4eeae330f28d37ec676b383c6c6` | fix deterministic seed injection |

### C. Archive/port split and initial paper comments

| SHA | Intent |
| --- | --- |
| `7513e846ff478baf7db1189b006857f038e07a13` | create `ports/` and paper-to-code map |
| `f300e911c82973848352e67a3ab57ae68f04876c` | annotate Original SLIDE |
| `f4fde70e80a281682aaabc19cdb316bcf7a71b06` | annotate optimized SLIDE |
| `5cd4d294f92ee4636f1f2f4f5b63f4a4fe0f2378` | annotate MONGOOSE scheduler/learnable LSH |

### D. Maintained SLIDE fixes

| SHA | Intent |
| --- | --- |
| `bf5ca853cb2e79c8cdca32673b2bfdcd55dced9b` | fix Issue #1 |
| `bdb75c5a8868a5b8ede4f311ed719e8035f69ae7` | fix Issue #2 |
| `6fe7ceefe667660a1e4ea2c498f6f15e42d81c60` | fix Issue #6 |
| `aee726141bb725eafdda9f88ab9a5efd1c01fa5d` | first teardown/ownership fix for Issue #3 |
| `634e655681161beffd20e040f6d35e4eafaa2c54` | fix Issue #8 |
| `13997e91dc66082b0320892994b051a9ec8a5c6e` | fix Issue #9 |
| `b0e9b7b26c7cbdfe0c06548733c2464cbf916231` | fix Issue #7 AVX output tails |
| `f3f6e8f6ff53f8009863408af1a4340c774af0b4` | maintained-port build/training CI |
| `5562243557aade7df4fa65a963559ed3fb1f2359` | complete Original Adam-buffer ownership fix |
| `839f2c4479abbbc1338f6502822e476a8d5bb27b` | add ASan/UBSan CI |
| `f73db551c7cd7fa7ac8af5cb5555accf6049abfd` | fix Issue #10 |
| `c56b3d96f673054b0bd3777d3ba8dfc8f8f313b0` | fix Issue #11 |
| `252b0818559a3b08a241aa48e0ee08852d477bfa` | record maintained correctness results |
| `e117a8155420c9e753d4134e051d709e94d07570` | document maintained fixes and validation |

### E. MONGOOSE CPU/reference completion

| SHA | Intent |
| --- | --- |
| `0490c9ec2b204f027d8a489da56fa75b718d8369` | add CPU/reference Reformer path |
| `4c415f7202be00bbf57a610ac0372deb32056672` | add CPU entrypoint smoke |
| `b95e8bc7858c6d757843b88dd64ef487bdd2bafa` | verify main/hash parameter updates |
| `3fb34679a513879aa25c8de97d4b00abf8d034ac` | first package-import fix |
| `6776834d2d682ad1cc2da52bf4b86bd8273db1be` | complete relative-import conversion |
| `6f90b1bc84a99567c7b36503bfbbbc208890baf7` | document and close Issue #4 |

## 10. CI evidence worth retaining

The repository now contains both current gates and older diagnostic workflows.

The older diagnostics are intentionally useful: they show **how** an apparent
AVX failure was decomposed into source bugs, fixture assumptions, and genuine
compiler/runtime requirements.

Especially useful runs:

- Original parameter-update smoke:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37390030093
- deterministic scalar/AVX correctness:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37440311849
- maintained AVX tail + BF16:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37442222359
- maintained ASan/UBSan:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37443082986
- MONGOOSE CPU Reformer + main/hash update:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37452616474
- final maintained-ports smoke after documentation:
  https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37452963970

## 11. Current end-of-session state

At the handoff point captured by pre-document HEAD
`6f90b1bc84a99567c7b36503bfbbbc208890baf7`:

- all Issues #1–#11 are closed as completed;
- Original SLIDE maintained port builds, trains, updates parameters, saves, and
  tears down under normal and sanitizer CI;
- optimized SLIDE maintained port passes scalar, AVX-512, output-tail, and BF16
  correctness smoke tests;
- deterministic scalar vs AVX saved weights agree to roughly 1e-8 max absolute
  difference on the test fixture;
- MONGOOSE standalone learnable-hash training updates parameters;
- MONGOOSE maintained Reformer entrypoint runs on CPU without APEX;
- MONGOOSE main-model parameters and learnable rotation/hash parameters both
  update in the strengthened smoke;
- `third_party/` remains the provenance archive;
- `ports/` is ready for deeper paper-linked commentary and refactoring.

## 12. Recommended next phase

The next phase should return to the original research objective rather than
continue broad compatibility work.

### A. Deep paper-linked source comments

Expand the existing annotations function-by-function.

For each important function/class:

1. identify the smallest relevant paper section, equation, algorithm, figure,
   or table;
2. write a Doxygen/docstring summary in our own words;
3. describe the precise implementation correspondence;
4. state where the code diverges from or approximates the paper;
5. keep reported performance/accuracy effects separate from mechanism;
6. link maintenance Issues only when implementation behavior materially affects
   interpretation.

A useful standard already exists in `papers/CODE_PAPER_MAP.md`.

### B. Common-library extraction

Only refactor after the paper/code relationship is clear.

Likely commonality candidates include:

- LSH table/bucket abstractions;
- hash-index composition;
- Densified WTA / random-projection interfaces;
- sparse-record storage;
- scheduler/change-detection interfaces;
- training-smoke fixture helpers.

Avoid immediately merging algorithms merely because they look syntactically
similar. Original SLIDE, optimized SLIDE, and MONGOOSE intentionally make
different algorithmic and systems assumptions.

A safe refactor sequence would be:

1. annotate first;
2. identify behaviorally equivalent units;
3. add cross-implementation regression tests;
4. extract the smallest common interface;
5. preserve adapters for historical layouts;
6. rerun scalar/AVX/BF16/MONGOOSE smoke tests after each extraction.

### C. Performance/accuracy reproduction as a separate track

Current CI demonstrates correctness on tiny fixtures.

It does **not** reproduce:

- paper-scale datasets;
- reported throughput;
- paper accuracy curves;
- memory-footprint measurements;
- the historical CUDA/APEX MONGOOSE benchmark stack.

Those should be treated as a distinct reproduction effort, not mixed into
source-comment or library-refactor commits.

## 13. Files to read first next session

Recommended order:

1. this document;
2. `papers/SOURCES.md`;
3. `papers/CODE_PAPER_MAP.md`;
4. `BUILDING.md`;
5. `ports/README.md`;
6. the relevant paper;
7. the matching `ports/` implementation;
8. `third_party/` only when checking historical provenance.

## 14. Branch / migration note

All work in this session is on:

```text
research/lsh-lineage-vendor
```

The directory structure was intentionally designed so it can later be moved
into a separate repository.

When that move happens, preserve:

- the `third_party/` archive snapshots and their manifest/provenance;
- `ports/` as the maintained layer;
- `papers/` mapping/provenance documents;
- the relevant Actions workflows;
- this handoff/history document, or equivalent commit-history notes.

The commit SHAs in this document refer to the history of the current
`fregata-ariel/HashingDeepLearning` repository and should remain available as
historical references even after a later repository split.
