# CI architecture

This repository deliberately separates permanent maintenance gates from
temporary research diagnostics.

The permanent workflows should answer:

> Does the maintained, annotated implementation still satisfy the contracts we
> have decided to support?

Temporary diagnostics answer:

> Why did this historical/reference implementation behave this way?

Those are different questions. The second kind should be removed once the
answer has been converted into a maintained regression test.

## 1. Permanent workflows

The permanent set after cleanup is seven workflows.

| Workflow | Responsibility | Notes |
| --- | --- | --- |
| `traceability.yml` | paper/code/test ledger validation, C/C++/CUDA/Cython documentation audit, small deterministic paper-mechanism oracles, G-SLIDE CPU body emulation with ASan/UBSan and archive hashes, traceability artifact | Source-of-truth evidence workflow; CPU emulation does not validate GPU execution. |
| `python-typing.yml` | all maintained Python annotation audit, strict mypy, SLIDE Python example contracts, final traceability/doc gate | Auto-discovers all maintained Python files. |
| `ports-smoke.yml` | Original SLIDE end-to-end maintained smoke; Optimized SLIDE scalar integration; historical Intel AVX/BF16 maintained-port smoke | Real execution/save/teardown. Hardware capability is printed, not assumed. |
| `ports-sanitizers.yml` | ASan/UBSan on maintained native SLIDE ports | Finds lifetime/indexing issues missed by ordinary smoke. |
| `hardware-oracles.yml` | lightweight AVX-512/BF16 numerical oracles on GitHub-hosted CPU when flags permit | Capability-gated; unsupported CPU means skip, not pass. |
| `mongoose-native.yml` | focused C++/Cython native LSH set/buffer contract | Fast native-boundary regression. |
| `mongoose-reformer-smoke.yml` | MONGOOSE CPU integration from scheduler/mining through one-batch training and parameter updates | Includes maintained native boundary plus Python control-flow integration. |

These workflows intentionally overlap a little at integration boundaries. The
overlap is useful when the scope differs:

- `mongoose-native.yml` localizes native/Cython failures quickly;
- `mongoose-reformer-smoke.yml` proves that boundary works in the full
  maintained MONGOOSE path;
- `traceability.yml` proves paper/mechanism correspondence;
- `python-typing.yml` proves source contracts and also runs the final
  traceability/document checks.

## 2. Hardware policy

Do not pin assumptions about GitHub-hosted CPU models.

Before hardware-dependent execution, print:

```text
uname -a
lscpu
/proc/cpuinfo flags
GITHUB_SHA
```

Then gate:

- AVX-512F arithmetic on `avx512f`;
- native BF16 arithmetic on `avx512_bf16`;
- historical Intel compiler paths on the intended compiler/container.

A skipped hardware step is evidence that the capability was absent, not
evidence that the implementation passed.

Keep an independent scalar/reference oracle so correctness coverage is not
lost on a weaker runner.

G-SLIDE uses routine CPU serial emulation in the existing traceability gate.
No GPU service or GPU workflow is required. Actual CUDA build, parallelism,
device memory/synchronization and production training remain pending until an
infrequent user-provided GPU milestone session. See
[the CPU coverage contract](../ports/g-slide/CPU_VALIDATION.md).

## 3. Trigger design

Permanent workflows should trigger only when their protected contract could
have changed.

Examples:

- typing: `ports/**/*.py`, stubs, typing/audit tools, traceability metadata;
- traceability: ledger, tests, verified native sources/headers;
- native MONGOOSE: C++/Cython wrapper and boundary tests;
- hardware oracles: vector/BF16 kernel implementation and their tests;
- maintained smoke: maintained port source/build/test fixtures.

Avoid `push` with no path filter unless the workflow truly protects the entire
repository.

The current seven workflows use path-filtered pushes to
`research/lsh-lineage-vendor` plus `workflow_dispatch`; they do not currently
declare `pull_request` triggers. When this work is moved to another branch
or repository, update branch filters and decide which checks must run on PRs.
Preserving a workflow file without updating those filters does not establish
CI coverage for the new destination.

## 4. Traceability artifacts

`traceability.yml` materializes:

- current Git commit;
- registered local paper SHA-256 values;
- validated traceability ledger;
- human-readable paper/code/test mapping.

The Action artifact is a run snapshot, not the permanent source of truth.

Permanent evidence remains:

1. Git commit;
2. paper files/provenance;
3. traceability metadata;
4. tests;
5. workflow definition.

If an artifact expires, the same commit can regenerate it.

## 5. Adding a new paper to CI

Do not immediately create a new permanent workflow for every paper.

Prefer this order:

1. add small tests to `tests/traceability/`;
2. register them in `papers/traceability.json`;
3. run hardware-independent mechanism tests in `traceability.yml`;
4. extend an existing maintained smoke if the new reference code fits an
   existing runtime family;
5. create a dedicated native/hardware workflow only when it isolates a
   genuinely distinct boundary.

A separate permanent workflow is justified when it provides materially better
failure localization or requires a distinct build/runtime dependency set.

## 6. Temporary diagnostic workflow policy

Temporary workflows are useful and encouraged during archaeology.

Typical examples:

- old compiler probe;
- runner CPU/BF16 sampling;
- CI-only patch to discover the next historical failure;
- zero-initialization hypothesis;
- scalar/AVX A/B comparison;
- alignment/layout experiment.

Rules:

1. make the diagnostic purpose obvious in the name;
2. never treat CI-only patches to `third_party/` as maintained fixes;
3. record the useful Action run/commit in an Issue or handoff;
4. translate the finding into a `ports/` fix plus regression test;
5. delete the diagnostic workflow once superseded.

Git history preserves the workflow source after deletion.

## 7. Workflows retired after the SLIDE/MONGOOSE investigation

The following workflows existed during diagnosis but are not part of the
permanent architecture.

| Retired workflow | Historical question | Permanent successor |
| --- | --- | --- |
| `aligned-avx-diagnostic.yml` | Does aligned/full-block AVX arithmetic agree after controlling fixture issues? | `hardware-oracles.yml`, `ports-smoke.yml`, traceability AVX oracles |
| `bf16-runtime-probe.yml` | Which hosted runners expose AVX-512 BF16 and can run the historical path? | capability reporting in `hardware-oracles.yml` and Intel job in `ports-smoke.yml` |
| `build-smoke.yml` | Can each historical tree get through an initial build/import? | maintained `ports-smoke.yml`, `mongoose-native.yml`, strict typing |
| `optimized-correctness.yml` | Do deterministic scalar and AVX workspaces agree after CI-only historical fixes? | maintained Layer/layout/AVX tests in `ports-smoke.yml` and traceability |
| `ports-avx-probe.yml` | Does the maintained four-output tail/BF16 path execute? | optimized Intel job in `ports-smoke.yml` plus `hardware-oracles.yml` |
| `training-smoke.yml` | Can historical/reference trees perform one training step, often with CI-only compatibility patches? | maintained `ports-smoke.yml` and `mongoose-reformer-smoke.yml` |
| `vendor-papers.yml` | One-time download/commit of available SLIDE proceedings PDFs | committed paper files + traceability SHA-256 validation |
| `zero-init-diagnostic.yml` | Was uninitialized optimized state a cause of NaNs? | maintained initialization fixes + smoke/sanitizer/AVX regression tests |

Historical Action run IDs and investigation details remain in
`SESSION_HANDOFF_2026-10-06.md` and Git history.

## 8. Why cleanup matters

Leaving every diagnostic workflow enabled causes four problems:

1. duplicate compute cost;
2. stale CI-only patches continue to exercise code we no longer maintain;
3. new contributors cannot distinguish gates from archaeology;
4. a green historical probe can be mistaken for evidence about the maintained
   implementation.

A small, explicit permanent workflow set makes the repository's supported
contracts legible.

## 9. CI review checklist

When a paper/implementation integration milestone closes:

- [ ] every permanent workflow has a clear maintained contract;
- [ ] no permanent workflow depends on patching `third_party/`;
- [ ] hardware assumptions are capability-gated;
- [ ] paper mechanism tests live in traceability;
- [ ] native boundaries have focused tests;
- [ ] Python maintained scope is automatically enrolled in strict typing;
- [ ] sanitizer coverage exists for native ownership-sensitive code;
- [ ] temporary diagnostics have documented successors;
- [ ] temporary diagnostic workflow files are removed.
