# G-SLIDE integration milestones

These milestones apply the research integration playbook with CPU emulation as
the primary routine test. Dates are not assigned: GPU sessions depend on the
user's availability and are expected only at infrequent checkpoints.

## G1 — Provenance and CPU mechanism baseline

Goal: freeze the reference and establish independently checked selected CUDA
arithmetic/serial-state contracts without a GPU dependency.

Acceptance:

- exact 27-file upstream archive and MIT license pinned to
  `d93c2f6d0bbf1dd7b96d9c2340ed629c29f4f902`;
- maintained copy separated from the immutable archive;
- six mechanism/state oracles executing ten actual CUDA definitions;
- Softmax negative-logit regression and maintained fix;
- historical WTA packing recorded as a variant;
- ledger, CUDA code contracts, generated source/compiler evidence and
  hosted plain + ASan/UBSan CPU gate pass;
- current 31-file Python typing gate remains green.

Implementation is present at this checkpoint. Close the milestone only after
the exact committed revision passes hosted CI. It does not close the full
production CUDA execution baseline.

## G2 — Broader CPU coverage and composed training fixture

Goal: test more released kernel variants and the arithmetic connections between
selection, training updates and index reconstruction while GPU access is absent.

Acceptance:

- deeper-layer backward deltas and both matrix layouts against dense oracles;
- alternate sparse forward/Softmax/backward bodies where the serial adapter
  can represent their semantics; document reductions that remain substituted;
- deterministic candidate-to-active-set conversion and index rebuild after
  parameter updates; distinguish host substitutes from Thrust execution;
- composed tiny CPU fixture with finite loss, nonzero parameter changes,
  state reset, reproducible seed/active sets and normal host teardown;
- plain/sanitizer gates, per-mechanism ledger and code contracts extended;
- no claim that the production CUDA Network/Layer launch path is executed.

Potential capacity, ownership and synchronization findings become focused
regressions only when reproduced. Parallel-only questions stay in G3.

## G3 — Native CUDA checkpoint on an available GPU

Goal: complete the real production build/execution baseline and compare native
device behavior with the saved CPU expectations in a limited GPU session.

Blocked on a user-announced Hugging Face / Colab or other GPU environment.
Do not create a recurring GPU job or assume the service/toolkit is available.

Acceptance:

- record exact commit, GPU model/architecture, toolkit, driver and build logs;
- maintained CMake/CUDA build, kernel parity checks with explicit tolerances;
- real multi-warp reductions, tail handling, shared-memory dispatch,
  contention/ordering, overflow and bounded pool behavior;
- deterministic one-batch forward/backward/Adam/rebuild/save/teardown through
  the maintained production path, with observable parameter changes;
- device memory/race/synchronization diagnostics where supported, and
  investigation of CUDA allocation/free ownership;
- archived small fixtures/logs and precise passed/skipped/pending boundaries.

Large-dataset accuracy and the paper's throughput claims are a later milestone
to scope after G3; they are not acceptance criteria for these CPU-first stages.

## Resume points

Baseline before this addition: `602687f5a26720faf69ff65abf11dee7c6abefae`
on `research/lsh-lineage-vendor` (research guide and seven-workflow cleanup).
After G1 CI verification, G2 is the next work that can proceed without GPU
access. The detailed source/test limits and G3 checklist live in
`ports/g-slide/CPU_VALIDATION.md`.
