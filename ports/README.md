# Maintained / annotated working copies

`third_party/` is the provenance-preserving archive. It remains byte-for-byte
aligned with the selected upstream snapshots.

`ports/` contains the working copies that we can document and repair. Changes
here are intentionally split into reviewable commits so paper interpretation,
portability fixes, and behavior changes remain distinguishable.

## What is maintained here

- paper-to-code Doxygen comments for Original and optimized SLIDE;
- Python docstrings for the MONGOOSE scheduler and learnable-LSH paths;
- portability and correctness fixes discovered by build/training/sanitizer
  probes;
- CI that trains tiny fixtures through save and teardown.

The paper-section index is `papers/CODE_PAPER_MAP.md`.

## Optimized-SLIDE fixes

| Issue | Maintained-port change | Status |
| --- | --- | --- |
| #1 | repair invalid constructor; remove undefined `MAX_BUFFER_SIZE` guard | closed |
| #2 | resize per-record metadata vectors before indexed writes | closed |
| #3 | explicit ownership and safe teardown; remove duplicate/interior frees | closed |
| #6 | zero gradient and Adam-state buffers before accumulation | closed |
| #7 | masked AVX-512 output-tail handling after complete 128-output blocks | closed |
| #8 | include `<cstring>` where `memset` is used | closed |
| #9 | handle whitespace-only config lines | closed |
| #10 | do not read a previous layer when processing layer zero | closed |
| #11 | round 64-byte aligned-allocation sizes to a multiple of 64 | closed |

Issue #5, the original AVX NaN diagnostic, is also closed: deterministic
scalar-vs-AVX comparison and the corrected four-output tail fixture show finite,
numerically consistent behavior after #6/#7 are controlled.

## MONGOOSE fixes

Issue #4 is closed in the maintained port. The historical source remains under
`third_party/mongoose/`, while `ports/mongoose/` now provides:

- optional APEX use for the FP32 training path;
- explicit CPU/CUDA device selection;
- a PyTorch reference SimHash fingerprint backend when CuPy/NVRTC is absent;
- package-relative `reformer_lib` imports;
- a CPU one-batch Reformer smoke test that reaches train, eval, and normal exit;
- a separate optimizer-step assertion proving both main-model and learnable
  rotation/hash parameters change.

Successful verification:
https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37452616474

Observed deltas in the strengthened check were
`main_parameter_delta_l1 = 6.317929459735751` and
`rotation_parameter_delta_l1 = 0.03199991211295128`.

This resolves modern CPU correctness/portability for the smoke fixture; it does
not claim reproduction of the historical CUDA/APEX throughput results.

## Current verification

G-SLIDE now has CPU-first serial emulation of selected actual CUDA bodies,
independent arithmetic/state oracles, an exact upstream archive check and
ASan/UBSan checks in traceability CI. Full CUDA compilation, production
train/save/teardown and parallel GPU execution remain pending. See
[the coverage checkpoint](g-slide/CPU_VALIDATION.md) for precise boundaries.

- Original port: GCC train/save/teardown PASS.
- optimized port: GCC scalar train/save/teardown PASS.
- Original + optimized scalar: ASan/UBSan PASS.
- Intel Classic 2021.10 AVX-512 FP32: PASS.
- scalar vs AVX saved-weight maximum absolute difference on the deterministic
  128-output fixture: `7.45e-09`.
- AVX four-output tail: PASS.
- AVX512-BF16 mode 1 and mode 2 on capable hardware: PASS.
- MONGOOSE/Reformer CPU train/eval + main/hash optimizer updates: PASS.

See `BUILDING.md` for run links and details.

## Rules for future changes

1. Do not apply maintenance changes to `third_party/`.
2. Keep one behavioral problem per Issue and, where practical, per commit.
3. Doxygen/docstrings must distinguish paper claims from implementation
   inference.
4. Performance figures are attached to paper experiments, not individual
   functions unless the paper explicitly reports that function-level result.
5. New compatibility changes should add or strengthen a smoke/sanitizer test.


## Research integration process

For future paper/reference-code additions, follow
[the research integration playbook](../docs/RESEARCH_INTEGRATION_PLAYBOOK.md). In particular:

- establish behavior with independent tests before refactoring;
- add maintenance changes only under `ports/`;
- register stable paper/code/test IDs in `papers/traceability.json`;
- document ownership, preconditions, relation strength, and traceability IDs;
- enroll new maintained Python automatically in the all-ports strict gate;
- remove temporary diagnostic workflows after a maintained regression replaces
  them.

The permanent CI topology is documented in
[the CI architecture guide](../docs/CI_ARCHITECTURE.md).
