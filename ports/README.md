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

## Current verification

- Original port: GCC train/save/teardown PASS.
- optimized port: GCC scalar train/save/teardown PASS.
- Original + optimized scalar: ASan/UBSan PASS.
- Intel Classic 2021.10 AVX-512 FP32: PASS.
- scalar vs AVX saved-weight maximum absolute difference on the deterministic
  128-output fixture: `7.45e-09`.
- AVX four-output tail: PASS.
- AVX512-BF16 mode 1 and mode 2 on capable hardware: PASS.

See `BUILDING.md` for run links and details.

## Rules for future changes

1. Do not apply maintenance changes to `third_party/`.
2. Keep one behavioral problem per Issue and, where practical, per commit.
3. Doxygen/docstrings must distinguish paper claims from implementation
   inference.
4. Performance figures are attached to paper experiments, not individual
   functions unless the paper explicitly reports that function-level result.
5. New compatibility changes should add or strengthen a smoke/sanitizer test.
