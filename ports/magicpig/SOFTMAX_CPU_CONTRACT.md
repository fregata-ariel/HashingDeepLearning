# MagicPIG selected Softmax CPU contract (#58)

The fixture runs the actual extracted `softmax_kernel`,
`softmax_kernel_full`, and `softmax_kernel_optimized` bodies from the maintained
copy. It changes no production arithmetic and leaves archived blobs unchanged.
It does not build the Torch extension or execute a model, CUDA or FlashInfer.

## Inputs and independent expectations

The fixed matrix uses lengths 1, 2, 15, 16, 17, 31, 32, 33; common shifts
0, -32, -128; and square-root dimension factors 1, 2, 4. Each raw score is the
float32 literal arithmetic sequence `((i*7)%13-6)*0.25 + shift`.
The independent oracle converts those actual input floats to FP64, divides by
the dimension factor, subtracts the independently computed maximum, and uses
host FP64 exponential/logarithm to compute probabilities and metadata.

Sparse receives already-scaled scores; full and optimized receive raw scores.
This convention prevents an accidental second square-root division. The metadata
is base two: `max / ln(2)` and `(max + ln(sum(exp(score-max)))) / ln(2)`.
Sixteen zero scores independently pin the probability at exactly 1/16 and LSE
at exactly four. These metadata are attention support values, not a complete
sampling or generation result.

| Selected body | Numerical cases | Legal default allocation | Bounds interpretation |
|---|---:|---|---|
| sparse | 72 | logical length plus two unchanged guard floats | scalar tails respect logical length |
| full | 72 | rounded-to-16 length plus two guard floats | logical outputs agree, but rounded padding writes are known defects |
| optimized | 18 | complete 16- or 32-element tile plus two guards | only complete tiles establish numerical support |

Portable lanes represent sixteen FP32 floats. Their vector exponential is host
`std::exp(float)`; it is explicitly a software substitute. Native AVX512F/FMA
uses the actual extracted polynomial and constants, after capability gating.
Both modes use explicit host `expf32`/`log2f32` aliases for the full body's scalar
header calls. Original header/runtime linkage remains unvalidated.

Portable probability absolute tolerance is 2e-6. Native probability tolerance is
0.005 + 0.02 times the FP64 probability. The normalization sum tolerance is
2e-5. Metadata permits float common-shift rounding of
`2e-6*(1+abs(FP64 base2 maximum))`; native polynomial LSE permits an additional
0.03. The full body uses the host scalar exponential and does not receive that
additional polynomial allowance. Local maximum probability errors were
7.03589660977e-8 portable and 0.000619072889033 native; these are observations,
not a replacement for the fixed tolerances.

## Existing defects, with separate evidence

The default fixture allocates legal rounded storage before invoking either
vectorized tail. It records 54 full-tail and 54 optimized-tail cases separately
from the complete-tile numerical support. Full scales and normalizes padding.
Optimized's condition `i < nnz` selects the full mask even on the final partial
tile, treating padding as real candidates. Padding is given the same common
shift to keep relative logits moderate; a positive padded probability and a
valid-subset sum below one prove the normalization defect. Full length one with
dimension factor one can write identical padding bits, so canaries alone cannot
prove absence of invalid writes.

The executable accepts isolated negative-probe flags. Run these only in a child
process with ASan/UBSan, never through an in-process correctness test:

| Flag | Input | Required portable diagnostic |
|---|---|---|
| `--probe-full-tail` | exactly one allocated float, nnz=1, scale=2 | ASan heap-buffer-overflow |
| `--probe-optimized-tail` | exactly one allocated float, nnz=1, scale=2 | ASan heap-buffer-overflow |
| `--probe-sparse-empty` | null score, nnz=0 | fatal UBSan load of null float pointer |
| `--probe-full-empty` | null score, nnz=0, scale=2 | fatal UBSan load of null float pointer |

Require the start marker, nonzero exit, expected diagnostic, and the selected
body/adapter stack context. An arbitrary exception or crash is not a successful
reproduction. In local GCC native builds, the optimized masked-load intrinsic
was not instrumented by ASan and the source call returned; the fixture then
threw its fallback exception. That outcome does not establish native memory
safety or count as a reproduced ASan error. The strict negative probes therefore
use portable lanes. Native default allocations remain legal.

Optimized empty input uses valid guard storage in the default fixture. It writes
no score, but emits negative infinity for both metadata fields without a status.
Sparse/full empty input dereferences the end iterator. No empty-domain result is
counted as valid probability support.

Three separate extreme fixtures have sixteen zero scores and sixteen -90,
-100 or -1000 scores.
The host exponential underflows normally and agrees with the FP64 expectation;
the actual native polynomial produces an invalid or significantly wrong
distribution because its exponent-bit construction is unbounded. At -90 the
observed zero-row probability was negative zero with NaN LSE; at -100 it was
-1.45102e-35 while the low-score half received 1/16 and LSE was NaN; at -1000
the low-score half received 1/16 and LSE was approximately 97.3001 instead of
four. The native test requires a probability defect in each of these rows. This is not a claim of accuracy on all
finite logits. Negative-probe and expected-defect records must change when a
maintained repair is made, alongside tests asserting the corrected behavior.

## Focused repair issue drafts

1. **MagicPIG: bound full and optimized Softmax tails to logical candidate length.**
   Full loops over rounded storage; optimized chooses an all-ones mask whenever
   `i < nnz`, including partial tiles. Repair maintained bodies to avoid any
   read/write past `nnz`, exclude masked lanes from maximum and exponential sum,
   and preserve metadata. Test exact capacities 1/2/15/17/31/33 under ASan plus
   padded-lane contamination and independent probabilities. Keep the archive
   unchanged and update expected-defect enrollment to repaired support.
2. **MagicPIG: define and enforce empty Softmax candidate behavior.**
   Sparse/full dereference empty `max_element`; optimized silently returns -Inf.
   Choose an explicit rejection/status or documented empty result consistently,
   then test zero candidates without out-of-bounds dereferences or accidental
   probability/metadata claims. Maintain separately from GPU/model dispatch.
3. **MagicPIG: bound native polynomial exponential underflow domain.**
   Bit construction for large negative exponents can yield negative/nonfinite or
   incorrect probabilities instead of underflow. Repair the maintained
   exponential with a stated finite-domain contract and safe underflow handling;
   test vector/scalar boundaries near exponent underflow, -90/-100/-1000 and
   mixed moderate/extreme rows against FP64. Portable host exp success does not
   validate the native polynomial.
