# MagicPIG selected Softmax CPU contract (#58, #72–#74)

The fixture runs the actual extracted `softmax_kernel`,
`softmax_kernel_full`, and `softmax_kernel_optimized` bodies from the maintained
copy, including the repairs for logical tails, empty support and native
exponential extremes. Maintained arithmetic has changed; archived
`third_party/magicpig` blobs remain immutable. The suite does not build the
Torch extension or execute a model, CUDA or FlashInfer.

## Inputs, ownership and metadata

The caller supplies nonnegative `nnz`, valid maximum/LSE output pointers, and
exactly `nnz` initialized scores for a nonempty row. Buffers remain caller-owned.
Regular nonempty Softmax requires finite scaled logits and a positive finite
dimension scale where scaling is used. A finite raw score and scale alone do
not guarantee a finite scaled result for arbitrary inputs. Sparse receives
already-scaled scores; full and optimized receive raw scores. This convention
prevents an accidental second square-root division.

Each body changes the logical scores to normalized probabilities and returns
base-two metadata: `max / ln(2)` and
`(max + ln(sum(exp(score-max)))) / ln(2)`. The fields are a maximum and log-sum-exp,
not a raw exponential sum. Finite logits can still produce infinite FP32
metadata when the mathematical base-two value is outside FP32 range: for example,
a maximum near `FLT_MAX` overflows multiplication by `log2(e)`. Finite metadata
is promised only when representable; this does not imply nonfinite probabilities.
The extreme regression with maximum `1e38` keeps its metadata representable
while the shifted difference from `-FLT_MAX` overflows negatively.

At `nnz == 0`, all three bodies return without score access or writes; score may
be null. Maximum and base-two LSE are both `-Inf`, denoting zero mass. There are
no probabilities and the void ABI adds no status field. Six success cases cover
each body with null storage and with two unchanged sentinel floats. This kernel
contract does not by itself validate an empty attention-server output; those
caller checks belong to the attention suite. Negative lengths, invalid metadata
pointers and nonfinite regular Softmax logits remain outside this contract.

## Logical-length regression matrix

The fixed matrix uses lengths 1, 2, 15, 16, 17, 31, 32, 33; common shifts
0, -32, -128; and square-root dimension factors 1, 2, 4. Each raw score is the
float32 literal arithmetic sequence `((i*7)%13-6)*0.25 + shift`. The independent
oracle converts actual input floats to FP64, scales once, computes its own
maximum, and uses host FP64 exponential/logarithm for probabilities and metadata.
Sixteen zero scores independently pin probabilities to exactly 1/16 and LSE
to exactly four.

| Selected body | Guarded numerical cases | Exact-allocation cases | Tail behavior |
|---|---:|---:|---|
| sparse | 72 | 72 | complete SIMD tiles plus scalar tails |
| full | 72 | 72 | scaling and normalization never touch padding |
| optimized | 72 | 72 | maximum, sum and writes exclude padding |

The 216 exact-allocation calls use `new float[nnz]`, including lengths
1/15/17/31/33. ASan/UBSan/LSan must complete successfully with leak detection
enabled. Separate guarded calls retain two unchanged sentinel floats directly
after the logical length. There are no expected-failure Softmax probes in the
current descriptor: corrected behavior is tested as ordinary successful calls.

Portable lanes represent sixteen FP32 floats; their vector exponential is host
`std::exp(float)`, an explicit software substitute. Native AVX512F/FMA executes
the actual extracted exponential and constants after CPU/OS capability gating.
Both modes supply host `expf32`/`log2f32` aliases for the full body's scalar header
calls, so original header/runtime linkage remains unvalidated.

Portable probability absolute tolerance is 2e-6. Native probability tolerance
is `2e-6 + 2e-5*abs(FP64 probability)`. The normalization sum tolerance is 2e-5.
Metadata permits float rounding of `2e-6*(1+abs(FP64 base2 maximum))`; native
polynomial LSE permits another 5e-5. Full uses the host scalar exponential and
receives no polynomial allowance. These are fixed acceptance gates, not
performance or model-accuracy claims.

## Extreme exponential policy and native gate

Actual `avx512_exp_ps` uses a centered degree-six Taylor polynomial for finite
inputs in [-87,88], with a base-two fraction in [-0.5,0.5] and a normal binary
exponent. The mathematical polynomial relative remainder is at most 2.5e-7.
Inputs outside that interval are replaced with zero before integer conversion
and exponent-bit construction; their final results come from platform `expf`.
This fallback covers subnormal results, underflow, overflow, `+Inf -> +Inf`,
`-Inf -> 0` and `NaN -> NaN`. These exponential special values do not extend the
regular Softmax caller contract to NaN or infinite input logits. IEEE nearest
rounding and gradual underflow are assumed; checks do not enable fast-math/FTZ.

The native-only direct gate compares 2,821 actual intrinsic results with
independent FP64 `std::exp`: 2,801 points from -87 through 88 at step 1/16 and
20 extreme/special edges. The bound is relative error 2e-5 plus
`2*float32 denorm_min`; values whose FP64 reference rounds to zero must be zero,
and overflow/NaN signs and classes are checked explicitly. Portable execution
emits a native skip and cannot validate this polynomial.

A separate 24-row matrix checks normalized probabilities and base-two LSE in
all three Softmax variants: sixteen zero logits paired with sixteen -87/-90/
-100/-104/-1000/-FLT_MAX logits, sixteen 10000 paired with sixteen 9000, and
sixteen 1e38 paired with sixteen -FLT_MAX. Local portable and sanitizer checks
pass. Actual native plain and ASan/UBSan/LSan execution, including all2,821
direct cases, passed [PR #92 CI38433831517](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38433831517)
for candidate `52f0c3990289e8b5dddc09fc81adc652ed097bee`, tree
`7b73122e767a294dba31b265dd430d6a834b4a8f`, identical to integration
`59edfec2fe54dd6783afec2a0562aab62f70d84f`. Supported native sanitizer policy
is Ubuntu24/GCC14 with aligned reduction storage (#86) and leak checking enabled.
Unsupported hardware stays a skip for that run; it does not erase the recorded
actual native validation. Full Torch/FBGEMM extension dispatch remains untested.

## Historical characterization

Before the maintained repairs, #58 reproduced 54 full-tail writes and 54
optimized-tail denominator/write defects in safely rounded buffers, exact-buffer
ASan failures, and sparse/full null-load UBSan failures for empty input. The old
native exponent-bit construction also failed the -90/-100/-1000 rows. Those
observations describe the pre-repair implementation, not current expected
behavior. #72 replaces rounded tails with logical-length execution, #73 defines
zero mass explicitly, and #74 bounds exponent construction with special/extreme
fallback. The pinned archive preserves the original source; current regressions
and ledger entries exercise the maintained repairs.
