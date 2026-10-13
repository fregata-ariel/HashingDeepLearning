# Probability correction validation (#57)

`probability` executes the selected maintained `transform_kernel` from the
maintained v0.2 source. It uses standard C++ and no Torch, CUDA, model or dataset.
Run `python3 tools/check_magicpig_cpu.py --suite probability` for the focused gate;
the default driver and existing traceability CI enroll it automatically.

## Independent reference and scope

The fixture enumerates 15,108 weighted bit-event states: K=1..3, L=2..4,
bit collision probabilities 0.25/0.5/0.75. A table hits only when all its K bits
match; inclusion requires at least two tables. The exhaustive event result is
compared against an independently summed FP64 binomial distribution (absolute
bound `2e-13`), without using the source's cancellation expression.

140 bounded corrections cross cosines -1, -0.9, -0.5, 0, 0.5, 0.9, 1;
K=1..4; L=2,3,5,8,16. Endpoints are included. Inputs are supplied scalar dots,
unit norms and scale 2; QK kernels and LSH allocation/scheduling are not run.
The FP64 reference uses `acos`, the binomial mass and the source's explicit
`+1e-4` smoothing. That smoothing changes the unsmoothed paper estimator.

## Stable tiny inclusion (#68)

The maintained helper `magicpig_inclusion_log_probability` returns the natural
logarithm of the at-least-two-table probability. It computes the bit collision
log probability with `log(acos(-cosine)/pi)` for nonpositive cosine and
`log1p(-acos(cosine)/pi)` for positive cosine. These symmetric angular forms
avoid cancellation near -1 and avoid rounding a near-one collision probability
before a large K magnifies its logarithm near +1.
For expected table hits `L*p <= 0.5`, it starts with the positive two-hit
binomial mass in log space and sums subsequent positive masses by recurrence.
Successive ratios are below 0.25, so work remains bounded even for INT_MAX
counts. For larger expected hit counts, the excluded zero/one-hit mass is
computed with `log1p` and the included mass with `expm1`; this branch stays
away from the severe cancellation region. Intermediates use `long double`.

The logarithm remains finite for positive inclusion below exponent range.
Cosine=-1 is the genuine zero endpoint and returns negative infinity;
cosine=1 returns zero. The transform combines inclusion with the explicit
`+1e-4` smoothing through log-addition before FP32 score conversion. This
retains the released estimator variant; it does not remove its bias.

The original 141 corrections now use a fixed absolute budget `2e-6` rather
than the old cancellation allowance. The cos=0,K=16,L=2 regression deliberately
replaces the old known-defect expectation: true inclusion `2^-32` must agree
with the positive FP64 reference and differ from cancelled zero.

330 stable-probability cases include 54 independently generated 160-digit
Decimal references (positive exhaustive binomial masses, stored as rounded
long-double literals), 252 extended-precision cosine/K/L cases, two algorithm
transition cases, four large-count/underflow/endpoint cases and 18 direct-double
near-+1 corner references. The additional angular oracle derives acos using a
Decimal positive asin series and Machin pi, without host libm. Exact binary
cosines `1-2^-53`, `1-2^-52`, `1-2^-40` cross K=1/16/INT_MAX and L=2/512.
The old near-one collision formula fails this new regression before repair.
K reaches
2048 in Decimal references and INT_MAX in the exact L=2 log identity; L reaches
512 in full references and INT_MAX in a saturation check. The retained bit
outcome enumeration still covers 15,108 states independently.

Log comparison budget is `2e-11 + 1e-15*abs(reference_log)`. Probability relative
error must be at most `5e-12` whenever the probability is normally representable
in double. These are verified fixture budgets, not a claim of arbitrary-libm
correct rounding. Observed maximum log error is `1.034e-7` in the INT_MAX-K
case (log magnitude about `2.977e9`); maximum relative probability error is
`4.943e-14`. Original plus additional regularized correction error is at most
`4.682e-7`, within the `2e-6` FP32 budget.

Reproduce reference literals with
`python3 tools/generate_magicpig_probability_reference.py --check`.

The estimator reference separately enumerates 64 outcomes for two synthetic,
independent candidate table-hit processes. Shared geometric LSH hyperplanes can
correlate candidates; this model does not claim their joint distribution.
Unsmoothed unnormalized Horvitz–Thompson expectation equals its dense target.
Smoothing changes that expectation. Conditional on a nonempty set, finite
self-normalization yields approximately 0.781719 versus dense 2/3, demonstrating
bias of the ratio estimator. No production Softmax or estimator server is
executed by this mathematical example, and no finite unbiased-attention claim
is made.

## Domain validation (#67)

The maintained boundary now throws `std::invalid_argument` for nonfinite scores,
nonpositive or nonfinite query/key norms and scaling, negative counts, K<1,
L<2, negative indices or null buffers for nonempty input. Valid empty input
is a no-op. Indices still require caller-provided upper bounds; the raw kernel
signature has no cache length. Cosine overshoot at most eight FP32 epsilons is
clamped to the nearest endpoint; larger violations are rejected. Double norm
products prevent finite FP32 norms from overflowing or underflowing during
normalization. Nonfinite corrected outputs throw `std::overflow_error`.
All corrected entries are prepared before mutating the borrowed score buffer.

42 domain regressions cover accepted endpoint roundoff and extreme finite norm
products, nonfinite and zero/negative inputs, invalid counts/parameters, null
buffers, later invalid entries, and unrepresentable corrected scores. Invalid
entries leave scores and input buffers unchanged. The independent moderate
FP64 and estimator references above are preserved. The focused plain and
ASan/UBSan/LeakSanitizer runs pass 3203 assertions with leak checking enabled.

Native server callers must validate outside OpenMP regions or transport
exceptions to the joining thread; exceptions may not cross an OpenMP region.
Full Torch wrapper exception handling and index ownership are separate server
integration checks. This selected scalar suite makes no Torch/GPU claim.
[#68](https://github.com/fregata-ariel/HashingDeepLearning/issues/68) is the
separate maintained stable-small-inclusion repair described above. Deterministic
inputs use no RNG. Portable baseline and attention composition sanitizer runs
also pass; native SIMD was skipped by the CPU capability guard in this machine.
