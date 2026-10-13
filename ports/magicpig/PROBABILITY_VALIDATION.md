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

Near zero inclusion, FP32 subtraction magnifies relative probability error.
The correction bound is `2e-5 + 6e-7/(reference_inclusion + 1e-4)` (ceiling
0.00602): a conditioning allowance for up to `6e-7` absolute inclusion error,
plus the base scalar/log rounding allowance. It is a fixture acceptance budget,
not a proof for arbitrary K/L. The tested maximum correction error is about
0.004164. A separate 141st correction checks K=16,L=2,cosine=0: the positive
FP64 inclusion `2^-32` vanishes in the source subtraction. It must match the
zero-inclusion regularized result within `1e-6` and differ from the FP64
positive-inclusion correction by more than `1e-6`. This is characterization of
a reproduced numerical defect, not a claim of exact small-probability accuracy.

The estimator reference separately enumerates 64 outcomes for two synthetic,
independent candidate table-hit processes. Shared geometric LSH hyperplanes can
correlate candidates; this model does not claim their joint distribution.
Unsmoothed unnormalized Horvitz–Thompson expectation equals its dense target.
Smoothing changes that expectation. Conditional on a nonempty set, finite
self-normalization yields approximately 0.781719 versus dense 2/3, demonstrating
bias of the ratio estimator. No production Softmax or estimator server is
executed by this mathematical example, and no finite unbiased-attention claim
is made.

## Domain validation (#67) and pending stability repair

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
ASan/UBSan/LeakSanitizer runs pass 745 assertions with leak checking enabled.

Native server callers must validate outside OpenMP regions or transport
exceptions to the joining thread; exceptions may not cross an OpenMP region.
Full Torch wrapper exception handling and index ownership are separate server
integration checks. This selected scalar suite makes no Torch/GPU claim.
[#68](https://github.com/fregata-ariel/HashingDeepLearning/issues/68) remains the
separate stable-small-inclusion repair. Its tiny-cancellation characterization
still applies after #67. Deterministic inputs use no RNG.
