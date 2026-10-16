# Initial local hybrid CPU contract

The baseline is the supplied NeurIPS 2021 conference manuscript, p6
Section4.2. Eq(1) defines a residual on selected sparse support:
`S_ij = exp(q_i^T k_j) - phi(q_i)^T phi(k_j)`.
Eq(2) factors the unnormalized product as
`phi(Q) (phi(K)^T V) + S V`. The local support is an explicitly permitted
variant discussed after Eq(2). The initial implementation uses cosh features,
Gaussian-orthogonal projections, restored log scales, and a shared row
normalizer with a `1e-24` lower bound on its low-rank remainder. Those details
are source-level variants and support mechanisms. They do not establish the
printed entrywise-Gaussian sampling distribution, a variance/unbiasedness
theorem, or the absent AppendixC/D normalization/proof text.

## Numerical boundary

`SBLocalAttention.forward` calls the real pinned
`fast_transformers.local_product` autograd wrapper, whose local dot-product,
weighted-average and backwards execute the separately built CPU C++ extension.
The typed stubs are declarations of that boundary, not numerical substitutes.
The declared local-product arguments are CPU Tensor Q/K/V, an additive mask,
key lengths, and an integer context. Dtype/device/shape validation is not
supplied by Python type checking.

The bounded validation path is FP32 CPU, noncausal, dropout0, eps0, moderate
finite Q/K/V and positive finite row mass, with one actually sampled projection
held fixed for the independent reference. Small contexts cover odd/even
windows, heterogeneous key lengths, and full local coverage. Independent
FP64 reconstruction checks the full low-rank matrix, exact local replacement,
row normalization, output and gradients. This checks the source composition
conditioned on the supplied projection; it does not prove its sampling law.

Exact fixture cases, errors, build/source hashes and mutation observations
belong to the CPU driver's emitted evidence. A successful Torch extension
build or numerical run is not ASan/UBSan/LSan evidence: native memory safety
has not been validated by this port. No causal/CUDA/Apex runtime or model
training, generation, accuracy or speed claim is made. The original lazy
causal import and optional Apex control flow remain present and typed.

## Type boundaries and retained behavior

The five runtime modules and three package markers are enrolled together with
five narrow stubs. `mypy --strict --disallow-any-explicit` passes all13 files;
no broad import-ignore, untyped function, blanket `Any`, or replacement Torch
stub is used. The variadic feature-map factory has a private constructor
protocol returning `FeatureMap`; its arguments remain forwarded unchanged.
Identity casts narrow Einops outputs, the legacy `torch.cuda.HalfTensor`
class, the boolean result of a boolean Tensor's `item`, and the scalar Tensor
accepted by runtime `torch.arange` but omitted by current Torch stubs. No cast
converts a Tensor/value at runtime. Type checking targets Python3.10 syntax;
the tested modern runtime and dependencies are recorded by the CPU driver.

Cold `deterministic_eval`, the original projection-buffer shape, the irrational
`math.pi` redraw period, ignored constructor options, odd feature counts and
zero/extreme/all-masked domains are preserved for explicit characterization in
[#106](https://github.com/fregata-ariel/HashingDeepLearning/issues/106). The
historical code's behavior on those domains is not silently promoted to
support. This first port introduces no source algorithm fix.
