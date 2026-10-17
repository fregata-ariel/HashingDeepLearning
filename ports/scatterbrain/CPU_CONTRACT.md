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
theorem. The primary supplement has since been acquired: AppendixC Algorithm1
PDF7/printed21 lines21–23 specifies the shared normalizer; AppendixD.3
PDF15/printed29 has conditional moment assumptions, not a normalized-ratio
theorem. See [appendix review](../../docs/SCATTERBRAIN_APPENDIX_REVIEW.md).

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

## Feature-map lifecycle repair (#106)

The original initial release's lifecycle defects were separately executed in
both the frozen maintained preimage481dfd9 and immutable initial archive:
cold deterministic eval retained a transposed zero placeholder, requested
redraw1 was replaced by pi, a warm checkpoint could not load into a fresh
module, and `.to(float64)` was undone by the next sample's stale constructor
dtype. The maintained feature-map class now repairs that state lifecycle;
`softmax_kernel` and `gaussian_orthogonal_random_matrix` arithmetic are unchanged.

Construction consumes no RNG and creates a correctly shaped zero placeholder
`[projection_rows, query_dims]`, where cosh uses `n_features//2` rows. A first
explicit `new_feature_map` or cold query/key encoding samples once, even in
deterministic eval, at counter0. Subsequent eligible explicit requests advance
that counter and draw at multiples of the requested period. Query/key encoding
of a warm map does not advance it, so one SBLocal forward shares one projection.
Warm deterministic eval neither samples nor advances the cadence; training
resumes at that paused counter. `redraw=1` redraws every explicit warm request.

Periods may be positive int64 integers, integral floats in that range, or
positive infinity (sample once/freeze). Bool, other runtime types, zero,
negative, NaN, negative infinity, nonintegral values such as pi, and overflow
are rejected before changing an existing period or consuming RNG. The public
period setter preserves the current counter. Request counter overflow also
raises instead of wrapping. Query dimensions and effective projection rows
must be positive; the historical default expression and odd cosh truncation
remain intact, including requested7 yielding6 effective features.

The state_dict contains `projection_matrix`, `_projection_initialized` (bool),
`_feature_map_calls` (int64) and `_redraw_period` (int64, zero denotes infinity).
Checkpoint period overrides the receiving constructor's period. Cold and warm
loads, including nested modules, do not sample. Training mode and
`deterministic_eval`, other architectural configuration and global Torch RNG
remain caller policy. Exact next-draw reproduction requires restoring both
this lifecycle checkpoint and the separately saved global RNG state.

Legacy projection-only checkpoints with the supported warm shape load at
counter0 and keep the receiving constructor period. Every all-zero legacy
projection is explicitly treated as cold, including the historical transposed
placeholder and an ambiguous shape-correct zero matrix. That is a documented
compatibility policy, not an inferred fact about arbitrary checkpoints.
Intentional warm-zero maps must use the new explicit initialized metadata and
are preserved. Incomplete/invalid lifecycle metadata, nonfinite/nonfloating
projection storage and incompatible shapes are rejected by lifecycle validation
before this map's buffers change. Other strict-key errors and copy/assign
behavior retain standard PyTorch semantics; no general checkpoint transaction
is promised.

Samples take dtype from the current projection buffer after module `.to()`;
the unchanged QR helper still performs its historical CPU/default-dtype draw
before its existing conversion. This does not claim higher-precision QR or GPU
validation. Constructor factory kwargs no longer silently revert a moved buffer.
Explicit device requests retain the sampler's device argument; the validated
execution remains CPU. Ignored `orthogonal`, odd feature truncation and
non-finite/extreme/all-masked attention behavior are not repaired by this change.

The focused fixture tests fixed request schedules, exact sampled matrix/RNG
continuation, cold/warm and nested torch.save/load, scalar FP64 reconstructed
features, dtype migration, invalid-state preservation and substantive mutation
sensitivity. The native fixture additionally executes default-training redraw
and eval transitions against the actual CPU extension and independent dense
attention reconstruction. Conditional fixed-projection comparisons explicitly
freeze a genuinely sampled projection in deterministic eval; they do not
replace sampling or native arithmetic. Later feature-map implementations
remain a separate #106 scope until they are actually executed.
