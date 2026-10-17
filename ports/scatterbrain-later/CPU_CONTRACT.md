# Later SBSmyrf maintained CPU contract

The maintained implementation executes the selected actual `SBSmyrfAttention`
forward, XBOX+ transforms, floating projection/rank clustering, residual
composition, overlap correction and common normalizer. It uses the typed
initial feature/projection/linear/masking modules through explicit imports.
It does not interpret E2LSH output as integer buckets or claim collision-law
reproduction. Static input checks do not consume RNG or change model buffers.

Supported construction uses CPU float32 or float64, positive integer counts,
compatible equal padded query/key cluster counts, positive finite radius and
temperature, and at least two requested cosh features. Odd requests retain
`2*floor(n/2)` effective features. Noncausal, dropout0 and feature eps0 are the
validated mode. Causal/arbitrary attention masks, CUDA/meta/half tensors,
nonfinite inputs, invalid prefix lengths and mismatched Q/K/V dimensions,
dtype or device are rejected. Inputs are `[B,T,H,E]`, `[B,S,H,E]`, `[B,S,H,D]`.
Masks are CPU `LengthMask` prefix lengths in `[0,S]` with int64 dtype.

Query/key/value and the feature buffer must agree on dtype/device. #104 threads
query dtype into both actual E2LSH random allocations without changing global
default dtype or casting input to float32. #105 accumulates returned weights
with `dots.new_zeros`. Count-log correction uses `inner.dtype`; a restoring
`.float()` mutant must fail strict raw H/HV/H1 even when the normalized ratio
cancels its error. Float32 first-forward RNG draw order, sampled alpha/beta,
XBOX transforms, rank permutations, projection, outputs and raw/normalized
weights match the pinned original exactly under the validated dependency set.

Feature state is owned by shared repair #106: new training forwards redraw at
the declared default cadence; conditional derivatives explicitly freeze a
projection after a real hash/feature forward. Continued training after a module
float32→float64 move is accepted only with that repaired shared core, with a
real second forward and independent scalar raw/output checks. Constructor dtype
and coherent module movement are tested; no process default-dtype mutation is
used. Historical failures and mixed-precision budgets remain unchanged in the
frozen-reference driver; maintained success does not rewrite those observations.

The numerical acceptance domain is the deterministic moderate fixtures with
unclamped positive normalizers, plus separately checked zero-key-mass rows.
Finite input/norm checks are useful rejection contracts, not a proof covering
all finite extremes or clamps. Actual nonfinite output/weights are rejected;
no broad extreme-score stability or arbitrary-mask accuracy claim is made.
The source's inferred self-attention query-mask policy when padded query/key
lengths match is retained; it is not an assertion of general cross-mask
semantics. T3/S5 fixtures pad to4/6 and do not independently cover that heuristic.

The independent scalar oracle reconstructs true selected support from validated
actual padded permutations and raw Q/K/V/projection, never from the returned
residual-value mask. A selected exact-zero residual case demonstrates that
mask's limitation. It independently checks raw H, HV, H1, normalized outputs,
returned weights, output=weights@V, finite-difference conditional Q/K/V gradients
and gradients through returned weights. Hash source parity and dtype/device
sampling are checked; independent probability/theorem/hash-math derivation is
not claimed. No native extension, sanitizer, GPU, model-quality or performance
validation follows from these CPU Torch tests.

The driver rejects removal of alpha/beta dtype, returned-weight dtype and
count-log dtype fixes, subtraction/duplicate/normalizer errors, common scaling
invisible to output ratios, and late NaN/±Inf output/weight injections. Both
actual/reference values and errors must be finite. Invalid runs remove the old
success summary. Downloaded source and mutant copies are temporary and never
included in emitted artifacts. Strict checks cover every new tracked Python
file with `--strict --disallow-any-explicit`; no fetched archive is represented
as a fully typed maintained component.
