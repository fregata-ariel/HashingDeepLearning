# Scatterbrain feature/projection CPU report (#97)

`SCATTERBRAIN-FEATURES-CPU`, relation `variant`, covers the maintained initial
SBLocal feature map at main paper p6 §4.2. The ordinary imports are
`src.models.modules.attention.feature_maps_sb` and `projection_utils` from
`ports/scatterbrain`. Upstream lineage is
`eeb689c00b9f322f33f6b6f85eaaac0a1d6a5eab`.

## Command and observed result

```sh
python tests/traceability/test_scatterbrain_feature_cpu.py --emit-dir scatterbrain-cpu-artifact/features
```

For isolated preparation, `--port-root /tmp/scatterbrain-maintained/ports/scatterbrain`
selected the actual typed maintained modules. The default selects the integrated
repository port; there is no archive fallback or replacement numerical helper.
Python 3.12.14, Torch 2.14.1+cpu, einops 0.8.1, x86_64 CPU, one Torch thread:
The PR107 baseline passed 2,126 assertions. After the #106 repair, 2,122 assertions
pass across the same 72 dense feature cases, 12 restored sampled kernels,
24 input-gradient cases, 24 projection-gradient cases, 16 projection cases, and
three rejected arithmetic mutants. Lifecycle assertions now check repaired
behavior; the separate state fixture supplies the detailed regression below.

The JSON evidence contains passed assertion/case counts, an empty known-failures list,
SHA256 of both executed maintained sources and the test, runtime metadata and its
SHA256, error budgets, observed maximum absolute errors, and explicit false flags
for native dependency execution, GPU execution, statistical unbiasedness, and
variance reduction. It is written only after all checks succeed. Existing evidence
is removed before importing Torch or resolving the port. Independent negative CLI
probes confirmed that both missing Torch and a nonexistent port produce a nonzero
exit, no stdout pass record, and no stale evidence file.

## Independent numerical oracle

For each fixed projection row `p`, scalar FP64 arithmetic computes
`exp(±sqrt(t) * p.x - t * ||x||² / 2) / sqrt(M)`. Both signs are included for cosh;
noncosh uses only the positive sign. Scalar Python dot products, squared norms,
`exp`, and `max` define the raw features and the distinct query/key log scales.
The reference explicitly incorporates the input-dtype projection cast, nonzero
stabilizer, and logarithmic-output mode. The test checks features, log scales,
restored features, and the dense dot product of the finite sampled feature maps.
It does not compare this finite sampled kernel to exact `exp(t * q.k)` or infer
sampling guarantees.

Input and projection derivatives use detached raw FP64 feature values and the
analytic derivatives of this scalar expression. The reference does not call the
maintained helper, use its amax/einsum graph, or obtain gradients from a duplicate
implementation via autograd. Actual gradients do run through the maintained
helper and its stabilization. Gradients cover eps0 and nonlog restored feature
output only, across query/key modes, cosh/noncosh, default and two explicit
temperatures, and FP32/FP64 inputs. Log output and nonzero eps are forward-only
checks.
Nonzero eps with log output must raise its exact `NotImplementedError` message.

FP32 uses absolute and relative tolerances 3e-6; FP64 features/derivatives use
8e-15 absolute plus 5e-14 relative. Finite actual values, references and errors
are explicitly required before comparisons. The largest observed FP32 error was
3.83e-7 for a projection derivative. In-memory mutations of the actual helper
that halve the norm correction, duplicate the positive cosh half, or replace
`sqrt(M)` scaling by `M` all give finite errors exceeding 1e-3 against the oracle.
No archived or maintained source is modified by these mutant checks.

## Projection checks

Shapes `(1,3)`, `(3,3)`, `(7,3)`, `(4,1)` include partial blocks, full blocks,
multiple blocks, and one-dimensional projections. Both scaling modes and
FP32/FP64 default/output dtypes are exercised. Each block has orthogonal rows;
fixed scaling gives row length `sqrt(D)`. Gaussian scaling is compared against
independently replayed normal-vector lengths. RNG state must advance, and
restoring a seed state must reproduce both output and subsequent RNG state.

The sign/transpose oracle checks the defining QR property: normalized projection
rows multiplied by the original sampled input form an upper-triangular matrix
with positive diagonal. It does not compute a second QR or transcribe the QR
implementation. Projection tolerances are 2e-6 for FP32 and 2e-14 for FP64.
Invalid scaling must raise `ValueError` with the explicit invalid-scaling message.
No statistical distribution claim follows from these deterministic checks.

## Repaired lifecycle (#106)

- Cold deterministic evaluation initializes lazily, including direct query/key
  encoding, without consuming RNG in the constructor.
- Requested positive integral periods and +inf are honored; warm deterministic
  evaluation freezes projection and cadence. Query/key encoding shares a sample.
- Saved projection, initialization, counter and period load into a fresh module;
  exact future random samples also require caller-owned Torch RNG restoration.
  Projection-only legacy checkpoints follow the explicit migration policy in
  the port contract; ambiguous allzero legacy projections become cold.
- Redraw uses the current projection buffer dtype after `.to(...)`.
- Cosh with seven requested features returns six, preserving the initial truncation.

`test_scatterbrain_feature_state.py --port-root ports/scatterbrain --evidence PATH`
checks four schedules, 40 independent scalar FP64 feature expressions, six
checkpoint paths, 31 invalid-input rejections and five actual-source mutants.
Its preimage mode reproduces four failures in the separate frozen baseline or
immutable original; it does not relabel original defects as repaired source.
Native local-product builds,
models, datasets, long-sequence performance, theorem assumptions, unbiasedness,
variance reduction, and GPU behavior remain outside this fixture.
