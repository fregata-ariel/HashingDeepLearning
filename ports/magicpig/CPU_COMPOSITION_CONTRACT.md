# MagicPIG connected CPU composition (#60)

The `cpu_composition` suite connects the pinned Python SimHash, partition,
centering, host-cache, append-centering, plan, merge and clear statements to
actual maintained C++ retrieval and attention bodies. It does not run a sequence
of independent mechanism suites and infer composition from their success.
`composition_bridge` is its enrolled C++ dependency and also has a standalone
exact-set, attention, clear/refill self-test.

Run through the common driver:

```sh
python tools/check_magicpig_cpu.py --suite cpu_composition --build-dir /tmp/magicpig-composition-plain
python tools/check_magicpig_cpu.py --suite cpu_composition --sanitize --build-dir /tmp/magicpig-composition-sanitized
```

The driver builds the bridge first and passes that exact executable to the
Python fixture. Its `--composition-bridge 1|4|8` protocol keeps one LSH owner
and two attention-cache owners alive throughout both refill cycles. The bridge
returns outputs and intermediates as JSON; subprocess errors, sanitizer
messages, nonzero exits and unexpected trailing output fail the Python run.
The evidence records executable and bridge-fixture hashes, selected source
block hashes, compiler options, and the immutable-source gate.

## Connected execution and independent oracle

The fixed fixture has six initial tokens, one sink and one local token, four
offloaded tokens, dimension 128, one KV head, and GQA groups 1/4/8. It uses
literal dyadic keys/values/queries and fixed coordinate projections (no RNG or
model input). Two cycles change the center, values and offload token order.
Each cycle fills the host cache, produces key codes with actual selected AST
statements, sorts each table in the bridge, and fills the actual maintained LSH.

Two append/query/attention steps per cycle then execute selected plan and
append-centering statements, append into checked fixed-shape host slots,
produce query codes through actual decode AST, and retrieve through actual
`LSH::batch_retrieve`/`retrieve`. The retrieved IDs and nnz are passed directly
to actual `SparseAttentionServer::attention`: QK, released probability
correction, Softmax and BF16 output. Dense-resident attention uses the actual
selected QK/full-Softmax/WV kernels described below. Returned outputs and
base-2 LSE feed the selected original `flashinfer.merge_state` call site with
an explicit CPU merger substitute. The actual selected clear statements call
both actual C++ clears; owner cache/norm/score zeroes and empty retrieval are
checked before changed keys/hash tables are refilled into the same owners.

The independent oracle starts with raw keys, values, queries and disjoint token
IDs. It recomputes the offload mean, exact literal sign codes, and distinct-table
hit count. Every returned selected set, QK dot product, corrected logit,
Softmax probability, per-partition output/base-2 LSE and merged union result is
checked. K=1/L=2 inclusion is computed by independently enumerating the four
binary hit states and accepting `(1,1)`. The sampled estimator uses the released
`-log(inclusion + 1e-4)` variant; static/append tokens have no correction. These
literal correlated projection tables establish deterministic source plumbing,
not the statistical validity of independent random tables or empirical
inclusion rates.

The fixed matrix requires 104 merged head checks, 52 full-selection dense head
checks, 24 decode steps and 12 clear cycles. One half selects two offload tokens;
query signs switch selected sets on the second step and token order switches
again after clear/refill. The other half uses a zero projection to retrieve all
four offload tokens and explicitly takes the uncorrected p=1 dense baseline.
An independent direct attention over every raw token must equal the full union
oracle to 1e-12; actual merged BF16 output uses the stated rounding budget.
All head lengths are uniform and positive, buffers have padded capacity 32,
and only supported dimension/group paths are counted as numerical passes.

## Dense path and existing limitations

The original `SparseAttentionServer::full_attention` calls
`softmax_kernel_optimized`, whose partial-tile denominator includes padded
candidates (known #72). A resident append changes length 2 to 3 to 4, so that
method cannot supply a correct dense partition for this fixture. The explicit
CPU replacement for the unexecuted FlashInfer dense branch instead composes
actual maintained `qk_kernel_full`, `softmax_kernel_full` and
`wv_kernel_dim128_full`. Full Softmax changes padded storage but computes the
valid-subset normalization correctly; rounded capacity is explicitly provided.
This is a supported selected-body composition, not a repaired or validated
original full-attention server path. Full-selection offload similarly executes
actual QK/full-Softmax/WV without probability correction. Selecting every token
in the always-corrected sparse path would generally differ from dense; the
fixture does not assert otherwise.

The portable lane, Tensor, serial-copy and BF16 conversion substitutes are the
existing selected-body adapter models. The composition adapter supplies one
shared Tensor declaration for LSH and attention. Python arithmetic is FP64;
C++ Q/K/V input literals are exactly representable FP32/BF16 in this fixture.
No Torch extension, OpenMP scheduling, FBGEMM linkage/dispatch, real FlashInfer
cache append/merge, CUDA, model weights, dataset, model generation, accuracy or
throughput is executed. Native SIMD/AVX/BF16 instructions are unexecuted by this
suite, including when the common driver runs other capability-gated suites.
Repairs #65–#68 and #72–#77 stay separate and are not closed by this contract.

## Tolerances and actual mutation sensitivity

QK absolute error is below 1e-6, corrected logits and probabilities below 2e-5,
partition LSE below 3e-5, and union LSE below 4e-5. Partition/union output permits
absolute error below 0.004 for BF16 conversion and FP32 accumulation. The
observed maximum merged-output error is approximately 0.000966562; this
observation does not loosen the fixed budgets.

Mutation checks execute changed inputs/paths through the same actual bridge
and selected merge AST, then retain the original independent oracle:

| Mutation | Required rejection | Cases |
| --- | --- | ---: |
| Omit probability correction | Output differs from corrected union by >0.01 | 12 |
| Reuse previous query's lookup codes | Exact selected IDs differ | 6 |
| Reuse previous resident cache without last append | Output differs by >0.01 | 6 |
| Append key with inconsistent/stale center | Output differs by >0.004 | 12 |

Both append steps and changed refill cycles must observably change the output.
The sanitized driver compiles the actual bridge with ASan/UBSan and keeps
LeakSanitizer enabled. The Python host adapter is not sanitizer-instrumented;
its connected calls execute the exact instrumented bridge and propagate its
failures, so sanitizer coverage includes the workload rather than only the
bridge self-test. Immutable archived sources and maintained production bodies
remain unchanged.
