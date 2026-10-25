# G-SLIDE semantic contracts and their sources

This review resolves the questions raised before the named configuration/view
API. Algorithmic intent comes from the paper; concrete released behavior comes
from the pinned implementation. A disagreement is recorded as a variant rather
than silently resolved by changing arithmetic during an API refactor.

## Sources and limits

- [Author-hosted G-SLIDE manuscript](https://panzaifeng.github.io/assets/pdf/tpds22gslide.pdf),
  reviewed 2026-10-25, PDF pages 3, 5-9, sections 2.3, 4.2-4.4, Figures 5-8,
  Algorithms 1 and 4. This is the author manuscript, not a publisher PDF.
  Existing provenance remains URL-only; no local PDF/hash is asserted.
- [Released source](https://github.com/PanZaifeng/G-SLIDE/tree/d93c2f6d0bbf1dd7b96d9c2340ed629c29f4f902),
  archived unmodified in `third_party/g-slide/`; the exact commit, tree and 27
  Git blob identities are in [the pin manifest](../papers/g-slide-source-pin.json).
- Maintained CUDA and SYCL are separate implementations. Their passing tests
  do not establish bitwise equivalence or full paper experiment reproduction.

## Decisions for the maintained API

| Question | Paper evidence | Pinned source evidence | Maintained contract |
| --- | --- | --- | --- |
| Bias in output forward/evaluation | Section 4.4 describes bias addition; Algorithm 4 takes biases as input and initializes the output row from them. It does not separately specify the evaluation caller. | `src/kernel.cu::softmax_fwd_bp_rowmajor_slide_out_knl` starts logits from bias; `src/SoftmaxLayer.cu::forward_dense` only invokes SGEMM, with no bias addition. | SYCL keeps biased logits for both training and evaluation. This agrees with the described forward model; its application to evaluation is our inference. Archived SGEMM-only evaluation remains a distinct compatibility behavior, not the correctness oracle. |
| Multiple/duplicate labels | No explicit duplicate-label rule or multi-label delta equation was found in the reviewed training description. | `src/main.cu::get_batch_data` appends labels without deduplication. Softmax kernels use membership for the numerator and the number of listed entries for the denominator. | Preserve `delta = (1[label contains output]/listed_label_count - probability)/actual_batch`. Distinct labels yield normalized target mass; duplicates are accepted compatibility input and reduce that mass. Never silently deduplicate or claim the paper requires duplicates. |
| Hash concatenation | Section 4.3 / Figure 6 specifies shifts using `ceil(log2(window))`. | `src/LSH.cu` and `src/lshKnl.cu` use an integer cast of natural `log(bin_size)`; addresses are formed by addition of shifted winners. | Preserve and name the released natural-log packing variant, including aliases. Paper packing would be an explicit algorithm change requiring separate index/query/oracle tests. |
| Bucket overflow | No exact overflow-survivor rule was found in the reviewed construction description. | `src/lshKnl.cu` uses `atomicAdd(count,1) % bucket_capacity`; query lengths are clamped to capacity. | Keep bounded ring slots and unclamped insertion counts. Concurrent survivors/order are not deterministic FIFO or reservoir sampling. SYCL uses atomic slot writes; no safety claim is inherited from archived non-atomic stores. |
| Index freshness and rebuild timing | Section 2.3 puts index updates after weight updates; section 4.4 allows periodic rebuilding and mentions a preference for decaying frequency. | `src/Network.cu::train` rebuilds after all weight/bias updates when rebuild or reshuffle is requested. `src/main.cu` derives periodic flags from configured sample periods divided by maximum batch size. | Keep initial construction, post-update optional rebuild and reshuffle-implies-rebuild. `StepOptions` expresses a single step's decision. The application owns cadence; no unimplemented decay schedule is claimed. |

For example, labels `[3,3]` give output 3 target mass `1/2` under the released
membership/count rule, rather than `1`. With one sample and probabilities
`[0.25,0.75]` over IDs `[1,3]`, the signed deltas are `[-0.25,-0.25]`.
This characterization explains why distinct labels are the recommended input.

Two existing numerical qualifications remain: archived Softmax adds `EPS` to
the normalization denominator, whereas the SYCL reference uses the exact
max-subtracted sum shown in Algorithm 4; and the maintained per-step hidden
delta reset repairs stale cross-step accumulation. Neither qualification is
introduced by the named API refactor. Adam retains positive signed deltas,
fixed beta/epsilon values, caller-corrected learning rate and gradient consumption.

The paper also mentions layout conversion when saving for another framework
(section 4.2). The pinned Network/Layer/main interfaces expose no such save/load
API. Therefore serialization is mentioned by the paper but not implemented in
the selected public source surface; it is outside this API change.

## API consequences

`configuration.hpp` separates model dimensions, index parameters, workspace
limits, execution partitioning and initial values. `make_workspace_plan` computes
all allocation extents before device allocation, including overflow checks.
Bucket capacity is independent of output count, and controls gathered workspace
capacity. The old positional constructor delegates to this API.

`views.hpp` is SDK-free C++17. `Span` describes asserted allocation capacity;
`MatrixView` adds logical rows/columns and named contiguous row/column layout;
`SegmentedIdsView` separates sample count, used IDs and allocation capacity.
The current operations support these two contiguous layouts, not arbitrary
strides. Matrix shape and capacity failures are host-testable.

`component_api.hpp` validates descriptors, retained context provenance and runtime
USM information before launching
the existing numerical operations. Views borrow; owners allocate and release.
External pointers still require truthful capacities, compatible lifetime,
non-destructive aliasing and valid offset/ID payloads. Owner views carry a
borrowed context token. External USM must use `borrow_usm` with a persistent
context object and an explicit allocation/context assertion. This matters
because the OpenMP runtime recognizes ordinary host memory too; pointer
queries alone cannot establish ownership/context. Metadata checks cannot
infer an arbitrary pointer's real allocation bounds or prove device payload
contents. Network validates host batches and creates its own index/active data.
The original raw-pointer functions remain compatibility/internal interfaces.

For example, a caller can set named fields without relying on positional
integer arguments (the vectors and queue below are supplied by the caller):

```cpp
NetworkConfig config{};
config.model = {3, 4, 4}; // input, hidden, output dimensions
config.limits = {2, 4, 2}; // samples, input nonzeros, label entries
config.index.bucket_capacity = 4;
InitialState initial{hw, hb, ow, ob, permutation_bins};
Network network(queue, config, initial);
network.train(batch, StepOptions{corrected_rate, true, false});
usm_array<float> logits(queue, batch.size() * config.model.outputs);
network.evaluate_dense(batch,
    {logits.view(), batch.size(), config.model.outputs, Layout::row_major});
```

`Span` does not own its context token. Copying a view never extends buffer or
context lifetime. Owner-generated views borrow both from their owner; external
adapters borrow a persistent context supplied by the caller. The owner itself
keeps a queue/context copy, so destroying the original caller queue is safe.

`ModelState`, `IndexState` and `TrainingWorkspace` separate persistent parameters
and optimizer state, persistent index state, and reusable temporary storage.
Within each parameter, gradients and Adam moments/velocities retain distinct
buffers. `OutputGradients` and `AdamBuffers` expose mutation explicitly:
backward accumulates; Adam consumes and clears gradients. The controller resets
hidden deltas once per step. Diagnostic stale-state injection remains outside
`StepOptions`. All calls remain synchronous. Network retains its own queue copy.

No virtual backend hierarchy, new model topology, checkpoint format, automatic
device selection, changed default arithmetic or performance claim is added.

## Acceptance

The required suite registry adds standalone SDK-free configuration and view
tests to the existing components: four host suites, ten device components and
one connected model. Device tests call typed dense, selection, output/backward
and optimizer APIs directly against independent numerical oracles. They also
exercise undersized buffers and compare the positional and named Network paths.
Duplicate-label characterization and queue/context/lifetime checks are recorded
separately from normal distinct-label model acceptance. Execution identities
and results are retained in [the named API evidence](evidence/gslide-named-api-2026-10-25.json):
15/15 CPU and 15/15 NVIDIA suites pass; ordinary GNU C++ 13.3 passes all four
host suites. The failed pointer-query-only diagnostic is retained separately
from the passing explicit-context implementation.
