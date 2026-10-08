# G-SLIDE Softmax CPU arithmetic coverage

Issue #38 adds independently expected probabilities and compressed deltas for
`softmax_fwd_bp_rowmajor_slide_in_knl` and
`softmax_fwd_bp_rowmajor_slide_out_knl`. The existing
`softmax_fwd_bp_rowmajor_all_sm_knl` runs through the same oracle. These support
training in G-SLIDE Section 4.4; the test does not reproduce paper accuracy or
performance.

Run from the repository root:

```bash
python3 tools/check_gslide_cpu.py --suite softmax --build-dir build/gslide-softmax
```

The driver extracts unchanged maintained bodies at runtime. The suite includes
the baseline ten definitions and these two additional Softmax bodies. Generated
evidence records their hashes, compiler command, inputs, archive verification,
and `gpu_validated: false`.

The test expands each sample's CSC inputs into a dense vector, computes the
complete 6-by-4 row-major matrix product and biases in double precision, then
selects active outputs for max-subtracted Softmax. Its denominator includes the
literal retained epsilon `1e-8`. Each active label receives target mass
`1 / label_count`; expected deltas are `(target - probability) / batch_size`.
Every variant is checked against those expected values. Agreement among the
variants alone is insufficient to pass.

The four-sample fixture uses unsorted node IDs and distinct input, output and
label counts. It contains an empty input with nonzero biases, one sample with a
singleton candidate/label, and samples with multiple distinct active labels.
Each sample also runs separately with batch size one. Bias shifts of zero,
-1000 and -16384 exercise the maintained `MAX_INIT=-FLT_MAX` regression. Exact
binary fractions keep these shifts from obscuring arithmetic with float bias
rounding. Three variants run 45 fixtures covering 72 sample evaluations; a
probability and a delta are checked for every active output, and CSC identities,
input values and offsets must remain intact.

## Contracts for the maintained kernels

The caller owns all CSC buffers, row-major weights, biases and compressed delta
storage. Kernel arguments borrow that storage; outputs and deltas are mutated,
and no pointer is retained. Input/output IDs must index the supplied dimensions,
and offsets must delimit valid contiguous buffers. Active outputs and the batch
must be nonempty. Labels must be nonempty, unique, and contained in active
outputs. Synchronize the production CUDA launch before reusing or releasing any
borrowed storage.

| Kernel | Capacity arguments | Dynamic shared storage in 32-bit words |
| --- | --- | --- |
| `slide_in` | `max_out_num >= largest active output count`; `max_label_num >= largest label count` | `2 * blockDim.x + 2 * max_out_num + max_label_num` |
| `slide_out` | `max_in_num >= largest active input count`; `max_label_num >= largest label count` | `2 * max_in_num + max_label_num` |
| `all_sm` | All three capacities cover their largest sample counts | `2 * max_in_num + 2 * max_out_num + max_label_num` |

Multiply these word counts by four to obtain dynamic shared bytes for this
float/int layout. These formulas exclude static shared reduction storage.
Deltas are label-minus-probability, averaged by `gridDim.x`. The test uses exact
maximum sample counts as capacities, including zero input capacity for its
standalone empty-input case.

## Evidence boundary

Every shared-memory body executes with exactly one CPU thread per block. The
adapter supplies scalar `block_max`/`block_reduce`, host exponential/division,
serial atomics, and host-owned arrays using the actual data-structure headers.
The production warp helpers, barriers across multiple threads, CUDA allocator,
adaptive dispatch, launch resource limits, GPU numerical behavior and complete
training wiring remain pending GPU validation. This suite adds no source
arithmetic change and does not modify the archived reference.
