# MagicPIG contracts, maintained Python enrollment and CPU gates (#61)

This index describes the selected v0.2 CPU checkpoint. The immutable reference
is `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`; its source inventory must pass
before numerical tests. The contracts/typing changes do not alter arithmetic.
The final #61 base must include reviewed integration of CPU repairs
#65/#66/#67/#68/#72/#73/#75/#76/#77; until then their rows below describe the
reviewed preparation target. Archived Python remains unchanged. Inclusion of
native repair #74 additionally depends on #86 and actual native plain and
sanitized execution; it must not be inferred from portable or compile-only
success. The existing native Doxygen contracts are checked by `tools/audit_traceability_docs.py` against the stable
ledger IDs below.

## Mechanism and ownership contracts

| Mechanism / stable test ID | Contract and implemented boundary |
| --- | --- |
| `MAGICPIG-SIMHASH-CPU`, `MAGICPIG-HASH-DOMAIN-CPU` | [Hash domain](HASH_DOMAIN_CPU_CONTRACT.md): archived FP16 packing is characterized separately; maintained LSH allocation accepts only K1..11 before allocation/state mutation. Accepted key/query codes are checked exhaustively. |
| `MAGICPIG-LSH-RETRIEVAL-CPU` | [Retrieval](LSH_CPU_CONTRACT.md): contiguous int16 sorted key codes, int32 indices/query/output; unique IDs per table and at least two distinct table hits. Copied fill inputs, borrowed int8 mask view and bounded owner lifetime. |
| `MAGICPIG-FASTFILL-REPAIR-CPU`, `MAGICPIG-FASTFILL-REGRESSION-CPU` | [Fastfill](FASTFILL_REPAIR_CPU_CONTRACT.md): guarded int32 unsorted codes, stable token population and request replacement; scoped cursor ownership and clean sanitizer teardown. Historical wrong-table/leak evidence remains tied to its old commit. |
| `MAGICPIG-PROBABILITY-CPU`, `MAGICPIG-PROBABILITY-DOMAIN-CPU`, `MAGICPIG-PROBABILITY-STABILITY-CPU` | [Correction](PROBABILITY_VALIDATION.md): finite accepted scores, positive norms/scaling, K>=1/L>=2 and tolerance-limited cosine clamping. Stable log inclusion probability retains the explicit +1e-4 variant. Server QK/correction is staged before worker dispatch or exposed output writes. |
| `MAGICPIG-ATTENTION-CPU`, `MAGICPIG-ATTENTION-BOUNDARIES-CPU` | [Attention](ATTENTION_CPU_CONTRACT.md): BF16 caches, FP32 query/score/normalizer and int32 IDs/counts. Logical nnz controls QK accesses. Full nnz is rank-one [batch*query_heads], including differing grouped-head lengths; KV ownership uses query_head/groups. Empty output is zero with -Inf maximum/LSE. Explicit dimension/group checks precede output writes. |
| `MAGICPIG-BASELINE-CPU`, `MAGICPIG-SOFTMAX-BOUNDARY-CPU`, `MAGICPIG-SOFTMAX-EMPTY-CPU`, `MAGICPIG-SOFTMAX-EXTREME-CPU`, `MAGICPIG-EXP-NATIVE-CPU` | [Softmax](SOFTMAX_CPU_CONTRACT.md): exact logical allocations, moderate finite-logit normalization and base-2 maximum/LSE; empty rows touch no scores and emit -Inf metadata. Full/optimized Softmax preserve every score beyond logical nnz. Finite base-two metadata requires FP32 representability. The #74 candidate bounds native exponent construction and adds a 2821-case direct regression; actual sanitizer acceptance remains blocked by #86 until repaired and revalidated. |
| `MAGICPIG-BF16-MODEL-CPU`, `MAGICPIG-BF16-BOUNDARY-CPU`, `MAGICPIG-BF16-SELECTED-CPU` | [BF16](BF16_CPU_CONTRACT.md): conversion model is distinct from native dot execution. Native QK requires supported multiple-of-32 dimensions and logical nnz; capability-based skips do not verify native execution. Real conversion linkage/dispatch remains unexecuted. |
| `MAGICPIG-CACHE-MERGE-CPU` | [Cache/merge](CACHE_MERGE_CPU_CONTRACT.md): shaped FP64 host copies, padded prefill, disjoint tokens, heterogeneous probabilities, base-2 host merge, capacity negatives and clear/refill. Host guards/append do not establish native FlashInfer capacity or stream safety. |
| `MAGICPIG-COMPOSITION-BRIDGE-CPU`, `MAGICPIG-CPU-COMPOSITION-CPU` | [Connected composition](CPU_COMPOSITION_CONTRACT.md): actual retrieved IDs feed maintained correction/attention, then selected cache/merge across24 decode steps/12 reuse cycles; independent FP64 union and actual executed stale/correction mutations. Bridge executable/fixture/source/descriptor identities are checked. |

All borrowed views require their owner to remain alive. The selected-body
adapters replace Torch, dependency conversion/copy and scheduling; they do not
establish unrestricted extension or GPU runtime correctness. The independent
composition oracle executes the connected path, rather than inferring it from
separate mechanism passes. Each repair requires its reviewed task integration
and matching Hosted evidence. The #74 range/special-value fixture
has `MAGICPIG-EXP-NATIVE-CPU cases=2821`; it must actually run on AVX512 before
#74 can close. Its successful portable host-exp or native compile-only result
cannot establish native execution, and its proposal remains separate from
the integrated original exponential until that gate passes. A Hosted Ubuntu24
plain sample actually executed it, but sanitized native attention failed at
the local SIMD reduction store (#86); a passing plain phase cannot replace
the failed sanitizer phase.

## Deliberate maintained Python inventory

`tools/maintained_python_inventory.json` records exact `.py` and `.pyi` paths
and each group's mypy search roots. At this checkpoint it contains 31 maintained
Python sources and one maintained MONGOOSE stub. The `magicpig` group explicitly
contains zero maintained Python sources/stubs. Archived orchestration under
`third_party/magicpig/` and selected-AST fixtures under `tests/traceability/` are
not a maintained production Python port and are not a MagicPIG strict-typing
pass. CI reports `PYTHON_TYPING_EMPTY group=magicpig maintained_sources=0`.

The annotation audit compares actual maintained files with the exact inventory;
new sources, stubs, removals, same-count replacements or undeclared maintained
roots fail until deliberately enrolled. Every nonempty group is then checked
with the same `--strict` mypy flag and its isolated search roots. MagicPIG
additionally uses `--disallow-any-explicit` for future native/runtime APIs;
the existing projects retain their NumPy aliases (NumPy uses an `Any` array
shape internally). This replaces the former hard-coded 23/4/4 source lists, without
changing their strict checks or the separate Reformer attention audit.

When a maintained MagicPIG Python port is added, enroll its exact paths in the
`magicpig` group; annotate every function and supply dtype, shape/layout,
mutation, ownership, capacity and paper/test relation docstrings. Add narrowly
scoped Torch/native/FlashInfer stubs for the interfaces actually imported and
retain missing-import diagnostics. No placeholder production port or broad
`Any`/`ignore_missing_imports` stub is justified by the current zero-file scope.
The inventory runner automatically checks enrolled future sources and stubs.

## Existing workflow gates

| Existing workflow | Required CPU gate / trigger coverage |
| --- | --- |
| `traceability.yml` | Source identity/inventory, ledger and native-doc audit; source/capability/driver/boundary/reporting contracts; reproducible probability-reference generator; all enrolled MagicPIG portable C++/Python suites; plain and ASan/UBSan/LSan runs; bounded real OpenMP attention. Explicit generator/reporter/auditor/inventory tool filters, plus existing source/test filters, cover tool-only changes. Exact checkout SHA/tree and native execution/skip results appear in check notices. |
| `python-typing.yml` | Inventory/CI contract regressions, annotation audit, identical strict mypy checks for every nonempty maintained group, existing Reformer and SLIDE checks. Python 3.12 with mypy2.4.0, NumPy2.3.5, TensorBoard2.21.0 and official CPU-only Torch2.14.1+cpu reproduces the validated type environment. `.py`, `.pyi`, inventory, stubs, auditor, regression and documentation changes trigger the gate; the declared mypy compatibility target remains Python3.10. |

No new workflow or recurring GPU job is introduced by #61. The temporary
validation-branch diagnostic is excluded from the task branch and must be
retired once the permanent task gates supersede it. Routine MagicPIG numerical
CI uses tiny deterministic fixtures without CUDA, FlashInfer, model weights or
dataset downloads. The separate existing typing workflow installs Torch for
its maintained MONGOOSE checks; that installation is not MagicPIG extension
execution.

Use these commands from the checkout root:

```bash
python3 tests/traceability/test_python_typing_inventory.py
python3 tools/audit_python_annotations.py --root ports --mypy
python3 tools/validate_traceability.py --emit-dir /tmp/magicpig-ledger
python3 tools/audit_traceability_docs.py
python3 tests/traceability/test_magicpig_source_contracts.py
python3 tests/traceability/test_magicpig_capabilities.py
python3 tests/traceability/test_magicpig_cpu_contracts.py
python3 tests/traceability/test_magicpig_boundary_contracts.py
python3 tests/traceability/test_magicpig_ci_reporting.py
python3 tools/generate_magicpig_probability_reference.py --check
python3 tools/check_magicpig_cpu.py --native --build-dir /tmp/magicpig-cpu/plain
python3 tools/check_magicpig_cpu.py --native --sanitize --build-dir /tmp/magicpig-cpu/sanitized
```

Mypy requires the pinned typing workflow's dependencies. Native AVX512F/FMA
and BF16 bodies execute only after compiler/CPU/OS capability checks; missing
capabilities are explicit skips, not native numerical passes. Python adapters
are not ASan/UBSan/LSan evidence. Leak checks stay enabled in Hosted CI.
`tools/report_magicpig_cpu.py` reports a real native result separately from a
capability skip. It reports #74 pending if its experiment is not enrolled;
once the experiment is enrolled, capable native runs must carry the actual
2821-case marker as a complete token, not a prefix of another count.
Unsupported standard Hosted runners still skip explicitly. Phase logs are kept
with the evidence; reporting runs even after a driver failure. Missing/failed
final evidence produces an API-readable FAIL/INCOMPLETE error annotation with
the actual diagnostic and keeps CI failing. It never becomes a capability skip
or borrows the successful plain phase as sanitized evidence.

The OpenMP gate recompiles the sanitized portable attention translation unit
with `-fopenmp` and runs it under `OMP_THREAD_LIMIT=4` and `OMP_NUM_THREADS=4`.
It verifies the staged caller/worker boundary with the real OpenMP runtime,
while keeping the arithmetic/conversion dependency models explicit. It neither
forces unsupported SIMD flags nor validates unrestricted production scheduling.

## Pending closeout record

#61 formal preparation starts from integration merge
`8ec0115d3e61aec23f812f48966ff7d34286aea8` (PR #80 / #60). The integration
[dependency and recovery table](../../docs/MAGICPIG_INTEGRATION.md#mp3-recovery-checkpoint)
is the handoff record. #60 is integrated and closed; #61's final base after the
required repair merges, PR/head/merge/check URLs remain pending. Record them
before consuming this task. MP1/MP2 milestones are closed; MP3 and parent #52
wait for their remaining applicable gates. MP5 retains repair scope, including
the native #74/#86 gates, independently of scoped CPU mechanism milestones.
The final rebase must consume the actual publisher integration state, including
PRs #81–#88 where reviewed/merged, before final exact-candidate checks. A
temporary diagnostic workflow is not part of the formal branch.

Postponed #62 requires a separate native GPU restart. No CPU result here claims
Torch/FBGEMM extension linkage, FlashInfer execution, asynchronous transfer or
stream safety, production Llama generation, accuracy or throughput.
