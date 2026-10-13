# MagicPIG contracts, maintained Python enrollment and CPU gates (#61)

This index describes the selected v0.2 CPU checkpoint. The immutable reference
is `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`; its source inventory must pass
before numerical tests. The contracts/typing changes do not alter arithmetic; the current validation
candidate additionally contains the separately tracked CPU repairs. Archived
Python remains unchanged. The existing native Doxygen
contracts are checked by `tools/audit_traceability_docs.py` against the stable
ledger IDs below.

## Mechanism and ownership contracts

| Mechanism / stable test ID | Contract and implemented boundary |
| --- | --- |
| `MAGICPIG-SIMHASH-CPU`, `MAGICPIG-HASH-DOMAIN-CPU` | [Hash domain](HASH_DOMAIN_CPU_CONTRACT.md): archived FP16 packing is characterized separately; maintained LSH allocation accepts only K1..11 before allocation/state mutation. Accepted key/query codes are checked exhaustively. |
| `MAGICPIG-LSH-RETRIEVAL-CPU` | [Retrieval](LSH_CPU_CONTRACT.md): contiguous int16 sorted key codes, int32 indices/query/output; unique IDs per table and at least two distinct table hits. Copied fill inputs, borrowed int8 mask view and bounded owner lifetime. |
| `MAGICPIG-FASTFILL-REPAIR-CPU`, `MAGICPIG-FASTFILL-REGRESSION-CPU` | [Fastfill](FASTFILL_REPAIR_CPU_CONTRACT.md): guarded int32 unsorted codes, stable token population and request replacement; scoped cursor ownership and clean sanitizer teardown. Historical wrong-table/leak evidence remains tied to its old commit. |
| `MAGICPIG-PROBABILITY-CPU`, `MAGICPIG-PROBABILITY-DOMAIN-CPU`, `MAGICPIG-PROBABILITY-STABILITY-CPU` | [Correction](PROBABILITY_VALIDATION.md): finite accepted scores, positive norms/scaling, K>=1/L>=2 and tolerance-limited cosine clamping. Stable log inclusion probability retains the explicit +1e-4 variant. Server QK/correction is staged before worker dispatch or exposed output writes. |
| `MAGICPIG-ATTENTION-CPU`, `MAGICPIG-ATTENTION-BOUNDARIES-CPU` | [Attention](ATTENTION_CPU_CONTRACT.md): BF16 caches, FP32 query/score/normalizer and int32 IDs/counts. Logical nnz controls QK accesses. Full nnz is rank-one [batch*query_heads], including differing grouped-head lengths; KV ownership uses query_head/groups. Empty output is zero with -Inf maximum/LSE. Explicit dimension/group checks precede output writes. |
| `MAGICPIG-BASELINE-CPU`, `MAGICPIG-SOFTMAX-BOUNDARY-CPU`, `MAGICPIG-SOFTMAX-EMPTY-CPU`, `MAGICPIG-SOFTMAX-EXTREME-CPU`, `MAGICPIG-EXP-NATIVE-CPU` | [Softmax](SOFTMAX_CPU_CONTRACT.md): exact logical allocations, normalized probabilities and base-2 maximum/LSE; empty rows touch no scores and emit -Inf metadata. Native exponential has range/special-value regressions, whose actual AVX512 execution requires a capable runner. Finite metadata requires representability in FP32. |
| `MAGICPIG-BF16-MODEL-CPU`, `MAGICPIG-BF16-BOUNDARY-CPU`, `MAGICPIG-BF16-SELECTED-CPU` | [BF16](BF16_CPU_CONTRACT.md): conversion model is distinct from native dot execution. Native QK requires supported multiple-of-32 dimensions and logical nnz; capability-based skips do not verify native execution. Real conversion linkage/dispatch remains unexecuted. |
| `MAGICPIG-CACHE-MERGE-CPU` | [Cache/merge](CACHE_MERGE_CPU_CONTRACT.md): shaped FP64 host copies, padded prefill, disjoint tokens, heterogeneous probabilities, base-2 host merge, capacity negatives and clear/refill. Host guards/append do not establish native FlashInfer capacity or stream safety. |
| `MAGICPIG-COMPOSITION-BRIDGE-CPU`, `MAGICPIG-CPU-COMPOSITION-CPU` | [Connected composition](CPU_COMPOSITION_CONTRACT.md): actual retrieved IDs feed maintained correction/attention, then selected cache/merge across24 decode steps/12 reuse cycles; independent FP64 union and actual executed stale/correction mutations. Bridge executable/fixture/source/descriptor identities are checked. |

All borrowed views require their owner to remain alive. The selected-body
adapters replace Torch, dependency conversion/copy and scheduling; they do not
establish unrestricted extension or GPU runtime correctness. The independent
composition oracle executes the connected path, rather than inferring it from
separate mechanism passes. Repair code is now present in the validation
candidate; Issue completion still requires reviewed task integration and
matching Hosted evidence.

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
| `traceability.yml` | Source identity/inventory, ledger and native-doc audit; source/capability/driver/boundary contracts; all enrolled MagicPIG portable C++/Python suites; plain and ASan/UBSan/LSan runs. Existing `papers/**`, `ports/magicpig/**`, archive, driver and `tests/traceability/**` filters cover additions; `docs/MAGICPIG*.md` also triggers recovery/doc changes. |
| `python-typing.yml` | Inventory/CI contract regressions, annotation audit, identical strict mypy checks for every nonempty maintained group, existing Reformer and SLIDE checks. `.py`, `.pyi`, inventory, stubs, auditor, regression and MagicPIG documentation changes trigger the gate. |

No new permanent workflow or recurring GPU job is introduced. A temporary
CPU validation-branch workflow records combined/native evidence while GitHub
API access is pending; retire it after reviewed task PR gates supersede it. Routine MagicPIG numerical
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
python3 tools/check_magicpig_cpu.py --native --build-dir /tmp/magicpig-cpu/plain
python3 tools/check_magicpig_cpu.py --native --sanitize --build-dir /tmp/magicpig-cpu/sanitized
```

Mypy requires the existing typing workflow's dependencies. Native AVX512F/FMA
and BF16 bodies execute only after compiler/CPU/OS capability checks; missing
capabilities are explicit skips, not native numerical passes. Python adapters
are not ASan/UBSan/LSan evidence. Leak checks stay enabled in Hosted CI; expected
defects must match their precise enrolled diagnostic and implementation site.

## Pending closeout record

#61 preparation starts from integration merge
`b1a006d2ccaf15eaf8e1e751127101f74317114f` (PR #79 / #59). The integration
[dependency and recovery table](../../docs/MAGICPIG_INTEGRATION.md#mp3-recovery-checkpoint)
is the handoff record. #60's reviewed PR head, merge revision and applicable
Hosted CI URLs remain pending; #61's final PR/head/merge/check URLs must be
recorded after that dependency is integrated and validated. These pending
items prevent parent closeout, not independent contract/typing preparation.

Postponed #62 requires a separate native GPU restart. No CPU result here claims
Torch/FBGEMM extension linkage, FlashInfer execution, asynchronous transfer or
stream safety, production Llama generation, accuracy or throughput.
