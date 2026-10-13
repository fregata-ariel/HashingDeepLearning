# MagicPIG contracts, maintained Python enrollment and CPU gates (#61)

This index describes the selected v0.2 CPU checkpoint. The immutable reference
is `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`; its source inventory must pass
before numerical tests. Maintained native arithmetic and archived Python are
unchanged by the contracts/typing closeout. The existing native Doxygen
contracts are checked by `tools/audit_traceability_docs.py` against the stable
ledger IDs below.

## Mechanism and ownership contracts

| Mechanism / stable test ID | Contract and implemented boundary |
| --- | --- |
| `MAGICPIG-SIMHASH-CPU` | [SimHash/centering](../../docs/MAGICPIG_SIMHASH_CPU.md): selected archived AST, fixed sign/packing order and append centering; FP64 host tensor replaces Torch/BF16. K1..11 supported; K12+ loss is characterized under #65. |
| `MAGICPIG-LSH-RETRIEVAL-CPU` | [Retrieval](LSH_CPU_CONTRACT.md): contiguous int32 sorted codes/IDs and outputs, unique IDs per table, at least two distinct table hits; alloc-once owner storage, copied fill inputs, borrowed int8 mask view, capacity and GQA/batch/layer isolation. Direct under these preconditions. |
| `MAGICPIG-FASTFILL-DEFECT-CPU` | Same retrieval contract: wrong table contents and a temporary-allocation leak are reproduced, with the leak diagnostic strictly validated. This support characterization does not establish fastfill correctness (#66). |
| `MAGICPIG-PROBABILITY-CPU` | [Probability/correction](PROBABILITY_VALIDATION.md): FP32 borrowed score mutation, FP64 independent exhaustive/binomial expectations, nonzero norms and valid angular domain. Released `+1e-4` is a variant of paper correction; endpoint/conditioning repairs #67/#68 remain separate. |
| `MAGICPIG-ATTENTION-CPU`, `MAGICPIG-ATTENTION-BOUNDARIES-CPU` | [Attention](ATTENTION_CPU_CONTRACT.md): contiguous BF16 caches, FP32 query/score/normalizer buffers, int32 selected indices/counts; padding through `round_up(nnz,16)`, alloc-once server lifetime, copied fill and borrowed getter views. Sparse dimensions divisible by 16; full WV dimension128 and groups1/4/8. |
| `MAGICPIG-BASELINE-CPU`, `MAGICPIG-SOFTMAX-BOUNDARY-CPU`, `MAGICPIG-SOFTMAX-DEFECT-CPU` | [Softmax](SOFTMAX_CPU_CONTRACT.md): FP32 score mutation, normalized output and base-2 maximum/LSE; portable host exp is a substitution, native polynomial is capability-gated. Tail, empty-set and underflow characterizations remain repairs #72–#74. |
| `MAGICPIG-BF16-MODEL-CPU`, `MAGICPIG-BF16-BOUNDARY-CPU`, `MAGICPIG-BF16-SELECTED-CPU` | [BF16](BF16_CPU_CONTRACT.md): bit storage/conversion model is distinguished from actual native dot-product execution. Native conversion linkage and runtime dispatch are unexecuted; dimension/padding restrictions remain #75/#76. |
| `MAGICPIG-CACHE-MERGE-CPU` | [Cache/merge](CACHE_MERGE_CPU_CONTRACT.md): fixed-shape FP64 host copies, padded dense prefill, disjoint static/offload/new-token sets, heterogeneous sampled probabilities, base-2 host merge and clear/refill. Host capacity/append guards do not establish native FlashInfer capacity or stream safety. |

All server/view tests require the owner to outlive borrowed views. Real Torch
ownership, dtype/device conversion and concurrent native scheduling are outside
the substitutes. Varying full-attention GQA lengths remain unsupported (#77).
CPU composition (#60) must add its independent combined oracle and recovery
record before #61 closes; independently passing mechanisms do not establish
that combined result.

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

No new workflow or recurring GPU job is introduced. Routine MagicPIG numerical
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
