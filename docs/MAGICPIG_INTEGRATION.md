# MagicPIG provenance and CPU-first integration

Paper baseline: [arXiv:2410.16179v4](https://arxiv.org/html/2410.16179v4),
*MagicPIG: LSH Sampling for Efficient LLM Generation*.
The versioned primary HTML is identified; no local paper PDF is claimed.

## Selected reference lineage

| Lineage | Commit / tree | Role |
| --- | --- | --- |
| `v0.2` | `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb` / `87f5403c5d453af27bd03f1ec243f8c2098c0c80` | Selected working baseline; class-based CPU extensions and FlashInfer orchestration |
| historical `main` | `dc682a3d98bc4beddbb71e6679df1bdd69e341dd` / `3d4893ca4ce1eac406e2b28d3c8bd899bc59a28a` | Comparative lineage; function-based extensions and `models/cache.py::KV_Cache`; not imported |

Source: [Infini-AI-Lab/MagicPIG](https://github.com/Infini-AI-Lab/MagicPIG).
The selected commit dates to 2024-12-16, historical main to 2024-11-27.
Selection follows the upstream v0.2 class/API organization and documented
FlashInfer/prefill changes. It is not evidence of the exact paper experiment
commit, runtime compatibility or measured speedup. Main is not a renamed copy.

## Explicit archive scope

`third_party/magicpig/` is a **selected source subset**, not a full repository
checkout: 25 first-party reference/build/test/example files plus 16 dependency
source/include/license files, 41 exact blobs and 371,952 bytes in total.
The complete upstream tree has 7,227 blobs. Every upstream tree/blob path, mode,
Git SHA and selection/exclusion is recorded in
`papers/magicpig-upstream-manifest.tsv`; selected blobs and dependency subtrees
are pinned in `papers/magicpig-source-pin.json`.

Included first-party scope: root README/LICENSE/install/requirements,
`models/`, `examples/`, `library/lsh/`, and first-party `library/sparse_attention/`.
Artwork, data, evaluation wrappers, `.gitignore` and unneeded dependency files
are excluded by the inventory. Pretrained weights and benchmark data are not
downloaded by CI.

Source/include closure includes the five FBGEMM `.cc` inputs named by upstream
setup, `src/RefImplementations.h`, seven FBGEMM headers, cpuinfo's public header
and the applicable licenses. FBGEMM subtree is
`8a30c6128d8e63ee3f44a77001ce6687824647b7`; embedded cpuinfo subtree is
`62cae83bdd7895ff07a0df7a48d976049d2be06e`.
Root Apache-2.0, FBGEMM BSD and cpuinfo BSD notices remain byte-identical.

This is a source/include closure, not a complete link/runtime closure.
`FbgemmBfloat16Convert.cc` and `Utils.cc` call cpuinfo implementation functions;
upstream setup does not compile/link the archived cpuinfo implementation.
Torch wheel headers, exported symbols, ABI compatibility and exact native
library provision remain unverified. A future native build must explicitly
resolve that obligation using the pinned dependency lineage or verified Torch
provision. Do not fetch a moving cpuinfo HEAD or infer link success from headers.

`tools/check_magicpig_sources.py` reconstructs all 449 Git tree objects,
including the root, from the full inventory and checks their SHA identities.
It checks the exact selected archive file set, lengths and Git blob identities,
license/input enrollment and dependency subtree identities without network use.
Mutation tests reject changed source, missing license, extra archive file,
excluded-blob inventory tampering and unsafe paths. These are provenance tests.

Four native `.cc`/`.h` originals form the editable `ports/magicpig/` baseline.
No arithmetic/ABI change or new maintained Python source is introduced here.
Issue #54 adds two support records and Doxygen contracts after selected-body
CPU oracles pass. The immutable archive is unchanged.
[CPU_VALIDATION.md](../ports/magicpig/CPU_VALIDATION.md) records substitutions,
capability skips, tolerances and the common suite interface.

## Paper-to-code intake map

The mappings below schedule mechanism-level verification. Issue #54 establishes
restricted correction/Softmax and BF16 QK support tests; it does not complete the
remaining mechanism tasks.

| Paper location | Reference code | Relation / verification boundary |
| --- | --- | --- |
| §4.1, §4.3 Eq.9 | `sparse_attention.cc::transform_kernel` and attention path | Support/direct estimator arithmetic; distinguish theoretical SNIS from the practical sampling-set estimator |
| §4.3 Eq.10 | `lsh.cc::LSH::retrieve`; `transform_kernel` | Two-table inclusion and probability correction; deduplicated selection |
| §4.3 / Algorithm 1 | `attnserver.py::LSHSparseAttnServer.fill`, `decode` | SimHash, code packing and key centering; projection execution remains pending |
| §4.4 | `attnserver.py::decode`; `llama.py::layer_prefill` | Split CPU/GPU attention, normalizer merge and overlap; native scheduling/performance pending |
| §4.2 Eq.8 | conceptual augmented-vector transform | Prerequisite rationale, not a directly implemented feature; §5.1 says experiments omit it |
| Dense/evaluation support | `AttnServer`, `full_attention`, examples | Baseline/support; pretrained generation and accuracy/performance pending |

Intake observations to characterize in separate numerical tasks:
`log(probability + 1e-4)` regularizes the paper correction; AVX exponential
evaluation is approximate; BF16 storage/accumulation is implementation-specific;
normalizers are emitted as base-2 log-sum-exp for FlashInfer merge. Do not mix
these units with natural-log LSE. Zero norms, acos rounding, empty subsets,
packing ranges and vector tails are questions, not confirmed defects here.

## Task ownership and recovery

Integration branch: `research/lsh-lineage-vendor`.
Initial MagicPIG base: `0c4d05887edfd0dbc2625653bace913679a2da93`.
The formal milestones are MP1 #9, MP2 #10, MP3 #11 and deferred MP4 #12.
Current assignments/status are in the linked GitHub issues.

| Task | Proposed branch | Dependencies |
| --- | --- | --- |
| [#53](https://github.com/fregata-ariel/HashingDeepLearning/issues/53), provenance | `research/magicpig-mp1-provenance` | none |
| #54, portable CPU baseline | `research/magicpig-mp1-cpu-baseline` | #53 |
| #55, SimHash/centering | `research/magicpig-mp2-simhash` | #53, #54 |
| #56, retrieval/state lifetime | `research/magicpig-mp2-retrieval` | #53, #54 |
| #57, probability/correction | `research/magicpig-mp2-probability` | #53, #54 |
| #58, sparse attention/BF16 | `research/magicpig-mp2-attention` | #54, #57 |
| #59, cache/merge contract | `research/magicpig-mp3-cache-merge` | #55, #58 |
| #60, CPU composition | `research/magicpig-mp3-cpu-composition` | #55–#59 |
| #61, contracts/typing/CI closeout | `research/magicpig-mp3-contracts-ci` | #53–#60 |
| #62, deferred native GPU checkpoint | created only when resumed | CPU milestones plus explicit user restart |

After #54 defines common interfaces, #55/#56/#57 can implement in isolated
copies in parallel. Root owns the shared driver, ledger loader and overlapping
native-source annotations. Each implementation task owns its test/descriptor
and ledger fragment. Independent reviewers check mathematical expectations,
dependency closure and capacity/lifetime boundaries before PR integration.
Use separate branches and merge commits; inspect the exact reviewed head and
applicable merge-candidate CI, revalidate after base changes, then close issues.

For a restart, read [parent #50](https://github.com/fregata-ariel/HashingDeepLearning/issues/50)
and the existing task PR before creating anything. Each issue records head,
base/merge revision, changed paths, check URLs, passed/skipped/pending boundaries
and next action. Task branch/PR/Issue records survive communication interruptions.

## CPU and postponed GPU policy

#54 establishes a restricted actual-body adapter with independent FP64 oracles.
Its portable suite runs without CUDA, FlashInfer, Torch, downloaded model weights
or large datasets. Native selected-body AVX-512 and BF16 checks are separate and
capability-gated; capability absence is a skip, not numerical evidence. Full
extension linkage and runtime ownership are deferred to later integration.

Archived `install.sh` is historical provenance, not a CPU setup instruction:
it installs CUDA PyTorch and unpinned FlashInfer packages. Source identity does
not establish modern install/build compatibility.

Actual GPU tests are explicitly postponed. G-SLIDE #34/#43 remain open;
MagicPIG #62 is deferred, with no date or recurring GPU job. CPU composition
must not claim production Llama generation, GPU scheduling/device ownership,
paper accuracy or throughput. After #54 integration, #55/#56/#57 can proceed
in parallel using its per-task suite/ledger interface; #58 still depends on #57.

## MP2 selected-CPU checkpoint: SimHash (#55)

#55 supplies five actual archived AST blocks with a stdlib-only CPU tensor
adapter, independent literal/dense expectations and a separate Python suite
interface. Fixed code/sign ordering, centering/append and all K1..11 integer
codes pass. [Coverage](MAGICPIG_SIMHASH_CPU.md) names substitutions and excludes
Torch/BF16/GPU runtime. Packing loss at K12+ is reproduced, with repair #65.
Task #56/#57 mechanism integration proceeds in separate task PRs; #58 continues
to depend on #57. No full production extension or model inference is claimed.

## MP2 selected-CPU checkpoint: retrieval (#56)

#56 verifies sorted-fill retrieval/lifecycle with 2688 independent batched head
comparisons, including at-least-two distinct tables, deduplication, masks and
GQA/batch/layer routing. [Contract](../ports/magicpig/LSH_CPU_CONTRACT.md) marks
Tensor/serial/copy substitutions, owner/view lifetime and alloc-once limits.
Actual native copy is separately capability-gated; unsupported CPUs skip it.
Fastfill parity is reproduced as a defect, not reported as passing; repair #66
owns table population/temporary allocation repair. #57 remains separate and #58
still requires its probability contract. GPU work remains postponed.

## MP2 selected-CPU checkpoint: probability (#57)

#57 adds independent exhaustive/binomial correction expectations (15108 states,
141 corrections,14 domain cases,670 assertions), explicit +1e-4 variant and
finite-estimator bias/domain characterization. [Contract](../ports/magicpig/PROBABILITY_VALIDATION.md)
records conditioning budgets and excluded end-to-end claims. Repairs #67/#68
remain separate. After #55/#56/#57 integration, #58 can proceed using all four
portable suites and respecting known unsupported domains; parent MP2 remains
open until its remaining work completes. GPU work remains postponed.

## MP2 selected-CPU checkpoint: attention/Softmax/BF16 (#58)

#58 adds actual selected sparse/full attention composition and server lifetime,
separate Softmax/base-2 metadata and BF16 quantization/accumulation tests.
[CPU contract](../ports/magicpig/CPU_VALIDATION.md) links the three detailed
contracts. Seven portable suites run by default; AVX512F/FMA and native BF16
remain capability-gated with explicit skips. The conversion model is distinct
from actual BF16 dot-product execution and real FBGEMM linkage.

682 head compositions, 32 corrected logits, Softmax SIMD-boundary fixtures and
four strictly validated portable sanitizer diagnostics document both accepted
domains and existing defects. New repairs #72–#77 cover Softmax tails/empty
sets/native underflow, QK padding, dimension/group checks and varying GQA length
routing. Closing the scoped MP2 verification tasks does not close these repairs
or claim unrestricted correctness, whole-extension execution or GPU validation.
The next dependent task is #59 (cache/normalizer merge); #60/#61 remain separate.


## MP3 cache/merge work in progress (#59)

Task branch: `research/magicpig-mp3-cache-merge`, cut from reviewed
integration `53f816e3577f3e471876dcfce723a168a11b02b6`.
#59 adds a selected-Python-statement CPU host contract and independent
FP64 corrected-union merge oracle. The [cache/merge contract]
(../ports/magicpig/CACHE_MERGE_CPU_CONTRACT.md) records source
provenance, disjoint static/offload/new-token partitions, base-2 LSE
and the **unverified** Torch/FlashInfer/device boundary.

This work does not change `third_party/magicpig` or production arithmetic.
Before #59 closes, record exact PR head/CI/reviewed merge candidate here
and in #59. #60 and #61 stay separate. CPU-only fixtures do not
override known repairs or the postponed GPU milestone.
