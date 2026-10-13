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
The formal milestones are MP1 #9, MP2 #10, MP3 #11 and deferred MP4 #12;
MP5 #13 tracks CPU repairs #65–#68/#72–#77. MP1/MP2 are closed. Parent #52
remains open while #61 closeout and its applicable gates are pending.
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

The original #58 checkpoint recorded682 head compositions,32 corrected logits
and four strictly validated portable sanitizer defect probes. These counts
describe that historical implementation. Maintained #72–#77 regressions replace
repaired tail/empty/dimension/group failures with ordinary success checks;
source identity remains pinned. #59 and #60 are integrated and closed. #61
closes the CPU contract/typing/CI work after its repair dependencies and exact
combined candidate gates; native/GPU boundaries remain separately recorded.


## MP3 cache/merge integrated checkpoint (#59)

Task branch: `research/magicpig-mp3-cache-merge`, cut from reviewed
integration `53f816e3577f3e471876dcfce723a168a11b02b6`.
#59 adds a selected-Python-statement CPU host contract and independent
FP64 corrected-union merge oracle. The [cache/merge contract]
(../ports/magicpig/CACHE_MERGE_CPU_CONTRACT.md) records source
provenance, disjoint static/offload/new-token partitions, base-2 LSE
and the **unverified** Torch/FlashInfer/device boundary.

This work does not change `third_party/magicpig` or production arithmetic.
PR #79's reviewed head is `78a655bafd4dd879e85e4307cf70fbb41424feeb`;
integration merge is `b1a006d2ccaf15eaf8e1e751127101f74317114f`.
The PR/Issue retain applicable Hosted CI records. #60 and #61 stay separate.
CPU-only fixtures do not override known repairs or the postponed GPU milestone.

PR [#79](https://github.com/fregata-ariel/HashingDeepLearning/pull/79) review
follow-up corrects the CPU fixture's formerly shape-free cache copies and
token-ID-parity probability assignment. The pinned dense caller supplies padded
`max_length` buffers; this is a fixture correction, not a demonstrated native
dense-fill defect. Both 0.4/0.65 sampled probabilities, fixed-shape copies,
host capacity negatives and clear/refill state are checked. Merge algebra
continues to use the independent union-attention oracle. The final reviewed
head must be used for recovery; the prior successful run for `f816a50` does
not validate these follow-up edits.

## MP3 contracts/typing preparation (#61)

The [contract and CI index](../ports/magicpig/CONTRACTS_AND_CI.md) links each
core mechanism to stable test IDs and its dtype/shape/layout, mutation,
ownership and capacity restrictions. Native Doxygen/ledger gates already
cover the selected C++ implementation. Python orchestration remains immutable
archival source with selected-AST host support tests; `ports/magicpig/`
deliberately has zero maintained Python sources/stubs at this checkpoint.

The exact maintained Python/stub inventory now includes a named zero-file
MagicPIG group. The annotation audit rejects undeclared files or replacements;
every enrolled nonempty group uses identical strict mypy checks; MagicPIG
additionally forbids explicit Any. Future maintained MagicPIG sources need deliberate enrollment,
typed/docstring contracts and only narrow stubs for imported runtime APIs.
Existing workflows cover `.py`/`.pyi`, inventory, generator/reporter and contract
changes. Pinned Python3.12/CPU-only Torch typing, exact checkout SHA/tree notices,
native execution/skip/failure notices and bounded sanitized OpenMP attention replace
the useful temporary diagnostic boundaries without adding a new workflow.

## MP3 recovery checkpoint

| Task | Branch / PR | Exact reviewed head / integration | Remaining dependency or action |
| --- | --- | --- | --- |
| #53–#58 | Existing issue/PR records | Selected mechanism checkpoints already in the #59 base `53f816e3577f3e471876dcfce723a168a11b02b6` | Scoped tasks and MP1/MP2 milestones closed. MP5 repair records remain separate. |
| #59 | `research/magicpig-mp3-cache-merge` / [PR #79](https://github.com/fregata-ariel/HashingDeepLearning/pull/79) | Head `78a655bafd4dd879e85e4307cf70fbb41424feeb`; merge `b1a006d2ccaf15eaf8e1e751127101f74317114f` | Closed; integration CI [38417441911](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38417441911) passed. |
| #60 | `research/magicpig-mp3-cpu-composition` / [PR #80](https://github.com/fregata-ariel/HashingDeepLearning/pull/80) | Head `ab5d19f3af56b500c6999d534e4b037a615bce6b`; candidate `e2d438475226fc3c815d5ea708ba557a336870f0`; merge `8ec0115d3e61aec23f812f48966ff7d34286aea8`, tree `216572b8bebbd1e2048636f44013daf594c07243` | Closed; PR CI [38424598732](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38424598732) and integration CI [38425592161](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38425592161) passed for the identical candidate/integration tree. |
| #61 | `research/magicpig-mp3-contracts-ci` / pending PR record | Initial formal base `8ec0115d3e61aec23f812f48966ff7d34286aea8`; final base/head/merge/CI URLs pending | Rebase after required CPU repair merges, independently review contracts/typing, pass applicable exact-head/merge-candidate gates, and record evidence before closure. |
| #65 | `fix/magicpig-hash-packing-domain` / [PR #82](https://github.com/fregata-ariel/HashingDeepLearning/pull/82) | Head `4a6994ac2673459f8803dd7ade5dccd2ca3071d4`; merge `c8710a16ebf1af5bbd8d8439ed9f9a60857538a6` | Closed; PR CI [38427390230](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38427390230); final #61 rebase uses the publisher record. |
| #66 | `fix/magicpig-fastfill-population` / [PR #81](https://github.com/fregata-ariel/HashingDeepLearning/pull/81) | Head `67c4180acc0a629aef17a77017009b0de539068e`; merge `e86c7f338e2f6f806224e0bad26b65d3e70a1e2b` | Closed; PR CI [38426808802](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38426808802); final #61 rebase uses the publisher record. |
| #76 | `fix/magicpig-attention-dimension-contract` / [PR #83](https://github.com/fregata-ariel/HashingDeepLearning/pull/83) | Head `953b6b411d683162d289d4f11f6c78a87cd06f33`; merge `f52e636310b66d4883b420435f4c3b3152c8b7a0` | Closed; PR CI [38427708026](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38427708026); final #61 rebase uses the publisher record. |
| #75 | `fix/magicpig-qk-logical-length` / [PR #84](https://github.com/fregata-ariel/HashingDeepLearning/pull/84) | Head `f9fc523d6d6a4957218b3901c82a0228c4216743`; merge `2fbe4011d269358a1731437603d8f140ab37ce81` | Closed; PR CI [38428117893](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38428117893); final #61 rebase uses the publisher record. |
| #77 | `fix/magicpig-gqa-length-routing` / [PR #85](https://github.com/fregata-ariel/HashingDeepLearning/pull/85) | Head `5f9b4739d6f997ce5bddeabdf1ef2712dbdf2875`; merge `624da256778f20026427123c962e497d52774285` | Closed; PR CI [38428509041](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38428509041); final #61 rebase uses the publisher record. |
| #73 | `fix/magicpig-empty-attention-contract` / [PR #87](https://github.com/fregata-ariel/HashingDeepLearning/pull/87) | Head `c71a4bf513b206d11a04dcdc178cfa1a11760933`; merge `c0ebf2d4988b6ad05d5c645b8d1d81de4f5f7406` | Closed; PR CI [38429089001](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38429089001); final #61 rebase uses the publisher record. |
| #72 | `fix/magicpig-softmax-logical-length` / [PR #88](https://github.com/fregata-ariel/HashingDeepLearning/pull/88) | Head `fe31db83e8e7d6745b8b9d4b155c7cad4364aebc`; integration pending | Draft task PR; validation/integration pending; CI record pending; final #61 rebase uses the publisher record. |
| #67/#68 | Reviewed probability/caller repairs; task PR records pending | #68 corner follow-up `2ebae0b3e2749629843aeccdfe8c5fb51f7481d4` is independently reviewed preparation evidence | Consume actual task merge refs and matching CI before #61 final rebase. |
| #74/#86 | Native exponential repair and Hosted sanitizer follow-up | Actual plain2821-case execution observed; sanitized native attention failed at the local reduction store | #86 repair and actual sanitized native rerun are required before native repair acceptance; local compile-only/portable passes cannot substitute. |
| #62 | Deferred native GPU task | Unexecuted; no recurring job/date | Explicit native GPU restart, verified extension/link closure and runtime/device tests required. |

Concrete restart: inspect #60/#61 and their existing task PRs before creating
new work; start from the recorded integration merge, restore the task branches,
and run the commands in `CONTRACTS_AND_CI.md`. Use the existing checkout or
an isolated ordinary clone as appropriate; do not create a worktree unless
the user requests one. Retain CPU evidence JSON, capability skip reasons,
fixture/source hashes and CI links for the exact final revisions. After the
required repair merges, replace pending recovery cells and revalidate #61 on
the actual integration base. Close remaining parents only after their combined applicable gates pass;
deferred GPU work and open repair issues must remain explicit.

## Reviewed combined preparation evidence

Candidate `f854678a2c1480e5dda9ebbae4b3afbc7a02e6eb` passed twelve local
portable suites plain and ASan/UBSan/LSan, source/capability/driver/boundary
contracts, strict maintained typing,69 ledger records and56 native doc audits.
It includes prospective #74 code, so it is preparation evidence rather than
proof of the current integrated original exponential. Native AVX512/BF16 were
local hardware skips. The temporary diagnostic exists only on that validation
branch; #61 does not import its workflow. Permanent existing traceability CI
retains bounded real OpenMP and API-readable native/checkout notices.

The Hosted Ubuntu24 observation run
[38428254057](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/38428254057)
actually passed17 plain portable/Python/native suites, including the native
2821-case exponential regression. Its sanitized native attention phase failed
with an ASan SEGV at `_mm512_storeu_ps` in the local Softmax reduction (#86);
the sanitized native exponential gate was not reached. That is a failure, not
a hardware skip. Stack-buffer repair candidates and local compilation do not
establish successful Hosted native revalidation. Formal #61 integration must
wait for the applicable native repair result and retire the temporary workflow
without importing it. Permanent phase logs and failure annotations preserve
this distinction even when a final success-evidence file is absent.

GitHub API access is resolved. Existing task PRs and their actual tested
checkout/tree notices are the authoritative publication/integration record;
do not treat a successful event-head status alone as proof of the checked-out
merge candidate. No secret values or transient runtime setup belong in these
docs. MP4/#62 and native Torch/FlashInfer/GPU/model boundaries remain postponed.
