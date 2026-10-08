# G-SLIDE task orchestration and recovery

Integration branch: `research/lsh-lineage-vendor`.
Starting revision: `1e26e1326deedbe05760c700fd6324799fc07079`.

Confirmed milestones: [G1 #6](https://github.com/fregata-ariel/HashingDeepLearning/milestone/6),
[G2 #7](https://github.com/fregata-ariel/HashingDeepLearning/milestone/7),
[G3 #8](https://github.com/fregata-ariel/HashingDeepLearning/milestone/8).
G1 parent #32 is complete; G2 parent #33 tracks the CPU work below; G3 parent
#34 waits for user-announced GPU access.

| Issue | Task branch | Owner / dependencies | Integration PR |
| --- | --- | --- | --- |
| #35 | `research/gslide-g2-pr-ci` | root: PR gates, no dependency | [#41](https://github.com/fregata-ariel/HashingDeepLearning/pull/41), merged |
| #36 | `research/gslide-g2-cpu-suites` | root: common driver/ledger; #35 | [#42](https://github.com/fregata-ariel/HashingDeepLearning/pull/42), merged |
| #37 | `research/gslide-g2-backward` | backward agent; #36 | [#44](https://github.com/fregata-ariel/HashingDeepLearning/pull/44), merged |
| #38 | `research/gslide-g2-softmax` | Softmax agent; #36 | [#45](https://github.com/fregata-ariel/HashingDeepLearning/pull/45), merged |
| #39 | `research/gslide-g2-candidates` | candidate/rebuild agent; #36 | [#46](https://github.com/fregata-ariel/HashingDeepLearning/pull/46), merged |
| #40 | `research/gslide-g2-training` | training agent; #37, #38, #39 | [#48](https://github.com/fregata-ariel/HashingDeepLearning/pull/48), merged |
| #47 | `research/gslide-g2-closeout` | root: combined evidence/docs; #37–#40 | Closeout PR linked from [#47](https://github.com/fregata-ariel/HashingDeepLearning/issues/47) |

Agents use isolated working copies. Shared driver, adapter and ledger loader
belong to root. Each implementation task owns a separate test translation
unit, suite descriptor and experimental ledger fragment. Root attaches the
corresponding native-source contracts during review/publication, avoiding
concurrent source-file edits. An unexpected correctness fix is isolated in
its own follow-up task rather than hidden in an unrelated test change.

Independent #37–#39 work can run in parallel after the suite interface is
defined. #40 can prepare its fixture in parallel, but its merge depends on
those numerical contracts passing. Root reviews independent expected values,
valid capacities, serial-adapter limitations and unchanged archive identities.

Integration happens through task PRs. Before merging, inspect exact head SHA,
changed paths, applicable CI results and dependency completion. Never merge a
stale reviewed head. After a base update, check the new merge candidate or
sync/revalidate when needed. Use merge commits to preserve task lineage.

Four implementation agents worked in isolated copies; two independent review
agents checked dependencies, suite enrollment, CI and numerical contracts.
Reviews caught missing-suite enrollment and a same-basename test substitution;
both now have regression checks. The training fixture was strengthened so an
actual Adam update changes WTA membership and an omitted rebuild is rejected.

## Verified integration checkpoints

| Task PR | Merge revision | Relevant hosted CI |
| --- | --- | --- |
| #41 | `7de1bfac75a2dbc23e26f1b8e0d209cdc58b75ae` | All seven applicable workflows passed before merge |
| #42 | `a8345c34c73a04281a589fdd4a202dd8b4a9a829` | Traceability and strict typing passed |
| #44 | `516fbf186540fc51e0a94982944bf8b399f02ab8` | [37707847531](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37707847531) |
| #45 | `cc309bbc574b6f2924f2f97ac8b92183cedf0d07` | [37708115302](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37708115302), after base synchronization |
| #46 | `5acceaf6fb733f327ff7a31b3143aede019ab05d` | [37708215388](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37708215388), after base synchronization |
| #48 | `72f3a79096b411afe938f02db8b84da78f539cc5` | [37708385020](https://github.com/fregata-ariel/HashingDeepLearning/actions/runs/37708385020), all five suites plain + ASan/UBSan |

The closeout's own head, final CI and merge revision are recorded in #47 and #33
so this document does not require a self-referential commit hash. Task branches
remain available for recovery. Native hidden-delta reset finding #43 belongs to
G3 and does not silently become a production fix in G2.

## Resume protocol

GitHub issue and PR records are the durable checkpoints, not the chat session.
Each published task records:

1. parent/subissue, milestone, branch, PR and current commit;
2. changed paths and exclusive ownership;
3. dependencies and their merged revisions;
4. validation commands, results and CI URLs;
5. known CPU substitutions / GPU pending boundaries;
6. next action (implement, review, await CI, sync, merge or close).

After a communication interruption, read #33's task list and the existing
task branch/PR first. Fetch current refs/checks before resuming. Do not
re-create an existing issue, branch or PR. If implementation has finished,
publish/review its concrete changes before requesting any missing approval.

Routine validation remains CPU-first. Original GPU allocation, scheduling,
warp reductions, Thrust/cuBLAS and production CUDA training are not inferred
from serial oracle success. G3 is the infrequent native-device checkpoint.
