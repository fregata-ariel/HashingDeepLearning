# G-SLIDE task orchestration and recovery

Integration branch: `research/lsh-lineage-vendor`.
Starting revision: `1e26e1326deedbe05760c700fd6324799fc07079`.

Confirmed milestones: [G1 #6](https://github.com/fregata-ariel/HashingDeepLearning/milestone/6),
[G2 #7](https://github.com/fregata-ariel/HashingDeepLearning/milestone/7),
[G3 #8](https://github.com/fregata-ariel/HashingDeepLearning/milestone/8).
G1 parent #32 is complete; G2 parent #33 tracks the CPU work below; G3 parent
#34 waits for user-announced GPU access.

| Issue | Task branch | Owner / dependencies |
| --- | --- | --- |
| #35 | `research/gslide-g2-pr-ci` | root: PR gates, no dependency |
| #36 | `research/gslide-g2-cpu-suites` | root: common driver/ledger; #35 |
| #37 | `research/gslide-g2-backward` | backward agent; #36 |
| #38 | `research/gslide-g2-softmax` | Softmax agent; #36 |
| #39 | `research/gslide-g2-candidates` | candidate/rebuild agent; #36 |
| #40 | `research/gslide-g2-training` | training agent; #37, #38, #39 |

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
