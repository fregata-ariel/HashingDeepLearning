# Scatterbrain integration record

## Baseline and evidence boundaries

Parent: [#94](https://github.com/fregata-ariel/HashingDeepLearning/issues/94).
Intake base: `a7d0dbc74cfd4bf1b368a21fb4599d2ca13516e6` on
`research/lsh-lineage-vendor`. Intake branch:
`research/scatterbrain-sb1-source-intake`.

The primary baseline is the user-supplied **NeurIPS 2021 conference manuscript**:
[local PDF](../papers/pdf/scatterbrain-2021-neurips.pdf), 14 pages, SHA-256
`1503ff9a48b07855bbcb8e0cb3a09ff312dcdd3c8c6e1356ca1b0c67f9e97846`.
The title is *Scatterbrain: Unifying Sparse and Low-rank Attention Approximation*.
The supplied main file has no appendices. The separately published supplement
and distinct arXiv `2110.15343v1` were acquired on 2026-10-17; Appendix C/D.3
are now inspected, with printed causal inconsistencies and implicit moment
conditions recorded in [the appendix review](SCATTERBRAIN_APPENDIX_REVIEW.md).
Acquisition details and the historical blocked direct main-PDF download are in
[paper provenance](../papers/scatterbrain-paper-provenance.json).

Initial intake PR #103 at `916d988` established source identity and a reviewed
main-text/source map, without registering numerical experiments. The subsequent
CPU baseline below adds three bounded executable entries. Full theoretical
hypotheses, GPU execution and paper model accuracy/performance remain separate.

## Source history and archive scope

`HazyResearch/scatterbrain` and `HazyResearch/fly` resolve to the same GitHub
repository ID `421284664`; this is a rename, not a replacement by a fork.

| Role | Coherent source pin | Tree |
| --- | --- | --- |
| Initial release, 2021-10-29 | `eeb689c00b9f322f33f6b6f85eaaac0a1d6a5eab` | `b6a41881fc7d68f86961fab67f25b07c97aad7c6` |
| First SBSmyrf, 2022-04-15 | `74cc6b14347b4edcf5c49098e80a9ef47c0b21bf` | `1821d435ef810ce88038ef81965c65cd0ac8c6bc` |
| Later default HEAD observed 2026-10-17 JST | `6b73449a6b3e228af9e4afe4f153a384e9b537b9` | `8e9f6490e8c7c2acea1e9740b29556dcca440ee7` |

The initial root commit is titled “Code release & T2T-ViT experiments”. There
are no repository tags or GitHub releases. That title and the paper's author
README do not identify the exact commit used for every paper experiment.
The README describes **local attention + Performer** and has a stale
`sblocal.py` link; the actual entry point is `sblocal_attention.py`.
Initial `SmyrfAttention` is a separate sparse method, not the later
`SBSmyrfAttention` residual composition.

The immutable initial archive contains 15 selected upstream files:

- Five SBLocal runtime project files: `feature_maps_sb.py`,
  `projection_utils.py`, `sblocal_attention.py`, `scatterbrain_utils.py`,
  and `models/modules/masking.py`.
- The test's `full_attention.py` project dependency, three package markers,
  one SBLocal configuration, and the upstream SBLocal test.
- Upstream LICENSE, README, Dockerfile and setup.cfg as historical context.

The complete initial tree contains 170 blobs. Full-tree inventory and raw
commit identity preserve visibility of all exclusions. Archive equality is
checked against the selected Git blobs, without editing or annotating them.
External native fast-transformers local-product bindings are **not** included
in this source closure. The archived test additionally depends on external
masking classes and uses historical CUDA/large-feature configurations; merely
preserving it is not a supported CPU test run. The Dockerfile is a historical
GPU/application environment with unpinned dependencies, not a setup script.

Later SBSmyrf has an eleven-file project import closure using ordinary Torch
operations on its noncausal path. All eleven blobs at its first-introduction
pin are identical at the separately recorded later HEAD. Future CPU execution
must fetch one coherent pinned tree, not assemble unrelated helper revisions.
Later source metadata is not an archive or an execution result.

## Attribution and redistribution boundaries

The author's root license is Apache-2.0. The selected initial helpers credit
Idiap/fast-transformers; its historical MIT notice is preserved separately
without changing upstream archive bytes. The matching historical Idiap pin is
`f22c13716fc748bb21a7b226ada7f7b5f87f867f`. The selected
`scatterbrain_utils.py` also retains the causal helper scaffold from
lucidrains/performer-pytorch. Its historical Phil Wang MIT notice is preserved
even though execution of that causal branch is deferred. Import closure alone
does not identify all copied-source attribution obligations.

Other upstream files explicitly cite `giannisdaras/smyrf`. Its pre-Scatterbrain
snapshot `bbbc82e0126233da07841275daa238d447c37f66` carries GPL-3.0, and a
historical source comparison confirms copied hash-helper text. This documents
an attribution ambiguity; it does not determine the license of every Fly
file. Initial hash/SMYRF and later SBSmyrf-derived files are not redistributed
in this selected archive or silently described as Apache-only. Pinned ephemeral
source retrieval can support CPU research while a maintained redistribution
strategy is investigated in #99. Any later imported dependencies need their
own applicable notices, including additional lucidrains helpers where used.

## Reviewed main-text correspondence

Paths below refer to initial `src/models/modules/attention/`, unless marked
as later. These are source interpretations, not passed numerical ledger IDs.

| Main-text locator | Source | What is established |
| --- | --- | --- |
| p.5 footnote 3; p.6 §4.2 | `feature_maps_sb.py`, `projection_utils.py` | Positive random features target the unnormalized `exp(QKᵀ)` kernel; actual orthogonal/cosh/scaling choices need distribution-specific checks |
| p.6 §4.2 Eq.(1) | `SBLocalAttention.forward` | Sparse term subtracts the low-rank estimate on selected local entries |
| p.6 §4.2 Eq.(2) | `linear_attention`, `linear_attention_normalization`, `SBLocalAttention.forward` | Efficient factor association plus sparse residual; Eq.(2) itself is unnormalized output |
| p.6 §4.2 alternatives paragraph | Initial local support; later `SBSmyrfAttention` | Local, random block and Reformer support are explicitly discussed; later rank-cluster support must be characterized on its own |
| p.7 §4.3 Theorem 2, Eq.(3) | Future statistical feature/support validation | Per-entry exponential-kernel statement for nearby unit-sphere inputs, not a normalized-output unbiasedness result |

Write `A = exp(QKᵀ × temperature)`, `P = Φ(Q)Φ(K)ᵀ`, and let `M` denote the
logical selected support. The inspected residual identity is

`H = P + M ⊙ (A − P)`.

Thus selected **kernel entries** equal A. Their normalized probabilities need
not be exact, because the row denominator still includes approximate entries.
The independent CPU target uses the same H in `HV` and `H1`, then computes
`HV / H1`; separately normalizing and adding the two paths is different.
Dropout, feature epsilon, finite arithmetic, masks and denominator floors are
explicit implementation variants to characterize.

Theorem 2 states a per-entry variance reduction factor `1 − p` for nearby
unit-sphere q/k, with p expressed using dimension and the distance bound. Its
proof in Appendix D.3 is now inspected. Its conditional-to-unconditional moment
substitutions do not explicitly spell out their justification. Unbiased
features and independence of feature/support randomness are sufficient
assumptions for the simple variance identity used in our independent algebra;
suitable conditional moment assumptions can also suffice. Fixed-feature output
agreement cannot establish those statistical assumptions.
Even an unbiased unnormalized estimator generally yields a biased normalized
ratio. No theoretical LSH collision probability is assigned to the later
rank-sorted XBOX+/SMYRF support without a matching derivation.

The supplied bibliography cites SLIDE and MONGOOSE. This supports a conceptual
relationship among efficient neural computation methods, not direct code
inheritance. Scatterbrain approximates attention interactions; SLIDE's selected
neuron training path is a different integration target.

## Issue and Milestone progression

| Stage | Issue | Acceptance focus |
| --- | --- | --- |
| SB1, milestone 14 | [#95](https://github.com/fregata-ariel/HashingDeepLearning/issues/95) | Supplied paper identity and main-text/source map; identify missing appendix/version evidence |
| SB1 | [#96](https://github.com/fregata-ariel/HashingDeepLearning/issues/96) | Exact initial archive, source/history attribution, full-tree identity gate |
| SB1 | [#102](https://github.com/fregata-ariel/HashingDeepLearning/issues/102) | Obtain absent Appendices C/D and verify complete algorithm/proof assumptions |
| SB2, milestone 15 | [#97](https://github.com/fregata-ariel/HashingDeepLearning/issues/97) | Actual features/projection/redraw state on Torch CPU |
| SB2 | [#98](https://github.com/fregata-ariel/HashingDeepLearning/issues/98) | Initial local residual, common normalizer, masks and gradients; native boundary is explicit |
| SB2 | [#99](https://github.com/fregata-ariel/HashingDeepLearning/issues/99) | Later actual SBSmyrf CPU path, independent support and duplicate correction, attribution strategy |
| SB2 | [#100](https://github.com/fregata-ariel/HashingDeepLearning/issues/100) | Combined contracts, maintained strict typing, CPU CI and execution closeout |
| SB3, milestone 16 | [#101](https://github.com/fregata-ariel/HashingDeepLearning/issues/101) | Deferred GPU/native causal paths, training and accuracy/performance reproduction |

Proceed through dedicated reviewed PRs into the integration branch. Verify
applicable CI on the actual merge candidate and compare the tested tree with
the integration result. Record exact head/base/check URLs and next action in
each Issue. Only close the acceptance actually verified; missing proof/source
or concrete defects become explicit follow-up Issues.

## CPU verification design

Use tiny deterministic synthetic Q/K/V tensors with actual CPU Torch execution.
No model weights, datasets or full historical Docker environment are needed.
Begin with dropout zero and noncausal FP32; FP64 comparison has separately
controlled source dtype behavior.

1. Test actual feature/projection code against direct dense expectations;
   cover scaling, temperature, cosh, odd feature counts, redraw/eval before
   first sample, state serialization and finiteness.
2. Form an independent dense hybrid kernel from literal local support or
   independently interpreted hash permutations. Compare numerator,
   denominator, output and conditional gradients, not output alone.
3. For later multiple rounds, count each selected pair's occurrences and
   verify correction of **both** exact and low-rank residual terms. Derive
   logical support from permutations/clusters: `attn_lsh != 0` can omit a
   selected zero-residual pair and is not a valid support oracle.
4. Cover padding, supplied heterogeneous key masks, batches/heads, cross
   attention, zero/full/partial support, negative residuals, dtype and reuse.
   Floors, all-masked rows, dropout and rejected causal/attention-mask cases
   need explicit contracts before being claimed supported.
5. Add meaningful regressions for reproduced failures and execute mutations
   for missing subtraction, bad duplicate correction and mismatched support.
   Make maintenance changes only under `ports/`, after characterization.

Extend the existing traceability and typing workflows. Add every maintained
Python file to an explicit strict inventory group; a zero-size group or an
archived module is not a typing pass. Record source/fixture/dependency hashes
and actual native versus substitute execution. Python/Torch CPU tests do not
become ASan-instrumented merely by sharing a job with native sanitizer tests.
GPU and large experiment scope remains #101.

## Maintained CPU baseline

`ports/scatterbrain/` retains the initial five-module core and original namespace.
Independent erasure of annotations, private type declarations, identity casts
and docstrings leaves all five numerical ASTs unchanged. Eight Python files and
five narrow native/optional-import stubs are enrolled in strict mypy, including
`--disallow-any-explicit`. Archived source is not included in this typing claim.

The real native dependency has a separate three-blob MIT archive at
`third_party/scatterbrain-fast-transformers/`: LICENSE, CPU C++ source and the
original autograd wrapper. `papers/scatterbrain-native-dependency-pin.json`
records exact identities and the supported profile. An explicitly adapted
GNU14/C++20 setuptools recipe builds a fresh CPU extension, verifies the actual
binary import location and staged source bytes, and records source/fixture/
binary/compiler/runtime/ABI hashes. No native arithmetic patch or Torch
substitute is used. Missing required build/link/import capabilities fail;
stale evidence is removed before execution.

| Ledger ID | Bounded execution and independent checks |
| --- | --- |
| `SCATTERBRAIN-FEATURES-CPU` | 72 feature, 12 restored-kernel, 24 input-gradient, 24 projection-gradient and 16 projection cases; scalar FP64 expressions/analytic derivatives, structural projection checks, three actual-function mutants |
| `SCATTERBRAIN-LOCAL-PRODUCT-CPU` | Five actual native product contexts and original custom backwards; independent dense/indexed expectations, lengths and an additive-mask primitive case |
| `SCATTERBRAIN-LOCAL-HYBRID-CPU` | Ten maintained forward/backward fixtures, two full-local dense equalities; independent detached FP64 oracle and two AST mutants rejected in all ten cases |

Local-hybrid scope is FP32 CPU/noncausal/dropout0/eps0, finite moderate inputs
and positive finite row mass. Raw H1/HV are reconstructed from the returned raw
hybrid matrix; the internal log-normalization variable is not directly traced.
Feature gradients cover eps0/nonlog restored features; log/nonzero-eps modes
have forward checks. Four separately reported lifecycle observations remain
#106, including cold eval/warm-load failures, ignored redraw and odd-count
truncation. They are characterizations, not repaired lifecycle guarantees.
See the [port contract](../ports/scatterbrain/CPU_CONTRACT.md) and
[feature report](../tests/traceability/scatterbrain_feature_cpu_report.md).

Later SBSmyrf remains a **pinned reference characterization**, not this maintained
port. The driver fetches exact `74cc6b1` into an ephemeral directory, checks all
eleven dependency fingerprints, and executes ten whole-module cases plus ten
source mutants. Independent permutation-derived support and raw H/HV/H1 gates
catch duplicate/subtraction errors and common scaling invisible in the ratio;
nonfinite outputs/weights are rejected. Source-failure controls constrain exact
exception categories/diagnostics. Downloaded/mutant source is excluded from
artifacts. Numeric XBOX+/E2LSH correctness itself is not independently validated.
FP64 multi-round raw precision has a separate budget because of
`log(dup_counts.float())`; benign ratio cancellation is not a strict raw-FP64
pass. #104–#106 remain open. No maintained-later-port, theoretical hash-selection
probability or general archive-maintenance pass is claimed.

Python3.12, CPU Torch2.14.1 and the narrow dependencies in
`tools/scatterbrain-cpu-requirements.txt` reproduce these suites:

```sh
python tests/traceability/test_scatterbrain_feature_cpu.py --emit-dir /tmp/sb-features
python tools/check_scatterbrain_initial_cpu.py --test-dependency-contracts
python tools/check_scatterbrain_initial_cpu.py --cxx g++-14 --build-dir /tmp/sb-initial
python tools/check_scatterbrain_later_cpu.py --emit-dir /tmp/sb-later
```

The existing traceability workflow owns CPU execution/artifacts; existing strict
typing owns the maintained files. No new workflow or GPU job is added. Native
memory sanitizers, causal paths, arbitrary masks, initial all-zero-key rows,
extreme/clamped masses, full training and model accuracy/performance remain
unverified by these fixtures.
