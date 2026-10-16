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
Its main text and references are available, but the Appendices C/D cited for
the normalization algorithm and proof are absent. arXiv `2110.15343v1` remains
a distinct, uncompared version. Acquisition details and the blocked direct
publisher download are recorded in
[paper provenance](../papers/scatterbrain-paper-provenance.json).

This intake establishes source identity and a reviewed main-text/source map.
It does not establish runtime correctness, complete theoretical hypotheses,
GPU execution, or the paper's trained-model accuracy/performance. No numerical
Scatterbrain experiment is registered in the executable ledger at this stage.

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
proof is referred to missing Appendix D. The displayed main-text statement
does not spell out independence of feature and support randomness. Unbiased
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
