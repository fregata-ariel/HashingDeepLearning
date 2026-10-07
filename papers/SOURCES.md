# Papers, repositories, and source provenance

## 1. Original SLIDE — MLSys 2020

**Paper:** *SLIDE: In Defense of Smart Algorithms over Hardware Acceleration
for Large-Scale Deep Learning Systems*

Authors: Beidi Chen, Tharun Medini, James Farwell, Sameh Gobriel, Charlie Tai,
Anshumali Shrivastava.

- Paper: https://arxiv.org/abs/1903.03129
- Vendored proceedings PDF: `papers/pdf/slide-2020-mlsys.pdf`
- Upstream code: https://github.com/keroro824/HashingDeepLearning
- Snapshot: `c9283490ffe34ba97005ec6e11800fdcdf165d79`
- Local path: `third_party/slide-original/`
- License: MIT

This is the latest commit on the upstream default branch. The later commit in
`fregata-ariel/HashingDeepLearning` added paper documentation and is therefore
not treated as a newer upstream SLIDE implementation.

SLIDE uses LSH to select a small likely-active neuron set for an input, enabling
sparse forward/backward computation rather than evaluating every neuron.

## 2. Optimized SLIDE for modern CPUs — MLSys 2021

**Paper:** *Accelerating SLIDE Deep Learning on Modern CPUs: Vectorization,
Quantizations, Memory Optimizations, and More*

Authors: Shabnam Daghaghi, Nicholas Meisburger, Mengnan Zhao, Yong Wu, Sameh
Gobriel, Charlie Tai, Anshumali Shrivastava.

- Paper: https://arxiv.org/abs/2103.10891
- Proceedings: https://proceedings.mlsys.org/paper_files/paper/2021/hash/de4086ad4276d895be8ef25ec03c964b-Abstract.html
- Vendored proceedings PDF: `papers/pdf/slide-optimized-2021-mlsys.pdf`
- Historical canonical repo: https://github.com/IntelLabs/SLIDE_opt_ia
  (currently unavailable)
- Coauthor Yong Wu's surviving mirror/history:
  https://github.com/uyongw/SLIDE_opt_ia
- Yong Wu mirror lineage anchor:
  `85f758c631fd2e0e9f2d33f5fefb897370e4e911`
- RUSH-LAB aggregator: https://github.com/RUSH-LAB/SLIDE
- Exact optimized-SLIDE pin recorded by RUSH-LAB:
  `29e40b45d89d62d50bc4a86df5b804b0594ce514`
- Recovery source for that exact descendant tree:
  https://github.com/kurnianggoro/SLIDE_opt_ia
- Local path: `third_party/slide-optimized-avx512/`
- License: MIT

### Why two surviving repositories are mentioned

Yong Wu is a coauthor of the optimized-SLIDE paper, and his repository preserves
the Intel-authored AVX-512/BF16 development history. GitHub ancestry comparison
shows that the RUSH-LAB-pinned `29e40b45...` is **6 commits ahead and 0 behind**
Yong Wu's `85f758c6...`, with `85f758c6...` as the merge base. The exact
later RUSH-LAB pin is not present in the Yong Wu mirror, so a surviving fork is
used only to recover that exact descendant Git tree. An arbitrary fork HEAD is
not substituted.

Verified implementation landmarks include:

- `CMakeLists.txt`: `OPT_AVX512`, `OPT_AVX512_BF16`
- `SLIDE/DensifiedWtaHash.cpp`: AVX-512 `_mm512_*`
  gather/compare/scatter path
- `SLIDE/Bfloat16.h`: BF16 representation and conversion support

## 3. MONGOOSE — ICLR 2021

**Paper:** *MONGOOSE: A Learnable LSH Framework for Efficient Neural Network
Training*

Authors: Beidi Chen, Zichang Liu, Binghui Peng, Zhaozhuo Xu, Jonathan Lingjie
Li, Tri Dao, Zhao Song, Anshumali Shrivastava, Christopher Ré.

- Paper: https://openreview.net/forum?id=wWK7yXkULyh
- Local PDF: not vendored yet; OpenReview currently rejects non-interactive download from GitHub-hosted Actions runners. Do not substitute slides or reconstructed text for the paper PDF.
- Upstream code: https://github.com/HazyResearch/mongoose
- Snapshot: `890043b39b59a93b8e91a30bc79f4b8125febb78`
- Local path: `third_party/mongoose/`
- License: MIT

Beidi Chen and Anshumali Shrivastava are authors of both original SLIDE and
MONGOOSE. MONGOOSE is better described as a **learnable-LSH training framework**
than as a separate "learning format": it learns a data-dependent hash function
and uses a low-cost change-detection scheduler to avoid rebuilding LSH
structures unnecessarily as model parameters move.

Verified implementation landmarks include:

- `mongoose_reformer/reformer_lib/scheduler.py`
- `mongoose_slide/slide_lib/triplet_network.py`
- `mongoose_slide/slide_lib/simHash.py`
- `mongoose_reformer/reformer_lib/reformer_pytorch.py`

## 4. G-SLIDE — TPDS 2022

**Paper:** *G-SLIDE: A GPU-Based Sub-Linear Deep Learning Engine via LSH Sparsification*

- Publisher record: https://ieeexplore.ieee.org/document/9635657
- DOI: `10.1109/TPDS.2021.3132493`; TPDS 33(11), 3015–3027 (2022).
- Author-hosted manuscript: https://panzaifeng.github.io/assets/pdf/tpds22gslide.pdf
  (the version used for mechanism review; not substituted for a publisher PDF).
- Local PDF: not vendored in this milestone; URL-only provenance, no local hash.
- Repository: https://github.com/PanZaifeng/G-SLIDE
- Snapshot: `d93c2f6d0bbf1dd7b96d9c2340ed629c29f4f902`.
- Tree: `7271376329b09ce2b86e40f3b190b387d8abe376`.
- Archive: `third_party/g-slide/`; MIT license; 27 exact upstream Git blobs.
- Pin manifest: `papers/g-slide-source-pin.json`; maintained copy: `ports/g-slide/`.

CPU tests execute ten selected CUDA definitions via a restricted serial
adapter, without CUDA/Thrust/cuBLAS. They establish selected arithmetic/state
contracts only. Full CUDA build and GPU behavior remain pending. WTA's natural
log packing is recorded as a variant; the maintained Softmax maximum identity
fix has an independent negative-logit regression. See
`ports/g-slide/CPU_VALIDATION.md` for evidence and remaining milestones.

## Annotation policy for the next pass

For each relevant function/class:

1. cite the paper plus the smallest useful section/figure/table;
2. explain the implementation role in our own words;
3. use Doxygen for C/C++ and docstrings for Python;
4. document measured effects separately from the mechanism;
5. explicitly distinguish paper claims from implementation inference.
