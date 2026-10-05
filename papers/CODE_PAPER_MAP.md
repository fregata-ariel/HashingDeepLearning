# Code ↔ paper map

This file is the source-of-truth for the future Doxygen/docstring annotation
pass. The source trees under `third_party/` remain archival snapshots and should
not be edited merely to add commentary. If annotated/maintained variants are
needed, create them as explicit ports or patch layers.

The comments we eventually add should distinguish:

1. **paper claim** — what the paper explicitly states or measures;
2. **implementation mapping** — which symbol realizes that mechanism;
3. **implementation inference** — behavior inferred from source rather than
   explicitly stated by the paper.

---

## Original SLIDE — MLSys 2020

Paper: *SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for
Large-Scale Deep Learning Systems*  
https://arxiv.org/abs/1903.03129

### LSH table construction and lookup

**Paper anchors**

- §2, *Locality Sensitive Hashing*
- §2.1, *LSH for Estimation and Sampling*
- Algorithm 1, *SLIDE Algorithm*
- Algorithm 2, *Algorithm for LSH Sampling*
- §3.2, *Details of Hash Functions and Hash Tables*

**Code**

- `third_party/slide-original/SLIDE/LSH.cpp`
  - `LSH::LSH`
  - `LSH::hashesToIndex`
  - `LSH::add`
  - `LSH::retrieveRaw`
- `SLIDE/DensifiedWtaHash.cpp`
- `SLIDE/DensifiedMinhash.cpp`
- `SLIDE/WtaHash.cpp`
- `SLIDE/srp.cpp`

**Mapping**

The paper describes an LSH layer as `L` tables, each addressed by a
concatenation of `K` hash values. The implementation converts the per-table
hash values into bucket indexes in `hashesToIndex()`, stores neuron IDs with
`add()`, and retrieves the candidate buckets with `retrieveRaw()`.

§3.2 explicitly lists SimHash, WTA, densified WTA, and MinHash as supported hash
families. The individual hash implementations should cite the more specific
hash-function papers as well as SLIDE when we annotate them.

### Sparse forward pass / active-neuron selection

**Paper anchors**

- Algorithm 1
- Algorithm 2
- §3.1, overall SLIDE workflow / forward pass
- Figure 3, sparse forward pass
- §4.1, *Sampling Overhead*

**Code**

- `SLIDE/Layer.cpp`
  - `Layer::queryActiveNodeandComputeActivations`
  - `Layer::queryActiveNodes`
  - `Layer::computeActivations`
  - `Layer::computeSoftmax`
  - `Layer::addtoHashTable`

**Mapping**

For each layer, the input is hashed, buckets are queried, a sparse active set is
formed, and activations are computed only for that set. Softmax normalization is
also over the sampled active output set rather than every output neuron.

The paper discusses three candidate-selection policies in §4.1: vanilla
sampling, Top-K sampling, and hard thresholding. Source comments should identify
the selected implementation policy without implying that every helper
implements all three.

### Sparse backpropagation and asynchronous update

**Paper anchors**

- §3.1, *Sparse Backpropagation or Gradient Update*
- §1.1, contribution describing sparse asynchronous SGD
- §5.3 for scaling observations

**Code**

- `SLIDE/Network.cpp`
  - `Network::ProcessInput`
- `SLIDE/Node.cpp`
  - first-layer and hidden-layer backpropagation helpers

**Mapping**

Only active neurons participate in backward propagation. The paper connects the
sparse/random overlap pattern to HOGWILD-style asynchronous updates. Comments
should describe the code's actual OpenMP/update behavior and separately cite the
paper's convergence/scaling argument; do not claim lock-free safety merely from
the presence of OpenMP.

### Hash-table update overhead

**Paper anchors**

- §4.2, *Updating Overhead*

**Code**

- `SLIDE/Network.cpp`
  - `tmpRehash` / `tmpRebuild`
  - calls to `_hashTables->clear()`
  - calls to `Layer::updateTable()`
  - reinsertion after parameter updates
- `SLIDE/Layer.cpp`
  - `Layer::updateTable`

**Mapping**

The paper argues that recomputing/rebuilding hash state after every gradient
step is too expensive and proposes reducing the update frequency as training
progresses. Source annotation should state the concrete scheduling behavior in
this snapshot rather than assuming it exactly matches every formula/heuristic in
the paper.

---

## Optimized SLIDE — MLSys 2021

Paper: *Accelerating SLIDE Deep Learning on Modern CPUs: Vectorization,
Quantizations, Memory Optimizations, and More*  
https://arxiv.org/abs/2103.10891

### Baseline SLIDE workflow

**Paper anchors**

- §2, *Background: Sub-Linear Deep Learning Engine (SLIDE)*

**Code**

- `third_party/slide-optimized-avx512/SLIDE/Network.cpp`
- `SLIDE/Layer.cpp`
- `SLIDE/LSH.cpp`

This section is the conceptual bridge to the original implementation: LSH
selects active neurons, followed by sparse forward/backward work and hash-table
maintenance.

### AVX-512 vectorization

**Paper anchors**

- §4.2, *Vectorization with AVX-512*
- §4.3, *AVX-512 in SLIDE*
- §4.3.1, *Parameter Updates with ADAM*
- §4.3.2, *Vectorizing Sparse-Dense and Dense-Sparse Operations in SLIDE*
- §5.5, *Impact of AVX-512*

**Code**

- `SLIDE/Network.cpp`
  - AVX-512 branch in `Network<T,Tp>::ProcessInputOpt`
  - `vecAdamWeights`
  - `vecAdamBias`
- `SLIDE/Layer.cpp`
  - AVX-512 activation, gradient, and sparse/dense kernels
- `SLIDE/DensifiedWtaHash.cpp`
  - AVX-512 gather / compare / scatter hashing path
- `CMakeLists.txt`
  - `OPT_AVX512`

**Paper-reported effect**

§5.5 reports that enabling AVX-512 reduces average training time per epoch by up
to about 1.2× relative to the same optimized configuration with AVX-512
disabled, while performing the same computation.

**Current runtime finding**

Our one-batch smoke test under Intel C++ Classic 2021.10 successfully compiles
and executes the AVX-512 path, but the strengthened finite-value check currently
detects NaN parameters after the update. Track this independently from the paper
claim in Issue #5; do not annotate the source as numerically validated until the
runtime discrepancy is resolved.

### BF16 representation and arithmetic

**Paper anchors**

- §4.4, *BF16 Optimization*
- §5.6, *Impact of BF16*

**Code**

- `SLIDE/Bfloat16.h`
- `SLIDE/Layer.cpp`
  - `OPT_AVX512_BF16` paths
  - BF16 dot-product / conversion intrinsics
- `SLIDE/Network.cpp`
  - BF16 parameter handling
- `SLIDE/main.cpp`
  - `Bfloat16Opt` mode dispatch
- `CMakeLists.txt`
  - `OPT_AVX512_BF16`

**Mapping**

§4.4 describes two reduced-precision strategies:

- BF16 activations with FP32 master parameters;
- BF16 activations and BF16 parameters.

The implementation exposes these as `Bfloat16Opt=1` and `Bfloat16Opt=2`,
respectively. Comments should keep the compile-time availability
(`OPT_AVX512_BF16`) separate from the runtime precision mode
(`Bfloat16Opt`).

### Densified-WTA vectorization

**Paper anchors**

- §4.3, AVX-512 in SLIDE; the paper discusses vectorizing the hashing operation
  by precomputing the random map and using vector max-style processing.

**Code**

- `SLIDE/DensifiedWtaHash.cpp::getHashEasy`

**Mapping**

The optimized source loads precomputed index/position vectors and uses
`_mm512_i32gather_ps`, masked comparisons, and masked scatter operations to
update WTA bins in batches of 16 lanes.

---

## MONGOOSE — ICLR 2021

Paper: *MONGOOSE: A Learnable LSH Framework for Efficient Neural Network
Training*  
https://openreview.net/forum?id=wWK7yXkULyh

### Slow-change observation

**Paper anchor**

- §3.1

The paper measures weight movement and hash-code movement during training and
uses their slow-change relationship as the basis for MONGOOSE's scheduler.

### Adaptive LSH-update scheduler

**Paper anchor**

- §3.2, efficient LSH update scheduling

**Code**

- `third_party/mongoose/mongoose_reformer/reformer_lib/scheduler.py`
  - `Scheduler.__init__`
  - `Scheduler.detect_change`
- `mongoose_reformer/reformer_lib/reformer_pytorch.py`
  - scheduler gate around `calc_triplet`
- `mongoose_slide/slide_lib/network.py`
  - `LSHSampledLayer.rebuild`

**Mapping**

The scheduler keeps a compact hash-based view of parameters and compares it with
the updated parameter state. A sufficiently large change triggers an expensive
LSH/hash-function refresh. Comments must distinguish this implementation's
specific hash-code threshold from the paper's general theoretical scheduler.

### Learnable LSH functions

**Paper anchor**

- §3.3, learning parameterized LSH hash functions

**Code**

- `mongoose_slide/slide_lib/triplet_network.py`
  - `TripletNet`
- `mongoose_slide/slide_lib/simHash.py`
  - `SimHash.generate_from_weight`
- `mongoose_reformer/reformer_lib/reformer_pytorch.py`
  - `TripletLSHAttention`
  - `triplet_forward`
  - collection of positive/negative examples during attention

**Mapping**

The paper proposes tuning parameterized LSH (such as SimHash) using training
signals gathered with little additional overhead. In the Reformer
implementation, attention produces positive/negative examples and trains the
rotation/hash parameters with a triplet-style objective. The SLIDE-side
`TripletNet` provides another learnable-hash implementation.

The current CI verifies a genuine `TripletNet` forward/backward/optimizer step
and checks that the learned hash weights change.

### End-to-end integration / evaluation

**Paper anchor**

- §4, MONGOOSE applications to SLIDE and Reformer

**Code**

- `mongoose_reformer/train_reformer.py`
- `mongoose_reformer/reformer_lib/reformer_pytorch.py`
- `mongoose_slide/slide_lib/network.py`
- `lsh_lib/`

The archived Reformer training script is tied to its historical CUDA/APEX
environment. This portability limitation is tracked as Issue #4 and should be
documented as an implementation/environment constraint rather than a limitation
claimed by the paper.

---

## Annotation style to use in maintained/annotated code

### C / C++

Use Doxygen immediately above the narrowest symbol that implements a paper
mechanism:

```cpp
/**
 * @brief ...
 *
 * Paper mapping:
 * - <paper>, Sec. X.Y, Fig./Alg./Table Z.
 *
 * Implementation:
 * ...
 *
 * Reported effect:
 * ...
 *
 * @note <implementation inference, if any>
 */
```

### Python

Use concise docstrings:

```python
def symbol(...):
    """...

    Paper mapping:
        <paper>, Sec. X.Y.

    Implementation:
        ...

    Reported effect:
        ...

    Notes:
        ...
    """
```

Do not copy paper paragraphs into source. Prefer section identifiers and short
paraphrases, and place benchmark numbers only where the code has a clear causal
relationship to the measured feature.
