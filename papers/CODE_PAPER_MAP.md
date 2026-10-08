# Paper-to-code map

This document is the index for the annotations under `ports/`. The raw
upstream snapshots remain unchanged under `third_party/`.

## Original SLIDE — MLSys 2020

Paper: *SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for
Large-Scale Deep Learning Systems*  
https://arxiv.org/abs/1903.03129

| Code | Paper mapping | What the code implements |
| --- | --- | --- |
| `SLIDE/LSH.cpp::hashesToIndex` | §2 Locality Sensitive Hashing; §2.1 LSH for Estimation and Sampling; Algorithm 2 | Combines K component hashes into one bucket address for each of L tables. |
| `SLIDE/LSH.cpp::retrieveRaw` | §2.1; Algorithm 2, bucket-query loop | Probes one bucket from every LSH table and exposes the candidates used by the sampler. |
| `SLIDE/Layer.cpp::addtoHashTable` | §3.1 Initialization; Figure 2; Algorithm 1 initialization | Hashes each neuron's weight vector and inserts its id into the layer's LSH tables. |
| `SLIDE/Layer.cpp::queryActiveNodeandComputeActivations` | Algorithm 1 sampling/forward steps; §3.1 Sparse Feed-Forward Pass; Figure 3 | Queries the LSH tables, forms an active-neuron set, and computes activations only for that set. |
| `SLIDE/Network.cpp::ProcessInput` | Algorithm 1 backpropagation; §3.1 Sparse Backpropagation / Gradient Update | Runs sparse forward/backward work and accumulates gradients on the selected computation path. With ADAM enabled, the subsequent optimizer loop still traverses every parameter and may move a currently zero-gradient parameter because historical moment state is non-zero. |
| rehash/rebuild logic in `Network.cpp` and `main.cpp` | §4.2 Updating Overhead | Implements periodic hash-table maintenance. The snapshot uses fixed configured intervals; the paper additionally describes an exponentially decaying update frequency, so the code should not be described as an exact implementation of that heuristic. |

| `SLIDE/WtaHash.cpp` | LSH prerequisite used by released experiments; WTA itself predates SLIDE | Selects the winner position inside each permuted bin. The maintained port returns the bin-local position expected by the downstream (K,L) packing rather than a raw feature id. |
| `SLIDE/DensifiedWtaHash.cpp` | LSH prerequisite / released hash-family choice; densified WTA itself predates SLIDE | Maps features into WTA bins, keeps per-bin maxima, and densifies empty bins. `SLIDE2020-WTA-DWTA-PRIMITIVES` covers winner, tie, empty-bin, and output-domain behavior. |
| `SLIDE/DensifiedMinhash.cpp` | Released alternative hash mode, not a standalone SLIDE contribution | Implements the top-k/densified-MinHash experimental option selected by `HashFunction==3`. It should be cited as an implementation alternative, not as an algorithm introduced by SLIDE. |
| `SLIDE/srp.cpp` | Released alternative SimHash/sparse-random-projection mode, not a standalone SLIDE contribution | Implements the `HashFunction==4` projection/sign hash path used as an alternative LSH family. |

The paper's main system-level claim is that LSH-selected adaptive sparsity
avoids computing most neuron activations and enables sparse asynchronous
updates. Performance numbers belong to the complete system and should not be
attributed to a single function.

## Optimized SLIDE — MLSys 2021

Paper: *Accelerating SLIDE Deep Learning on Modern CPUs: Vectorization,
Quantizations, Memory Optimizations, and More*  
https://arxiv.org/abs/2103.10891

| Code | Paper mapping | What the code implements |
| --- | --- | --- |
| `SLIDE/DataLayerOpt.cpp::loadData` | §4.1 Memory Coalescing and cache utilization; “Removing Data Memory Fragmentation” | Packs sparse indices/values into long contiguous vectors and keeps per-record offsets/lengths. |
| contiguous layer buffers in `Layer.cpp` | §4.1, “Removing Parameter Memory Fragmentation” | Stores weights and related state in contiguous layer-wide allocations for cache/coalescing behavior. |
| `SLIDE/DensifiedWtaHash.cpp::getHashEasy` | §4.3.3 Vectorizing Densified-Winner-Takes-All | Uses the precomputed index map plus AVX-512 gather/compare/scatter operations when the vectorized conditions hold. |
| `Layer.cpp::queryActiveNodeandComputeActivationsOpt` | §4.3.2 Vectorizing Sparse-Dense and Dense-Sparse Operations | Contains the vectorized forward/activation path and its layout-sensitive loops. |
| `Network.cpp::ProcessInputOpt` | §4.3.1 Parameter Updates with ADAM; §4.3.2 | Runs optimized sparse backpropagation and the scalar/AVX parameter-update paths. |
| `SLIDE/Bfloat16.h` and BF16 template instantiations | §4.4 BF16 Optimization | Supplies BF16 storage/conversion helpers for activation-only and activation+weight modes. |

Reported effects must remain separate from mechanism:

- §5.5 / Table 4 reports AVX-512 reducing average training time by up to about
  1.2x in the paper's tested configurations, with unchanged accuracy because
  the computation is intended to be equivalent.
- §5.6 / Table 3 reports dataset-dependent BF16 effects; BF16 is not uniformly
  faster on every workload.
- §5.7 says the new implementation is 2–7x faster than the older SLIDE
  implementation overall, with AVX+BF16 accounting for roughly 1.7x and memory
  optimizations providing the remaining improvement.

These paper results are **not** a claim that the historical snapshot is
numerically correct on arbitrary modern machines. Our smoke tests found
separate source/runtime issues tracked in Issues #1–#7.

## MONGOOSE — ICLR 2021

Paper: *MONGOOSE: A Learnable LSH Framework for Efficient Neural Network
Training*  
https://openreview.net/forum?id=wWK7yXkULyh

| Code | Paper mapping | What the code implements |
| --- | --- | --- |
| `mongoose_reformer/reformer_lib/scheduler.py::Scheduler` | §3.1 Slow Change; §3.2 Smart Scheduler | Keeps compact SimHash codes of current parameters and uses code changes as a cheap trigger for expensive LSH-related work. This repository implementation is a simplified practical trigger, not a literal transcription of Algorithm 1's full maintenance data structure. |
| `mongoose_reformer/reformer_lib/reformer_pytorch.py::LSHSelfAttention.forward` | §3.2 scheduler + §3.3 learnable LSH | Calls the scheduler before deciding whether to collect triplet examples and update learnable rotations. |
| `TripletLSHAttention.triplet_forward` | §3.3.1 Learnable LSH, Equation 3 / Algorithm 2 | Optimizes parameterized hash rotations using positive and negative examples and a margin-based cosine triplet objective. |
| `mongoose_slide/slide_lib/triplet_network.py::TripletNet.forward` | §3.3 learnable LSH; implementation-specific SLIDE-side objective | Learns a hash projection from pair labels. This code uses a differentiable pairwise/BCE objective rather than being a literal copy of Equation 3. |

Section 3 of the paper explicitly separates the two MONGOOSE ideas: §3.2
schedules LSH updates under the slow-change observation, while §3.3 learns
parameterized hash functions. Section 4 evaluates those ideas on SLIDE and
Reformer. The headline speed/accuracy/memory results are framework-level
measurements and should not be attached to an individual helper function.

## G-SLIDE — TPDS 2022 (CPU-first milestone)

Paper: https://ieeexplore.ieee.org/document/9635657; mechanism review uses the
[author-hosted manuscript](https://panzaifeng.github.io/assets/pdf/tpds22gslide.pdf).
Paths below are relative to `ports/g-slide/`. CPU evidence executes selected
actual bodies with a serial adapter; parallel GPU behavior remains unverified.

| Code | Paper mapping | Tested contract |
| --- | --- | --- |
| `src/lshKnl.cu::init_hash_no_sw_knl`, `get_hash_knl`, `gather_buckets_knl` | §4.3; Fig. 6 | Fixed WTA permutations and candidate lists; natural-log packing is a variant |
| `src/GPUMultiLinkedHashTable.cu::d_block_reduce_cnt` | §4.3, assistant structures | Serial linked counts, thresholds and label activation |
| `src/kernel.cu::relu_fwd_slide_in_knl` | §4.2, §4.4 | CSC input/active outputs and column-major forward arithmetic |
| `src/kernel.cu::softmax_fwd_bp_rowmajor_all_sm_knl` | §4.4, training support | Stable probabilities and label deltas; maintained negative-logit fix |
| `src/kernel.cu::bp_first_layer_knl` | §4.2, §4.4 | First-layer column-major gradients and biases |
| `src/kernel.cu::update_weights_knl` | §4.4, optimizer support | Two Adam state updates, reset and tail guard |
| `src/kernel.cu::bp_knl`, `bp_rowmajor_knl`, `bp_rowmajor_no_sm_knl`, `bp_rowmajor_slide_knl` | §4.2, §4.4, sparse training support | Four deeper backward variants, both layouts, sparse gradients and activation gating; caller state reset remains explicit |
| `src/kernel.cu::softmax_fwd_bp_rowmajor_slide_in_knl`, `softmax_fwd_bp_rowmajor_slide_out_knl` | §4.4, training support | Additional Softmax bodies against dense FP64 expectations; scalar reduction substitutes |
| `src/GPUMultiLinkedHashTable.cu::d_block_reduce_cnt` with candidate fixture | §4.3, candidate selection | Serial threshold/label selection, host active-set conversion, ring capacity and stale/rebuilt memberships |
| `src/lshKnl.cu` selected hash/query bodies with training fixture | §4.3, index maintenance | Actual Adam change alters three WTA addresses; host reconstruction required by the deterministic fixture |
| `src/kernel.cu::update_weights_knl`, `bp_rowmajor_knl` with training fixture | §4.4, composed training support | Two connected updates and independent dense math; explicit host delta reset and stale-state regression (#43) |

Six baseline and eleven fragment `GSLIDE-*-CPU` ledger records specify independent
oracles and execution limits across five suites. These mappings include direct,
support and variant relations; the ledger and native contracts distinguish them.
The [coverage checkpoint](../ports/g-slide/CPU_VALIDATION.md) describes adapter
substitutions and the pending production/GPU baseline. The immutable archive is
unchanged; G2 adds tests/contracts without changing production CUDA arithmetic.

## MagicPIG selected CPU support

Paper baseline: arXiv v4; selected source: v0.2.

| Maintained symbol | Paper location | Verified boundary |
| --- | --- | --- |
| `transform_kernel`, `softmax_kernel` | §4.3 Eq.9–11, §4.4 support | Independent dense FP64 correction, probabilities, host weighted output and base-2 LSE on supplied subsets; portable lanes replace exponential evaluation, native mode checks the actual polynomial |
| `qk_kernel_bf16_impl` | §4.3–4.4, implementation support | Capability-gated actual BF16 QK body, one complete 16-row × 32-coordinate tile |

These two ledger records are support mappings, not complete sampling/attention
or paper benchmark reproduction. QK/WV in the baseline are host substitutes;
Torch/FBGEMM linkage and GPU execution remain pending. Details and tolerances:
[CPU contract](../ports/magicpig/CPU_VALIDATION.md). Remaining mappings and task
dependencies: [integration record](../docs/MAGICPIG_INTEGRATION.md).

### SimHash/centering support (#55)

Selected immutable `LSHSparseAttnServer` packing/fill/decode AST statements map
as support to §4.3 angular hashing and empirical centering. Literal independent
projection/code expectations and integer enumeration cover 4094 codes K1..11.
The FP64 list adapter does not validate native Torch/BF16/GPU execution. Equation
8 augmentation is not claimed. [Contract](../docs/MAGICPIG_SIMHASH_CPU.md);
packing outside the exact domain is tracked in repair #65.

## Annotation conventions

1. C/C++ functions use Doxygen `/** ... */` blocks with `@par Paper mapping`,
   `@par Implementation note`, and, only when appropriate,
   `@par Reported effect`.
2. Python classes/functions use docstrings with the same three concepts.
3. Comments paraphrase papers; they do not paste long passages.
4. If code only approximately realizes a paper algorithm, the annotation says
   so explicitly.
5. Known defects and reproduction findings point to GitHub Issues instead of
   silently rewriting the historical behavior.
