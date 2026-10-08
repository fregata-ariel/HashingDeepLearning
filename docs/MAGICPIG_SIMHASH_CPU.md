# MagicPIG SimHash and key centering CPU contract (#55)

`test_magicpig_simhash_cpu.py` executes selected statements from the pinned
`third_party/magicpig/models/attnserver.py`, class `LSHSparseAttnServer`. It parses
the source without importing its GPU/runtime dependencies. It selects and hashes
five blocks: binary weights, sparse `fill` centering, chunked `fill` hashing,
normalized-query `decode` hashing, and newly appended key centering. The archive
is unchanged. This is an explicitly scoped extracted adapter, not a second
production implementation or a complete server build.

The CPU list tensor supplies only these selected operations. Projection, mean,
norm and dense attention use FP64; `torch.mv` binary16 output is represented by
IEEE half rounding with `struct`. It does not reproduce GPU BF16 matmul,
PyTorch reduction order, GPU projection timing, CUDA, FlashInfer, model loading,
or extension linkage. Inputs have nonzero norms and nonempty offload partitions.

## Independent expectations

The literal two-row, six-column projection has positive, negative and zero dots.
Literal bit-code tables cover strict `> 0` zero ties, least-significant first bit
weights, both `(K,L)=(3,2)` and `(2,3)`, key head/token/table transpose order,
query batch/head/table reshape order and a partial fill chunk. Identical blocks
execute repeatedly with equal results. The key fixture contains distinct
retained sink/local and offloaded keys in two heads. The actual fill block
subtracts each offloaded mean from both partitions; the actual decode assignment
subtracts stored means from two batches of new keys, with different second-batch
means to verify request isolation. The mean cache is explicitly supplied by the
fixture; the production cache-storage assignment is outside this selected block. An independent dense
FP64 softmax and weighted-value oracle verifies common-translation invariance
across retained, offloaded and appended keys (absolute tolerance `1e-12`).

This tests empirical centering in the reference implementation. It does not
claim the augmented MIPS transform of paper Equation 8; Section 5.1 explicitly
reports omitting that transform in the experiments.

## Packing domain and confirmed loss

The source stores powers of two and dot-packed hash sums in binary16 before
converting fill codes to signed int16. Identity projections exhaust all 4094
bit codes across `K=1..11`, and the actual selected fill block returns the exact
integer code. This defines the supported exact tiny-fixture domain. The default
reference `K=10` lies inside it; arbitrary K is not supported by this result.

All-positive projections show a loss starting at `K=12`: intended code 4095
rounds to 4096. Analogously, K13 maps 8191 to 8192 and K14 maps 16383 to 16384.
K15 maps 32767 to 32768, outside positive signed int16. The fixture records that
boundary before any out-of-range cast, whose runtime-specific result is not
claimed. This is binary16 storage precision, independent of projection rounding.

Focused repair: [#65, guard or replace binary16 SimHash code packing for K >=12](https://github.com/fregata-ariel/HashingDeepLearning/issues/65). A maintained implementation should either reject K outside
its exact supported domain or use integer accumulation/storage consistently in
fill/decode and native table allocation. Add native CPU Torch checks when a
maintained tensor runtime exists, include K11/K12 and K14/K15 boundaries, and
keep the archive unchanged. No repair is silently applied in #55.

## Reproduce

```sh
python3 tests/traceability/test_magicpig_simhash_cpu.py --root . --evidence simhash-evidence.json
```

The standard MagicPIG CPU driver enrolls this fixture via a separate Python-suite
descriptor. Evidence records source SHA256, selected statement lines/SHA256,
fixture/descriptor hashes (driver), deterministic counts and explicit absence of
GPU and Torch execution. The fixture requires only Python's standard library.
