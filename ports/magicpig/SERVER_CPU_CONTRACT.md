# Complete maintained CPU server control (#118)

`magicpig_cpu.CPUAttentionServer` is an explicitly maintained, backend-independent
CPU orchestrator derived from the pinned `LSHSparseAttnServer` and `AttnServer`.
It executes complete constructor, allocation, fill/chunk, build/sort, plan,
decode and clear methods. It does **not** execute the original CUDA class with
spies. Its selected native primitives are actual maintained Torch extensions
validated by #117; resident attention and base2 merge are actual CPU Torch math.
Actual FlashInfer paged append/decode/merge and stream semantics remain #62.

The descriptor `tests/traceability/magicpig_server_suite.json` pins the original
complete `attnserver.py`/`utils.py` file hashes. The runner verifies those pins,
then emits original class/method/function AST SHA256 and line/end locators,
plus hashes of maintained source, fixture and descriptor. Original constructor
lines9..104/336..421 map to constructor/config/allocation; fill lines112..178
and428..471 map to maintained `fill`; build lines181..194/474..484 map to
`build_table`; plan/decode/clear map by their original names. Emitted AST
locators are authoritative if prose line ranges drift. The archive is untouched.

The adaptation deliberately:

- accepts explicit tiny config and cloned BF16 projection instead of model
  loading/RNG/device allocation;
- keeps hashes, resident lengths and offload lengths per layer/request so
  interleaved fill/build and varying request lengths cannot alias shared buffers;
- shares baseline/sampled control, selecting K=0 full native offload attention
  or K1..11/L>=2 native threshold-two retrieval and corrected attention;
- plans one finite append per layer per generation step, rejects duplicate
  plan/decode and capacity exhaustion, and commits resident appends after native
  validation;
- removes pinned buffers, CUDA copies/events/streams and large workspaces;
- uses CPU Torch resident softmax/base2 LSE and merge, with explicit source
  adaptation rather than claiming original FlashInfer API compatibility.

`ServerConfig` fixes positive layer/head/KV/dim/batch/capacity/chunk values,
head divisibility and dim multiple32. Baseline offload requires dim128 and
native full-attention group ratios1/4/8. Dense layer IDs must be unique/in range.
Prompt keys/values are finite contiguous CPU BF16 `[sequence,KV,dim]`. Prompts
must leave generation room in the configured dense capacity. Sparse prompts
require nonempty centered offload with every norm positive. Sink/local/offload
are disjoint row ranges `[0,sink)`, `[sequence-local,sequence)` and the middle.
Projection is finite contiguous CPU BF16 `[dim,K*L]`, cloned by the owner.
Projection signs use strict `>0` and exact FP16 packing for K1..11; sign-margin
fixture results are separated from strict-zero tie witnesses.

`fill` copies caller storage. Hash chunks cover only nonempty ranges, including
partial final chunks. `build_table` sorts codes and corresponding row-ID
permutations together; equal keys have no promised stable order. Full ready
state requires every request/layer filled and built. `plan` reserves exactly one
decode for every layer. Queries are finite contiguous CPU BF16
`[batch,Q,1,dim]`; appended keys/values are `[batch,KV,1,dim]`. Zero query norms
reject before append. Returned owned FP32 output is `[batch,1,Q*dim]`.
Cache/average/hash/scratch buffers belong to the orchestrator. No concurrent
calls, in-place caller mutation during calls, or external mutation of its
inspection attributes are supported. Native backends are synchronously called
through annotated narrow protocols; no broad stubs or `Any` APIs are added.

Nonfinite derived centering/norms or hash projection products reject before native fill. Nonfinite query
norms, resident logits/LSE, native nonempty LSE/output or merged output reject
before resident commit, even for finite input tensors whose arithmetic overflows.
Empty native heads retain zero output/negative-infinity metadata.

User shape/dtype/device/config/phase errors reject before resident cache/length
changes. A failed native primitive can update retrieval scratch buffers but
cannot append resident tokens or consume the pending layer. `clear` resets all
owned CPU state and both native owners for refill. Mathematical errors remain
exceptions; no fallback to a generic dense result conceals failure.

Tests include 192 main union checks plus 64 two-KV/GQA4 routing checks,
and four finite-input overflow rejection/precommit regressions. They use
two requests with different prompt/offload lengths, three layers,
GQA4, interleaved fill/build, two appends, and clear/refill for sampled and K0
baseline independently. Expected token partitions/centering/hash payloads and
state transitions are computed independently. A single FP64 softmax enumerates
all supplied resident/offload tokens, using corrected logits only for sampled
rows; the oracle never merges two independently computed outputs with the same
production merge formula. Existing native output .008 and base2 LSE .03 budgets
are retained on identical BF16-quantized operands. Real Torch results are not
FP64/list surrogate execution. Selected-body composition/probability/BF16 oracle
suites remain separate evidence and keep their existing tighter stage budgets.

Five deterministic helpers are selected from `models/utils.py`: causal mask,
GQA repeat, rotate-half, absolute-position RoPE, and RMSNorm. Literal operands
check direction/broadcast, grouping, position/sign and scalar RMS formulas.
RMSNorm is a CPU mathematical implementation, not a FlashInfer kernel call.
Top-p is outside this server API, as are all `LLMLayer`/`LLM` methods, model/weight
loading, request tokenization, position/generation/EOS loops, CLI/dataset,
RULER/accuracy/performance and unused variants. No pretrained generation claim
is made. Alternate distributed/Quest/TopK/Mistral/70B historical closures remain
outside this issue. Unsupported native CPUs record a skip, not server success;
portable pre-existing numerical suites remain required.

The optional local gate `--require-native` fails a skipped
native orchestrator after writing evidence and a bounded escaped Actions notice.
The server requires the BF16-disabled native CPU mode, without requiring the
AVX512 BF16 ISA; #117 separately requires both native build modes. Default
exploratory skip behavior and imported `build_extensions` calls are unchanged.

Required Hosted acceptance uses `--execution-mode simde --require-execution`:
the unchanged 256-head union/overflow/helper fixture invokes actual maintained
C++ server/retrieval bodies through the pinned intrinsic compatibility adapter,
with real Torch and FBGEMM runtime conversion dispatch. Resident attention and
merge remain real CPU Torch math. Its notice is `EMULATED_PASS`, with
`native_simd_verified=false` in evidence. It requires only the BF16-disabled
source build; #117 separately executes both source builds. This does not turn
an unsupported native CPU into a native SIMD pass. `--require-native` rejects
combination with SIMDe mode. See the [extension boundary](EXTENSION_CPU_CONTRACT.md)
for source pin, header/binary audits, adapted converter calls and ISA limits.
