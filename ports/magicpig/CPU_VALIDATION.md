# MagicPIG CPU validation contract

The maintained source starts from selected v0.2 commit
`ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`. Numerical tests extract actual
definitions from `ports/`; the independent oracle lives in test fixtures.
The 41-file immutable archive and full upstream inventory are checked first.
This is a restricted selected-body baseline, not a Torch extension build or
production Llama runtime. No model weights, datasets, CUDA or FlashInfer are used.

## Execution modes

| Mode | Executed source | Substitutions / requirements |
| --- | --- | --- |
| Portable (always) | `transform_kernel`, `softmax_kernel` | Software 16-lane FP32 intrinsics and host `std::exp(float)`; host QK/WV; no SIMD ISA flags or vectorization |
| Native AVX-512 (opt-in) | Same bodies plus actual exponential polynomial/constants | Confirmed x86, GCC, AVX512F and FMA CPU/OS support; host QK/WV |
| Native BF16 (opt-in) | `qk_kernel_bf16_impl` | Confirmed AVX512F/BW/BF16 and GCC >=11; `uint16_t` storage alias matches pinned FBGEMM; no FBGEMM conversion/runtime |

The parent compiles a baseline GCC builtin CPU/OS probe without SIMD flags.
When available, reported flags are intersected across `/proc/cpuinfo` processors.
Unknown capabilities, unsupported architectures or compilers produce explicit
native skips before any SIMD executable, including its global initialization.
A skip does not establish native numerical correctness. Compiler failure after
positive eligibility is a failure, not a skip. BF16 is independent of FMA.

```bash
python3 tests/traceability/test_magicpig_capabilities.py
python3 tests/traceability/test_magicpig_cpu_contracts.py
python3 tools/check_magicpig_cpu.py --build-dir /tmp/magicpig-plain
python3 tools/check_magicpig_cpu.py --native --sanitize --build-dir /tmp/magicpig-sanitized
```

## Oracle and bounded coverage

`MAGICPIG-BASELINE-CPU` checks 42 fixed cases and 684 candidates: lengths
1, 2, 15, 16, 17, 31, 32, two K/L configurations and common corrected-logit
shifts 0, -32, -128. The fixture uses 47 rows × 16 coordinates, unsorted IDs,
finite positive norms and guard canaries. FP64 dense dots/norms are computed
before candidate gathering; inclusion is independently summed over binomial
terms for at least two hits. The source's `+1e-4` regularizer is explicit.
Expected normalized probabilities, weighted values and base-2 maximum/LSE are
independent of extracted source and the software lane adapter.

Portable probability/output absolute tolerance is `1e-5`. Native polynomial
probabilities allow `0.005 + 0.02*abs(expected)`; weighted output and LSE allow
`0.03`. Correction tolerance is `1e-5` in both modes. Common-shift-sized float
metadata has a separate rounding allowance. Polynomial tests cover finite,
moderate relative logits, not extreme exponential underflow/overflow.

`MAGICPIG-BF16-SELECTED-CPU` checks one complete 16-row × 32-coordinate tile
against dense FP64 dots of exactly representable BF16 operands, with unsorted
IDs and canaries; absolute tolerance `1e-6`. Arbitrary QK tails/dimensions,
conversion, WV, dispatch and production allocation are not validated.
Empty candidate sets, zero norms and unrestricted inputs await mechanism tasks.

The fixed literal fixtures use no RNG. Routine allocations are limited to tiny
fixture vectors; neither `LSH::alloc` nor `SparseAttentionServer::alloc` is executed. Upstream
LSH allocation scales with layers × batch × KV heads × L × `2^K` for bucket
arrays and with maximum length for tables/masks; attention caches scale with
layers × batch × KV heads × maximum length × head dimension. Integer overflow,
invalid dimensions, repeated allocation and lifetime require later contracts.

## Runtime and ABI decisions

No CPU-only Torch wheel is required for this baseline. Generated translation
units include only selected definitions, standard headers and explicit adapters;
native header hashes are recorded as provenance, not evidence of header/ABI
compatibility. The archived setup forces `_GLIBCXX_USE_CXX11_ABI=0`, OpenMP and
AVX-512F; this driver does not import that ABI choice or link Torch/cpuinfo.
Their exact headers, symbols, ABI and link provision remain unverified.

Upstream `LSH_THREADS` and `ATTENTION_THREADS` default to 64. Selected bodies
here are called serially without OpenMP; server scheduling, races and thread
ownership are unexecuted. Historical `install.sh` is not a CPU setup command.
The maintained Python inventory remains 31 files; no Python model port is added.

## Evidence and CI

The existing traceability workflow runs capability/driver contracts, then plain
and ASan/UBSan selected-body tests. UBSan uses `-fno-sanitize-recover=all`; a
signed-overflow regression proves that diagnostics cannot publish pass evidence.
Hosted numerical runs keep LeakSanitizer enabled. An explicit local-only
`--sanitize --disable-leak-check` workaround is available where sandbox process
inspection is unavailable; evidence records this reduced check. It is not used
by the hosted workflow.

`magicpig_cpu_evidence.json` records compiler identity, CPU/OS capability
decisions and per-mode results. Each mode/suite records source/header, extracted
body, fixture, descriptor, adapter and generated translation hashes, compiler
command and numerical output. GPU and full-extension validation remain false.
Reusing a build directory invalidates old summary/per-mode evidence before
selection or execution, preventing stale success after failure or skips.

## Common interface for #55 / #56 / #57

Each task owns a uniquely named JSON descriptor in
`tests/traceability/magicpig_suites/`, its fixture/optional `.hpp` adapter, and
`papers/traceability/magicpig-<task>.json`. Descriptor keys are `name`,
`test_file`, `definitions` and optional `adapter_file`; name matches filename.
Fixtures stay under `tests/traceability/`. Definitions select unique symbols
from approved maintained LSH/sparse-attention sources. The extractor supports `void`, `int`, `__m512`, `torch::Tensor`, qualified
constructors/destructors and static-inline helpers. Optional `native_avx512`
enrolls an independently gated native suite; root owns shared extensions.
Root also integrates overlapping native annotations and shared driver changes.

Every MagicPIG ledger ID ending `-CPU` must map to an enrolled fixture; every
fixture must have a ledger mapping. Default execution runs all portable suites;
repeatable `--suite NAME` selects focused runs and rejects unknown/duplicate
names. BF16 has a separate fixed, capability-gated native descriptor. CPU
oracles and adapters must not copy the production formula as their expectation.

#55, #56 and #57 may proceed in separate branches after #54 merges. #58 still
requires #57; later cache/composition/typing tasks follow the integration graph.
Actual GPU work remains explicitly postponed.

## SimHash and centering checkpoint (#55)

The separate `magicpig_python_suites/simhash.json` descriptor executes actual
archived AST blocks through the stdlib-only list-tensor fixture. The five blocks
cover packing, fill centering/hash, query normalization/hash and append centering.
Independent literal ordering/tie tests and 4094 integer codes K1..11 pass; common
key translation preserves dense FP64 attention, with distinct batch means.
[The contract](../../docs/MAGICPIG_SIMHASH_CPU.md) records FP64/BF16/Torch
substitutions and [packing defect #65](https://github.com/fregata-ariel/HashingDeepLearning/issues/65).

Python descriptors require name/test_file/source_file/class_name/blocks; source
and class are explicitly approved. Root reconstructs the recorded AST statement
hashes against the pinned source, records fixture/descriptor hashes and enforces
ledger enrollment. Python adapter execution is not an ASan/UBSan native test;
it is recorded as `not_applicable_python_adapter` in both plain/sanitized runs.
Default execution includes enrolled C++ and Python suites; focused `--suite`
selection covers both. Maintained production Python inventory remains 31 files.

The driver also supplies isolated expected-defect probe validation for future
LSH enrollment: exact wrong-result marker, fatal sanitizer outcome, allocation
site and count must match. Generic crashes or other UB do not satisfy it.
