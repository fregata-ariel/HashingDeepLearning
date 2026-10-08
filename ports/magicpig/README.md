# MagicPIG maintained baseline

This directory starts from four native files in the selected
`v0.2` snapshot `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`:
`library/lsh/lsh.cc`, `lsh.h`, and
`library/sparse_attention/sparse_attention.cc`, `sparse_attention.h`.
They are editable working copies; immutable originals, Python orchestration,
tests/examples and selected dependency inputs remain in `third_party/magicpig/`.

Issue #54 adds a dependency-free selected-body CPU driver and independent FP64
oracles. Portable software lanes always run; native AVX-512/FMA and BF16 bodies
run only after parent-process CPU/OS/compiler capability checks. Native-source
changes are Doxygen contracts, with arithmetic unchanged. Full Torch/FBGEMM
extension linkage, ABI, OpenMP scheduling and CPU-only Llama generation remain
unverified. See [the CPU validation contract](CPU_VALIDATION.md).

Archived `install.sh` installs historical CUDA/FlashInfer packages and must not
be used as the routine CPU setup. Python originals remain archival until their
maintained scope, runtime substitutions and strict typing are deliberately
enrolled. The existing 31-file maintained Python inventory is unchanged here.

Run source identity checks from the repository root:

```bash
python3 tools/check_magicpig_sources.py
python3 tests/traceability/test_magicpig_source_contracts.py
python3 tools/check_magicpig_cpu.py
python3 tools/check_magicpig_cpu.py --native --sanitize
```

The checker protects the archive, not equality of editable ports to upstream.
Initial port Git blob identities are recorded in the source pin. Subsequent
maintenance needs focused numerical tests and annotated contracts.

See [the integration plan](../../docs/MAGICPIG_INTEGRATION.md) and
[source pin](../../papers/magicpig-source-pin.json). Root source license is
Apache-2.0; selected dependency notices remain beside their archived sources.
