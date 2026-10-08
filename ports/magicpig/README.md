# MagicPIG maintained baseline

This directory starts from four unchanged native files in the selected
`v0.2` snapshot `ac9aa36c866330ca6ad2ce342a7848d7df6f49bb`:
`library/lsh/lsh.cc`, `lsh.h`, and
`library/sparse_attention/sparse_attention.cc`, `sparse_attention.h`.
They are editable working copies; immutable originals, Python orchestration,
tests/examples and selected dependency inputs remain in `third_party/magicpig/`.

This checkpoint establishes provenance only. It does not build the extension,
execute attention or claim a CPU-only Llama runtime. Native source expects
Torch, OpenMP and AVX-512F; BF16 dispatch is conditional. Source/include inputs
for FBGEMM and cpuinfo are archived, but cpuinfo implementation linkage and
Torch wheel headers/symbols/ABI still need verification in Issue #54.

Archived `install.sh` installs historical CUDA/FlashInfer packages and must not
be used as the routine CPU setup. Python originals remain archival until their
maintained scope, runtime substitutions and strict typing are deliberately
enrolled. The existing 31-file maintained Python inventory is unchanged here.

Run source identity checks from the repository root:

```bash
python3 tools/check_magicpig_sources.py
python3 tests/traceability/test_magicpig_source_contracts.py
```

The checker protects the archive, not equality of editable ports to upstream.
Initial port Git blob identities are recorded in the source pin. Subsequent
maintenance needs focused numerical tests and annotated contracts.

See [the integration plan](../../docs/MAGICPIG_INTEGRATION.md) and
[source pin](../../papers/magicpig-source-pin.json). Root source license is
Apache-2.0; selected dependency notices remain beside their archived sources.
