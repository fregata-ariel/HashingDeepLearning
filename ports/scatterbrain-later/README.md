# Maintained later Scatterbrain CPU component

`scatterbrain_later.attention.SBSmyrfAttention` preserves the selected actual
later noncausal hybrid forward and rank-hashing helpers from Fly `74cc6b1`.
It repairs FP64 hash allocations, count-log precision and returned weights.
The original eleven-file source and the frozen-reference characterization
remain separate historical evidence. This is a changed maintained subset.

Use both `ports/scatterbrain-later` and `ports/scatterbrain` on the import path.
The shared initial feature-map lifecycle repair (#106) is a required dependency
for continued-training/dtype-movement acceptance. No feature/kernel/mask fork
is included here. CPU Torch and einops suffice; no NumPy/tqdm, experiment stack,
local-product extension, causal extension, Apex or GPU kernel is required by
this selected noncausal execution path.

Run `python -B tools/check_scatterbrain_sbsmyrf_cpu.py --emit-dir <directory>`
from the repository with the validated CPU environment. The exact frozen
reference is fetched ephemerally outside the artifact directory; optional
`--reference-source-dir` accepts a clean hash-checked checkout. See
[CPU_CONTRACT.md](CPU_CONTRACT.md), [NOTICE](NOTICE) and
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). The selected SMYRF-derived
component retains GPL-3.0 terms and separate Apache/MIT notices; this is not a
license declaration for the entire upstream repository.
