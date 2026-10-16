# Initial Scatterbrain maintained CPU port

This port copies the initial release's five runtime project files at
`eeb689c00b9f322f33f6b6f85eaaac0a1d6a5eab`, preserving the
`src.models.modules` namespace and arithmetic. Put `ports/scatterbrain` on
`sys.path`; import `src.models.modules.attention.sblocal_attention`.
The immutable original remains in `third_party/scatterbrain`.

The maintained differences are explicit parameter/return/buffer types, private
typing protocols and factory-keyword declarations, postponed annotations,
identity `typing.cast` calls at framework boundaries, and documentation.
Erasing those type-only additions and docstrings leaves identical ASTs for all
five original modules. No sampling, log scale, normalization, residual,
masking, dropout, projection allocation or redraw expression was corrected.
The three original package markers are byte-identical copies. The attention
subdirectory retains the initial implicit namespace package layout.

[CPU contract](CPU_CONTRACT.md) records the paper mapping and tested boundary.
[Source provenance](SOURCE_PROVENANCE.json) records the original file hashes.
`typings/` declares the retained native and optional import interfaces; it
contains no replacement implementation. Strict checking includes every owned
Python file and stub through the `scatterbrain` group in
`tools/maintained_python_inventory.json`.

## Attribution

The copied root [Apache-2.0 license](LICENSE) applies with the source's existing
credits. The masking/feature helpers retain Idiap attribution; its historical
[MIT notice](notices/idiap-fast-transformers-LICENSE) is preserved. The retained
causal/Apex helper scaffold also requires the historical Phil Wang Performer
[MIT notice](notices/lucidrains-performer-pytorch-LICENSE). These notices are
copied exactly from the inspected historical snapshots. The runtime native
local-product dependency is distributed and built separately with its own
preserved MIT notice. This port contains no SMYRF/hash implementation.
