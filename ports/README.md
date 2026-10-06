# Maintained / annotated working copies

`third_party/` is the provenance-preserving archive of the three imported
upstream snapshots. Do not edit it for commentary or portability fixes.

`ports/` starts from exactly the same Git blobs, then carries our work:

- paper-to-code Doxygen comments and Python docstrings;
- compatibility fixes, when we intentionally decide to implement an Issue;
- additional tests needed to validate those fixes.

The first `ports/` commit contains no source-code changes relative to
`third_party/`. Annotation and functional changes are kept in later,
reviewable commits.

When a comment says **Paper mapping**, it identifies the smallest relevant
paper section/algorithm/figure. **Implementation note** describes what this
specific snapshot does. **Reported effect** is reserved for measurements
reported by the paper, not measurements inferred from the source.
