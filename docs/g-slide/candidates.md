# G-SLIDE candidate and rebuild CPU oracle

Run the isolated suite from the repository root:

```bash
python3 tools/check_gslide_cpu.py --suite candidates --build-dir build/gslide-candidates
```

The suite extracts maintained CUDA bodies at invocation and includes the actual
linked-table header and `filter::operator()`. It does not copy those algorithms
into a reference implementation. Three cases use independently enumerated
addresses, frequency maps, sets, gathered lists and CSC offsets.

| Traceability ID | Checked behavior | Correspondence |
| --- | --- | --- |
| `GSLIDE-CANDIDATE-FILTER-CPU` | Cutoff below/at/above 2, forced linked collisions, promotion of low-count, active and new labels, repeated labels, isolated samples, empty sample and fresh query storage | Direct serial mapping of frequency cutoff and label activation |
| `GSLIDE-INDEX-REBUILD-CPU` | A changed weight maximum leaves an existing index stale; explicit reset and rebuild recompute its bucket addresses and gathered membership | Variant preserving released natural-log packing |
| `GSLIDE-BUCKET-RING-CPU` | Four insertions into a two-slot bucket retain bounded ring payloads, preserve raw counts, clamp query counts and yield expected downstream filtered sets | Implementation support for bounded storage |

CPU shared-memory bodies run with one thread per block. Linked and bucket
insertion order is serial; the exact physical chain and retained ring payload
are serial oracles. Parallel GPU order, surviving candidates, synchronization,
lock contention and memory fences remain pending. Linked tables have sufficient
capacity before every insertion; pool exhaustion is not exercised.

The prefix scan and copy/filter traversal are explicitly host adapters. They
call the maintained predicate but do not execute Thrust or production
`get_act_nodes`. Recreated `OwnedTable` allocations and manually zeroed bucket
counts model fresh state without executing CUDA allocation/reset APIs. The
parameter change is an explicit fixture mutation, not optimizer or
`Network::train` execution, and does not test LSH rebuild scheduling.

WTA packing remains `floor(ln(bin_size))`, with the bin-size-8, K=2 alias of
tuples `(0,4)` and `(1,0)`. The paper's bit-width packing is not substituted.
Alternate `init_hash_tt_knl` and `init_hash_knl` remain untested: their `int smem`
declarations conflict with the existing adapter's `char smem`. Random padding,
CUDA launches, GPU gather/filter/scan, and allocation/free ownership are also
outside this suite. Plain CPU and local UBSan runs validate only the exercised
host execution; hosted ASan remains the sanitizer completion gate.
