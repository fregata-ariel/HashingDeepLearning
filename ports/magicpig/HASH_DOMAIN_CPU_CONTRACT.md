# Maintained K-domain policy (#65)

The archived key/query SimHash packing stores powers and accumulated codes in
FP16. All integer codes are exact for K1..11; all-one K12/13/14/15 codes round to
4096/8192/16384/32768. The last example also exceeds positive signed int16.

The maintained `LSH::alloc` entry now rejects K outside1..11 with `TORCH_CHECK`
before computing buckets, allocating storage or writing owner members. This
chooses an explicit accepted-domain policy, rather than introducing an untested
Python port or changing the archive. Both key table construction and query
lookup use that owner's single K. A rejected call on an already populated owner
leaves its prior dimensions/tables intact; valid reallocation remains unsupported.

`MAGICPIG-HASH-DOMAIN-CPU` executes maintained allocation/fastfill/retrieval and
destruction bodies. Independent integer bit enumeration checks4094 accepted
codes, including zero, across K1..11. Each key/query code must retrieve exactly
its unique ID from two tables. Eighteen unsupported-K calls test fresh and
populated owners, including K11/12 and K14/15 boundaries and extreme integers,
before state mutation. Existing `MAGICPIG-SIMHASH-CPU` continues to exhaust the
archived packing AST across the same accepted4094 codes and retain outside-domain
rounding examples as provenance characterization.

```
python3 tools/check_magicpig_cpu.py --suite hash_domain --suite simhash
python3 tools/check_magicpig_cpu.py --suite hash_domain --sanitize
```

This maintained variant enforces the exact archived packing domain. It does not
support K>=12, execute real Torch/BF16 projection, or prove Torch ABI. Confirm the
policy with CPU Torch when a maintained tensor integration exists; no such model
port is created by this repair. No GPU/model/download prerequisite is introduced.
Other L/K/domain validation for correction remains Issue67; this repair does not
broaden the allocation configuration or reallocation contract.
