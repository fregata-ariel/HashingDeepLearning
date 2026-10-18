# MONGOOSE packing on a SYCL CPU device (#120)

`include/mongoose/simhash.hpp` is a maintained SYCL rewrite of the sign packing
in `ports/mongoose/mongoose_slide/slide_lib/simHash.py`, baseline
`097b79528f3b0eceebd715ecc6460e9236ecfd78`. It retains the MONGOOSE MIT license
and HazyResearch attribution in `../LICENSE`; immutable archives are unchanged.
`MONGOOSE-PACKING-SYCL-CPU` is new evidence, separate from all existing CPU IDs.

Inputs are contiguous value-owned FP32 vectors: data `[N,d]`, weights `[d,kL]`,
projection `[N,L,k]`; `N,d,L > 0`, `1 <= k <= 32`, and projection storage must
fit an int index. Input/projection values must be finite. Bit i is set exactly
when projection i is strictly positive (both signs of zero are false). The
result is row-major `[N,L]` int32, or a contiguous `[L,N]` transpose. Bit31 is
interpreted as two's-complement int32; all32 bits set yields -1. The 32-bit
boundary follows the original 32-lane launch domain; the existing Torch
fallback's k33..63 domain is deliberately not added to this API.

The caller supplies a real SYCL queue. Each code and projection entry is owned
by a different work-item; no subgroup/warp property is assumed. Buffers own
runtime allocations, borrow host vectors only within synchronous calls, and
copy back on destruction. `wait_and_throw` propagates submission/asynchronous
errors before return. No serial C++ fallback or skip is available. The same
kernel source can receive a separately validated GPU queue, but GPU behavior
is untested.

Build after activating the shared #114 toolchain:

```sh
cmake -S ports/mongoose/sycl -B /tmp/mongoose-sycl -DCMAKE_CXX_COMPILER=clang++
cmake --build /tmp/mongoose-sycl -j2
ctest --test-dir /tmp/mongoose-sycl --output-on-failure
python ports/mongoose/sycl/tests/fallback.py
```

The C++ fixture checks literal codes9/6/0/15, zeros, signed bit31, k1/2/7/31/32,
multiple rows/tables, projection with FP64 scalar expectations, transpose,
repeated calls, rejection, and insertion/query against the actual maintained
`lsh_lib/LSH.cpp`, with independent expected bucket unions. This native table
connection uses C++ int32 keys directly; it does not claim the Cython ABI ran.
The Python fixture separately verifies the existing maintained Torch fallback
on identical projection cases. Shared run tooling records checkout/source/test/
binary hashes and device/compiler descriptions; output JSON marks CUDA false.

The legacy `lsh_lib/matrix_simhash.py` remains GPU-only int64 and is excluded
from this new maintained API. `rg matrix_simhash` finds only its own definitions
and `test_mongoose_native_python_helpers.py`; no maintained production import
uses it. Its source is retained for reference, not retired or represented as
executed. Legacy generation/hash/transpose remain unexecuted. This explicit
scope keeps its distinct int64 API rather than silently narrowing it to int32.

Original CUDA compatibility is a separate result: the source uses a long-valued
`__shfl_down_sync(0xFFFFFFFF,value,offset,32)` across32 participating lanes. A
SYCL code-per-work-item loop is not execution of that body. The #114 AdaptiveCpp
generic JIT/OpenMP build supplies SYCL CPU execution and PCUDA compilation,
but PCUDA lacks this shuffle wrapper. The original-source compile probe is recorded
separately when run; no renamed/mock shuffle or CUDA stream substitute is used.
The exact-body probe includes the standard `<pcuda.hpp>` header and fails at
`__shfl_down_sync`; all32-lane mask/width/long shuffle semantics therefore remain
unexecuted. Original CUDA, CuPy/PyNVRTC loading/current-stream ABI and native GPU execution
remain unvalidated under #121.
