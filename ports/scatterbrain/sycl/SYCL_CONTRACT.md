# Initial SBLocal with actual SYCL local products (#119)

This port provides a production shared SYCL implementation of four operations:
`local_dot_product`, `local_dot_backward`, `local_weighted_average`, and
`local_weighted_average_backward` in `include/scatterbrain/local_product.hpp`.
It rewrites the selected MIT Idiap CPU extension at
`f22c13716fc748bb21a7b226ada7f7b5f87f867f`, not an executed CUDA body. The
original notices remain in `../notices/idiap-fast-transformers-LICENSE`.
`SOURCE_PROVENANCE.json` pins the selected baseline and sources. Immutable
`third_party` files and existing maintained feature/state arithmetic are unchanged.

The C++ API takes a caller-selected SYCL queue and value-owned vectors, with
contiguous `[B,H,Q,E]` queries, `[B,H,K,E]` keys, `[B,H,Q,W]` local weights,
`[B,H,K,V]` values, `[Q,K]` additive mask and `[B]` int64 prefix lengths.
Q and K may differ. Every dimension must be positive; the conservative product
of all seven dimensions must fit int32. Prefix lengths may range from0 toK.
Inputs must be finite FP32. Local slot r uses key `i - W//2 + r`; dot fills
invalid prefix/physical slots with -1e24 and valid slots add the supplied mask.
Dot backward differentiates valid dot products; the additive mask is constant.
Weighted average uses physical K boundaries, independently of prefix lengths,
matching the original API. Its invalid physical slots have zero derivative.

Every output scalar belongs to one real kernel work-item. Backward reductions
are gathered per output scalar instead of atomically scattering, so no atomic,
subgroup, tile-size or scheduling assumption is needed. SYCL buffers own device
allocations and borrow host vectors only within a synchronous operation.
`wait_and_throw` and buffer destruction complete work/copyback before return;
no pointer escapes. The C adapter uses a call-scoped queue, avoiding a static
queue whose destruction can outlive runtime state at Python process shutdown.
The Python ctypes boundary owns contiguous NumPy arrays for the call and checks
CPU/FP32/rank/shape/dtype, including int64 lengths, before entering C. C ABI host calls are serialized by a mutex, including queue/context lifetime;
Python loading and calls additionally share a lock. This bounds foreign-thread
runtime use without serializing kernel work-items or substituting arithmetic.
Loading the same library is idempotent; replacing a live library is rejected.
C ABI callers must supply buffers of the documented sizes. Exceptions return an error
and no fallback runs. The Python autograd wrapper supports first derivatives;
`once_differentiable` rejects higher derivative use.

`sycl_local_product` provides only these actual SYCL-backed operations. Place
this directory and the maintained port root on PYTHONPATH, import `sblocal_sycl`,
and call `sycl_local_product.load_library` before construction. The wrapper
explicitly binds the initial source's `fast_transformers.local_product` import
to this real numerical module; an already selected different backend is rejected.
This does not claim that the original native autograd wrapper ran.
`sblocal_sycl.SBLocalSYCLAttention` wraps the maintained Apache-2.0 initial Torch
hybrid composition. Its low-rank/features/normalization remain maintained Torch
CPU code, while both local forward operations and their backward operations
run these production SYCL kernels. This is not a claim of an all-SYCL hybrid.
It rejects causal, stochastic dropout, feature eps, arbitrary attention masks,
non-CPU host tensors and non-FP32 input. The hybrid wrapper explicitly rejects zero/invalid prefixes. Hybrid fixtures
select moderate finite inputs, positive key lengths and finite positive row
mass; zero prefix is validated only for local dot. Unsupported extreme/all-masked hybrid behavior,
feature sampling laws, CUDA/half/Apex, later SBSmyrf, full models/optimizer
checkpoints and performance remain outside this API. No silent causal fallback
or new causal dependency is introduced.

After activating the shared #114 toolchain and CPU Python environment:

```sh
cmake -S ports/scatterbrain/sycl -B /tmp/scatterbrain-sycl \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/workspace/.magicpig-cpu-venv/bin/python
cmake --build /tmp/scatterbrain-sycl -j2
ctest --test-dir /tmp/scatterbrain-sycl --output-on-failure
MYPYPATH=ports/scatterbrain/sycl:ports/scatterbrain:ports/scatterbrain/typings \
  python -m mypy --strict --disallow-any-explicit \
  ports/scatterbrain/sycl/sycl_local_product.py \
  ports/scatterbrain/sycl/sblocal_sycl.py ports/scatterbrain/sycl/tests/validate.py
```

`SCATTERBRAIN-LOCAL-PRODUCT-SYCL-CPU` reuses the existing independent dense/gather
oracle, now dispatching all4 operations to this real shared library. It covers
B2/H2/Q5/K4/E3/V2, windows1/2/3/4/9, mask effects, prefix lengths4/3, forward
and all Q/K/weight/V gradients. Eight extra repeated scalar-FP64 cases for all four operations cover
Q/K1/1,7/3,3/7,17/16 and zero prefixes, boundaries and odd/even windows. These use detached FP64
leaves for Q/K and weight/V gradient expectations.
`SCATTERBRAIN-LOCAL-HYBRID-SYCL-CPU` uses the existing independent FP64 cosh
oracle conditioned on an actually sampled projection: ten contexts/temperature
cases check raw H, HV, H1, common normalization, weights, outputs and Q/K/V
gradients. Two full-support cases equal dense softmax; four training/eval
lifecycle calls preserve the existing behavior. Both subtraction and mismatched
normalizer mutants are rejected in all10 cases, using actual SYCL dispatch.
The unchanged independent fixture's 2e-6..4e-6 FP32 tolerances allow bounded
rounding against FP64; extra larger scalar/FP64 reductions use5e-6. Both actual
and reference values must be finite before their errors are compared.

CTest requires successful process exit, a real CPU selector and complete test
counts. Evidence hashes the library, kernel/adapter/fixture sources and exact
checkout; `sb_describe` prints actual device/vendor/type/runtime information to
the captured test log. Shared #114 tooling records compiler/build/log identity.
Source/binary/import identity and existing CPU/C++ fixture passes are distinct
from these new SYCL execution IDs. GPU/CUDA, memory sanitizers and statistical
or paper accuracy/performance claims remain unvalidated. CI and central
traceability enrollment are integrated separately by the shared owner.

The `scatterbrain_sycl_threads` CTest is a process-lifetime regression: four
foreign Python threads each execute three local-dot/weighted-average/autograd
calls and invalid-prefix rejections, comparing output and all Q/K/V gradients
with detached FP64 expectations. It then repeats same-library selection and
garbage collection three times; the process must exit successfully. Its
separate appdb directory exercises cold JIT on a fresh build and warm repeats.
Initial unrestricted foreign-thread calls intermittently crashed, so the host
adapter deliberately serializes submissions and lifetime. No backend-wide
thread-safety claim or root-cause claim is inferred from this bounded repair.
