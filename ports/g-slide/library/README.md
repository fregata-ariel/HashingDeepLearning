# G-SLIDE SYCL component library

The package exposes the existing maintained G-SLIDE behavior through public
headers and CMake targets. Packaging does not change the numerical algorithms,
hash/label/bias semantics, model topology or execution policy. Further semantic
choices and a stable application API remain separate work.

| Header | Target | Use |
| --- | --- | --- |
| `<gslide/control.hpp>` | `gslide::control` | Shared training operation order; ordinary C++17. |
| `<gslide/host.hpp>` | `gslide::control` | Configuration, workspace planning, sparse-batch contracts and borrowed shape/capacity views; ordinary C++17. |
| `<gslide/sycl.hpp>` | `gslide::sycl` | Checked numerical components, storage owners and the current Network facade; requires a SYCL compiler/runtime supplied by the consumer. |

The types retain their current `gslide` and `gslide::sycl_port` namespaces.
Individual maintained headers are also installed under `g-slide/sycl/`.
Existing relative includes and unqualified include paths remain available for
compatibility; public entry headers are preferred for new consumers.

## Build and install

This standalone header-only project needs CMake 3.20 and a C++17 compiler.
Configuration and installation do not discover a SYCL SDK, select a device,
download dependencies or modify the system. Use a task-local installation:

```bash
cmake -S ports/g-slide/library -B /tmp/gslide-package-build \
  -DCMAKE_INSTALL_PREFIX=/tmp/gslide-prefix
cmake --build /tmp/gslide-package-build --parallel 2
cmake --install /tmp/gslide-package-build
```

The install contains headers, exported targets, package/version configuration,
the MIT license and the pinned reference-source manifest. Version `0.1.0` is
experimental; the generated version file accepts exact-version requests.
The installed package can be moved to a different prefix. It contains no
checkout or producer-build include paths.

## Consume from an independent project

An ordinary host consumer uses:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_host_program LANGUAGES CXX)
find_package(GSlide 0.1.0 EXACT CONFIG REQUIRED COMPONENTS control)
add_executable(my_host_program main.cpp)
target_link_libraries(my_host_program PRIVATE gslide::control)
```

Configure it with `-DCMAKE_PREFIX_PATH=/tmp/gslide-prefix`, and include
`<gslide/host.hpp>` from `main.cpp`. No CUDA/SYCL headers are needed.

The tested SYCL integration uses AdaptiveCpp explicitly in the consumer:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_sycl_program LANGUAGES CXX)
find_package(GSlide 0.1.0 EXACT CONFIG REQUIRED COMPONENTS sycl)
find_package(AdaptiveCpp CONFIG REQUIRED)
add_executable(my_sycl_program main.cpp)
target_link_libraries(my_sycl_program PRIVATE gslide::sycl)
add_sycl_to_target(TARGET my_sycl_program SOURCES main.cpp)
```

Include `<gslide/sycl.hpp>` and configure with both prefixes, for example
`-DCMAKE_PREFIX_PATH="/tmp/gslide-prefix;/opt/acpp"`, using the supported SYCL
compiler/toolchain. The consumer owns queue/device selection. The exported
`sycl` component signifies available headers/target, not automatic compiler or
device capability detection. An unknown required component fails configuration.

In-repository consumers can still use `add_subdirectory` and the same targets.
The original native CUDA project in `ports/g-slide/CMakeLists.txt` is unchanged.

## Verification

The SDK-free installed-consumer check is:

```bash
python3 tools/check_gslide_package.py --output /tmp/gslide-package-check-NEW
```

The focused Docker profile executes the existing 15 component/integration suites
on required CPU and NVIDIA devices, then installs, relocates and independently
consumes the package for each device:

```bash
bash tools/run_gpu_container.sh /tmp/gslide-package-devices-NEW sycl run gslide-library
```

`tests/gslide_consumer` is copied outside the checkout before configuration.
It uses only `find_package`, exported targets and public entry headers. Two host
translation units check workspace planning/views. Two SYCL translation units
call a component directly and perform a complete small training update plus
evaluation against an independent FP64 oracle. The fixture checks observable
change and the actual required device. Exact enrollment, completion markers,
compile commands without checkout paths, relocated payload hashes, binary hashes
and JUnit output are retained. The 15 existing suites remain distinct evidence.

The routine CI runs the SDK-free consumer and the installed real-SYCL CPU
consumer. GPU execution remains the explicit bounded Docker/manual profile.

The [2026-10-26 execution evidence](../../../docs/evidence/gslide-package-2026-10-26.json)
records 15/15 existing suites and 2/2 installed-consumer tests on each required
CPU/NVIDIA profile, plus an ordinary GNU C++ SDK-free installed consumer.
Both device consumers have maximum absolute error `5.1888e-8` against the FP64
oracle. All 23 installed public/implementation/controller headers match their
source bytes; the 20 existing numerical/controller headers match the incoming
commit exactly.
