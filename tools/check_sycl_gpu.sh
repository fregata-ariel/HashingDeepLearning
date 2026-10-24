#!/usr/bin/env bash
# NVIDIA backend only; the permanent CPU-only driver is unchanged.
set -euo pipefail
[[ $# == 1 && ! -e $1 ]] || { echo 'usage: check_sycl_gpu.sh NEW_OUTPUT_DIR' >&2; exit 2; }
output=$1
mkdir -p -- "$output"
export ACPP_TARGETS=generic ACPP_VISIBILITY_MASK=cuda
export ACPP_APPDB_DIR="$output/appdb"
failed=0
for spec in 'gslide:tests/gslide_sycl' 'mongoose:ports/mongoose/sycl' 'scatterbrain:ports/scatterbrain/sycl'; do
  name=${spec%%:*}
  source=${spec#*:}
  if cmake -S "$source" -B "$output/$name" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/acpp -DCMAKE_CXX_COMPILER=/usr/lib/llvm-18/bin/clang++ \
    -DHASHING_SYCL_REQUIRE_NVIDIA=ON -DPython3_EXECUTABLE=/opt/gpu-venv/bin/python \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
    cmake --build "$output/$name" --parallel 2 && \
    ctest --test-dir "$output/$name" --verbose --output-on-failure --no-tests=error --timeout 180; then
    echo "SYCL_NVIDIA_PROFILE $name PASS"
  else
    echo "SYCL_NVIDIA_PROFILE $name FAIL"
    failed=1
  fi
done
exit "$failed"
