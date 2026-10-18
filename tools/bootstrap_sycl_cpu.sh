#!/usr/bin/env bash
# Reproduce the real AdaptiveCpp LLVM/OpenMP CPU backend on Debian 13 amd64.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=${ACPP_TOOLCHAIN_ROOT:-/workspace/.sycl-toolchain}
revision=31db2e83903c1db9f50b03b620eb0a09224cd0eb
[[ $(dpkg --print-architecture) == amd64 ]] || { echo 'requires Debian13 amd64' >&2; exit 1; }
. /etc/os-release
[[ ${ID:-} == debian && ${VERSION_ID:-} == 13 ]] || { echo 'requires Debian13 amd64' >&2; exit 1; }
mkdir -p "$root/setup" "$root/apt/packages" "$root/debian-root"
root=$(realpath "$root")
# Existing verified caches still run all numerical suites; only installation is reused.
if [[ -f "$root/install/toolchain-manifest.txt" && -f "$root/install/runtime-sha256.txt" ]]; then
  grep -qx "AdaptiveCpp_commit=$revision" "$root/install/toolchain-manifest.txt"
  (cd "$root/install" && sha256sum --check --quiet runtime-sha256.txt)
  echo "SYCL_TOOLCHAIN_CACHE_VERIFIED $root"
  exit 0
fi
# Tool versions are pinned; they do not supply the SYCL compiler or runtime.
python3 -m venv "$root/bootstrap-venv"
"$root/bootstrap-venv/bin/python" -m pip install --disable-pip-version-check \
  --target "$root/python-tools" cmake==4.2.3 ninja==1.13.0 > "$root/setup/python-tools.log" 2>&1
# These exact hashes were derived from Debian's signed trixie package metadata.
# Extraction needs no root and does not modify the distribution installation.
while IFS=$'\t' read -r digest url filename; do
  artifact="$root/apt/packages/$filename"
  if ! printf '%s  %s\n' "$digest" "$artifact" | sha256sum --check --status; then
    curl --fail --silent --show-error --location --retry 2 "$url" -o "$artifact"
  fi
  printf '%s  %s\n' "$digest" "$artifact" | sha256sum --check --status
  dpkg-deb --extract "$artifact" "$root/debian-root"
done < "$repo/tools/sycl-cpu-packages.tsv"
if [[ ! -d "$root/AdaptiveCpp/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/AdaptiveCpp/AdaptiveCpp.git "$root/AdaptiveCpp"
fi
git -C "$root/AdaptiveCpp" checkout --detach "$revision"
[[ $(git -C "$root/AdaptiveCpp" rev-parse HEAD) == "$revision" ]]
source_digest=$(git -C "$root/AdaptiveCpp" archive "$revision" | sha256sum | cut -d' ' -f1)
[[ "$source_digest" == 1f0816f8c6272334ee4c5ba8315828178c28a2bf35d82c82c07f3e3118be661a ]]
[[ -z $(git -C "$root/AdaptiveCpp" status --porcelain) ]] || { echo 'modified compiler source' >&2; exit 1; }
# Emit activation instructions; quoted heredoc preserves the caller PATH.
cat > "$root/env.sh" <<'TOOLCHAIN_ENV'
export ACPP_TOOLCHAIN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SYCL_TOOLCHAIN_PREFIX="$ACPP_TOOLCHAIN_ROOT/install"
export ACPP_INSTALL_DIR="$SYCL_TOOLCHAIN_PREFIX"
export SYCL_CXX="$SYCL_TOOLCHAIN_PREFIX/bin/acpp"
export PATH="$SYCL_TOOLCHAIN_PREFIX/bin:$ACPP_TOOLCHAIN_ROOT/debian-root/usr/lib/llvm-19/bin:$ACPP_TOOLCHAIN_ROOT/python-tools/bin:$PATH"
export PYTHONPATH="$ACPP_TOOLCHAIN_ROOT/python-tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SYCL_TOOLCHAIN_PREFIX/lib:$ACPP_TOOLCHAIN_ROOT/debian-root/usr/lib/llvm-19/lib:$ACPP_TOOLCHAIN_ROOT/debian-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$SYCL_TOOLCHAIN_PREFIX:$ACPP_TOOLCHAIN_ROOT/debian-root/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ACPP_APPDB_DIR="$ACPP_TOOLCHAIN_ROOT/appdb"
mkdir -p "$ACPP_APPDB_DIR"
export ACPP_TARGETS=generic
export ACPP_VISIBILITY_MASK=omp
export OMP_NUM_THREADS=4
TOOLCHAIN_ENV
source "$root/env.sh"
"$root/debian-root/usr/lib/llvm-19/bin/clang++" --version | tee "$root/setup/compiler-version.txt"
grep -q 'clang version 19.1.7' "$root/setup/compiler-version.txt"
"$root/debian-root/usr/bin/time" -v -o "$root/setup/toolchain-configure-time.txt" cmake -S "$root/AdaptiveCpp" -B "$root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/install" \
  -DCMAKE_CXX_COMPILER="$root/debian-root/usr/lib/llvm-19/bin/clang++" \
  -DCMAKE_C_COMPILER="$root/debian-root/usr/lib/llvm-19/bin/clang" \
  -DLLVM_DIR="$root/debian-root/usr/lib/llvm-19/lib/cmake/llvm" \
  -DClang_DIR="$root/debian-root/usr/lib/llvm-19/lib/cmake/clang" \
  -DBOOST_ROOT="$root/debian-root/usr" -DACPP_COMPILER_FEATURE_PROFILE=full \
  -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF \
  -DWITH_LEVEL_ZERO_BACKEND=OFF -DBUILD_TESTING=OFF 2>&1 | tee "$root/setup/configure.log"
"$root/debian-root/usr/bin/time" -v -o "$root/setup/toolchain-build-time.txt" cmake --build "$root/build" --parallel "${SYCL_BUILD_JOBS:-2}" 2>&1 | tee "$root/setup/build.log"
cmake --install "$root/build" 2>&1 | tee "$root/setup/install.log"
(
  cd "$root/install"
  find bin lib include etc -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum ../debian-root/usr/lib/llvm-19/bin/clang \
    ../debian-root/usr/lib/x86_64-linux-gnu/libLLVM.so.19.1 \
    ../debian-root/usr/lib/llvm-19/lib/libomp.so.5
) > "$root/install/runtime-sha256.txt"
{
  printf 'AdaptiveCpp_commit=%s\nLLVM_version=19.1.7\ncompilation_flow=generic\nruntime_backend=OpenMP\n' "$revision"
  printf 'AdaptiveCpp_git_archive_sha256=%s\n' "$source_digest"
  printf 'compiler_feature_profile=full\nGPU_backends_built=none\n'
  sha256sum "$repo/tools/sycl-cpu-packages.tsv" "$repo/tools/bootstrap_sycl_cpu.sh"
  cat "$root/setup/compiler-version.txt"
  dpkg-query -W -f='${Package}=${Version}\n' libc6 libstdc++6 libgcc-s1
  cat "$root/install/runtime-sha256.txt"
} > "$root/install/toolchain-manifest.txt"
cp "$repo/tools/sycl-cpu-packages.tsv" "$root/install/dependency-artifacts.tsv"
echo "SYCL_TOOLCHAIN_READY source $root/env.sh"
