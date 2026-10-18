#!/usr/bin/env bash
# Build a bounded, explicitly selected SYCL suite and retain reproducible evidence.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir="$repo/tests/sycl_cpu_backend"
evidence="$repo/sycl-cpu-evidence"
cmake_args=()
usage() { echo "usage: $0 [--source-dir CMAKE_SUITE] [--build-dir EVIDENCE_DIR] [--cmake-arg VALUE]..."; }
while (($#)); do
  case "$1" in
    --source-dir|--build-dir|--cmake-arg)
      (($# >= 2)) && [[ -n "$2" ]] || { usage >&2; exit 2; }
      case "$1" in
        --source-dir) source_dir=$2 ;;
        --build-dir) evidence=$2 ;;
        --cmake-arg) cmake_args+=("$2") ;;
      esac
      shift 2 ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
source_dir=$(realpath "$source_dir")
[[ "$source_dir" == "$repo/"* ]] || { echo 'suite must belong to this checkout' >&2; exit 2; }
mkdir -p "$evidence"
evidence=$(realpath "$evidence")
[[ -f "$source_dir/CMakeLists.txt" ]] || { echo 'missing explicit CMake suite' >&2; exit 2; }
: "${SYCL_TOOLCHAIN_PREFIX:=/workspace/.sycl-toolchain/install}"
: "${SYCL_CXX:=$SYCL_TOOLCHAIN_PREFIX/bin/acpp}"
: "${ACPP_TARGETS:=generic}"
: "${SYCL_BUILD_JOBS:=2}"
time_bin=${SYCL_TIME_BIN:-${ACPP_TOOLCHAIN_ROOT:-/workspace/.sycl-toolchain}/debian-root/usr/bin/time}
export ACPP_APPDB_DIR="$evidence/appdb"
mkdir -p "$ACPP_APPDB_DIR"
export ACPP_VISIBILITY_MASK=omp OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-4}
export LD_LIBRARY_PATH="$SYCL_TOOLCHAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# A unique build guarantees the first measured build cannot reuse suite objects.
build=$(mktemp -d "$evidence/build.XXXXXX")
status=FAIL
trap 'printf "status=%s\nbuild=%s\n" "$status" "$build" > "$evidence/result.txt"' EXIT
# Container checkout ownership can differ from the shell user. Trust only the
# explicit checkout for each read; do not alter HOME or global Git configuration.
git_repo() {
  local command_status diagnostic
  if git -c safe.directory="$repo" -C "$repo" "$@" 2> "$evidence/git-command-stderr.tmp"; then
    cat "$evidence/git-command-stderr.tmp" >> "$evidence/git-stderr.log"
  else
    command_status=$?
    cat "$evidence/git-command-stderr.tmp" >> "$evidence/git-stderr.log"
    diagnostic="git $*: exit=$command_status; $(head -c 2048 "$evidence/git-command-stderr.tmp")"
    diagnostic=${diagnostic//'%'/'%25'}
    diagnostic=${diagnostic//$'\r'/'%0D'}
    diagnostic=${diagnostic//$'\n'/'%0A'}
    printf '::error title=SYCL CPU Git evidence failed::%s\n' "$diagnostic" >&2
    return "$command_status"
  fi
}
# Assign separately: printf would hide a failing command substitution's status.
checkout=$(git_repo rev-parse HEAD)
tree=$(git_repo rev-parse 'HEAD^{tree}')
{
  printf 'checkout=%s\ntree=%s\nsuite=%s\n' "$checkout" "$tree" "$source_dir"
  printf 'compiler=%s\ntargets=%s\nvisibility=%s\nOMP_NUM_THREADS=%s\nOMP_THREAD_LIMIT=%s\n' "$SYCL_CXX" "$ACPP_TARGETS" "$ACPP_VISIBILITY_MASK" "$OMP_NUM_THREADS" "$OMP_THREAD_LIMIT"
  git_repo status --short
  sha256sum "$SYCL_CXX" "$time_bin"
  "$SYCL_CXX" --version
  cmake --version
  uname -a
  lscpu
} > "$evidence/environment.txt"
git_repo diff --binary HEAD > "$evidence/checkout.patch"
(
  cd "$repo"
  git_repo ls-files -z -- ports "$source_dir" tools/check_sycl_cpu.sh tools/bootstrap_sycl_cpu.sh |
    sort -z | xargs -0 -r sha256sum
) > "$evidence/source-sha256.txt"
if [[ -f "$SYCL_TOOLCHAIN_PREFIX/toolchain-manifest.txt" ]]; then
  cp "$SYCL_TOOLCHAIN_PREFIX/toolchain-manifest.txt" "$evidence/toolchain-manifest.txt"
  cp "$SYCL_TOOLCHAIN_PREFIX/dependency-artifacts.tsv" "$evidence/dependency-artifacts.tsv"
else
  echo 'missing required pinned toolchain manifest' >&2
  exit 1
fi
"$time_bin" -v -o "$evidence/configure-time.txt" cmake -S "$source_dir" -B "$build" \
  -DCMAKE_PREFIX_PATH="$SYCL_TOOLCHAIN_PREFIX;${CMAKE_PREFIX_PATH:-}" -DCMAKE_CXX_COMPILER="$SYCL_CXX" \
  -DACPP_TARGETS="$ACPP_TARGETS" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${cmake_args[@]}" 2>&1 | tee "$evidence/configure.log"
"$time_bin" -v -o "$evidence/clean-build-time.txt" cmake --build "$build" -j "$SYCL_BUILD_JOBS" 2>&1 | tee "$evidence/clean-build.log"
"$time_bin" -v -o "$evidence/cached-build-time.txt" cmake --build "$build" -j "$SYCL_BUILD_JOBS" 2>&1 | tee "$evidence/cached-build.log"
cp "$build/compile_commands.json" "$evidence/compile_commands.json"
cp "$build/CMakeCache.txt" "$evidence/CMakeCache.txt"
find "$build" -maxdepth 1 -type f -executable -print0 | sort -z | xargs -0 -r sha256sum > "$evidence/binary-sha256.txt"
[[ -s "$evidence/binary-sha256.txt" ]] || { echo 'no suite binary built' >&2; exit 1; }
# CTest's no-tests policy makes an empty/incorrect suite fail rather than pass.
"$time_bin" -v -o "$evidence/test-time.txt" ctest --test-dir "$build" --verbose --no-tests=error \
  --output-junit "$evidence/results.xml" 2>&1 | tee "$evidence/test.log"
"$time_bin" -v -o "$evidence/cached-test-time.txt" ctest --test-dir "$build" --verbose --no-tests=error \
  --output-junit "$evidence/cached-results.xml" 2>&1 | tee "$evidence/cached-test.log"
status=PASS
printf 'SYCL_CPU_SUITE_PASS suite=%s evidence=%s\n' "$source_dir" "$evidence"
