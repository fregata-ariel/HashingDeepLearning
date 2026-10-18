#!/usr/bin/env bash
# Build a bounded, explicitly selected SYCL suite and retain reproducible evidence.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir="$repo/tests/sycl_cpu_backend"
evidence="$repo/sycl-cpu-evidence"
while (($#)); do
  case "$1" in
    --source-dir) source_dir=$2; shift 2 ;;
    --build-dir) evidence=$2; shift 2 ;;
    *) echo "usage: $0 [--source-dir CMAKE_SUITE] [--build-dir EVIDENCE_DIR]" >&2; exit 2 ;;
  esac
done
source_dir=$(realpath "$source_dir")
[[ "$source_dir" == "$repo/"* ]] || { echo 'suite must belong to this checkout' >&2; exit 2; }
mkdir -p "$evidence"
evidence=$(realpath "$evidence")
stage=preflight
stage_log=
status=FAIL
build=not-created
: > "$evidence/driver-stderr.log"
: > "$evidence/git-stderr.log"
# Retain complete stderr, then flush the collector before reading it on failure.
exec 3>&2
exec 2> >(tee -a "$evidence/driver-stderr.log" >&3)
stderr_collector=$!
finish() {
  local exit_code=$1 diagnostic
  exec 2>&3
  wait "$stderr_collector" || true
  printf 'status=%s\nbuild=%s\nstage=%s\nexit_code=%s\n' \
    "$status" "$build" "$stage" "$exit_code" > "$evidence/result.txt"
  if ((exit_code != 0)); then
    {
      printf 'stage=%s exit=%s\n' "$stage" "$exit_code"
      if [[ -n "$stage_log" && -f "$stage_log" ]]; then
        tail -n 12 "$stage_log"
      fi
      tail -n 6 "$evidence/driver-stderr.log"
    } > "$evidence/failure-summary.txt"
    diagnostic=$(tail -c 3072 "$evidence/failure-summary.txt")
    diagnostic="stage=$stage exit=$exit_code; $diagnostic"
    diagnostic=${diagnostic//'%'/'%25'}
    diagnostic=${diagnostic//$'\r'/'%0D'}
    diagnostic=${diagnostic//$'\n'/'%0A'}
    printf '::error title=SYCL CPU stage failed::%s\n' "$diagnostic" >&2
  fi
  exit "$exit_code"
}
trap 'finish "$?"' EXIT
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
stage=checkout
# Assign separately: printf would hide a failing command substitution's status.
checkout=$(git_repo rev-parse HEAD)
tree=$(git_repo rev-parse 'HEAD^{tree}')
stage=environment
stage_log="$evidence/environment.txt"
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
stage=checkout-patch
stage_log=
git_repo diff --binary HEAD > "$evidence/checkout.patch"
stage=source-hashes
(
  cd "$repo"
  git_repo ls-files -z -- ports "$source_dir" tools/check_sycl_cpu.sh tools/bootstrap_sycl_cpu.sh |
    sort -z | xargs -0 -r sha256sum
) > "$evidence/source-sha256.txt"
stage=toolchain-manifest
if [[ -f "$SYCL_TOOLCHAIN_PREFIX/toolchain-manifest.txt" ]]; then
  cp "$SYCL_TOOLCHAIN_PREFIX/toolchain-manifest.txt" "$evidence/toolchain-manifest.txt"
  cp "$SYCL_TOOLCHAIN_PREFIX/dependency-artifacts.tsv" "$evidence/dependency-artifacts.tsv"
else
  echo 'missing required pinned toolchain manifest' >&2
  exit 1
fi
stage=configure
stage_log="$evidence/configure.log"
"$time_bin" -v -o "$evidence/configure-time.txt" cmake -S "$source_dir" -B "$build" \
  -DCMAKE_PREFIX_PATH="$SYCL_TOOLCHAIN_PREFIX;${CMAKE_PREFIX_PATH:-}" -DCMAKE_CXX_COMPILER="$SYCL_CXX" \
  -DACPP_TARGETS="$ACPP_TARGETS" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$evidence/configure.log"
stage=clean-build
stage_log="$evidence/clean-build.log"
"$time_bin" -v -o "$evidence/clean-build-time.txt" cmake --build "$build" -j "$SYCL_BUILD_JOBS" 2>&1 | tee "$evidence/clean-build.log"
stage=cached-build
stage_log="$evidence/cached-build.log"
"$time_bin" -v -o "$evidence/cached-build-time.txt" cmake --build "$build" -j "$SYCL_BUILD_JOBS" 2>&1 | tee "$evidence/cached-build.log"
stage=binary-evidence
stage_log=
cp "$build/compile_commands.json" "$evidence/compile_commands.json"
cp "$build/CMakeCache.txt" "$evidence/CMakeCache.txt"
find "$build" -maxdepth 1 -type f -executable -print0 | sort -z | xargs -0 -r sha256sum > "$evidence/binary-sha256.txt"
[[ -s "$evidence/binary-sha256.txt" ]] || { echo 'no suite binary built' >&2; exit 1; }
# CTest's no-tests policy makes an empty/incorrect suite fail rather than pass.
stage=cold-tests
stage_log="$evidence/test.log"
"$time_bin" -v -o "$evidence/test-time.txt" ctest --test-dir "$build" --verbose --no-tests=error \
  --output-junit "$evidence/results.xml" 2>&1 | tee "$evidence/test.log"
stage=warm-tests
stage_log="$evidence/cached-test.log"
"$time_bin" -v -o "$evidence/cached-test-time.txt" ctest --test-dir "$build" --verbose --no-tests=error \
  --output-junit "$evidence/cached-results.xml" 2>&1 | tee "$evidence/cached-test.log"
stage=complete
status=PASS
printf 'SYCL_CPU_SUITE_PASS suite=%s evidence=%s\n' "$source_dir" "$evidence"
