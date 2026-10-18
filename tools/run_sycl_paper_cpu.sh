#!/usr/bin/env bash
# Required paper acceptance on the real CPU backend; missing targets fail.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evidence="$repo/sycl-cpu-evidence/papers"
python=${SYCL_CPU_PYTHON:-python3}
usage() { echo "usage: $0 [--python PYTHON] [--build-dir EVIDENCE_DIR]"; }
while (($#)); do
  case "$1" in
    --python|--build-dir)
      (($# >= 2)) && [[ -n "$2" ]] || { usage >&2; exit 2; }
      case "$1" in
        --python) python=$2 ;;
        --build-dir) evidence=$2 ;;
      esac
      shift 2 ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
cd "$repo"
python=$(command -v "$python")
mkdir -p "$evidence"
evidence=$(realpath "$evidence")
# Never accept an old paper binary or old acceptance report after a failed build.
run=$(mktemp -d "$evidence/run.XXXXXX")
printf '%s\n' "$run" > "$evidence/latest-run.txt"
status=FAIL
trap 'printf "status=%s\nevidence=%s\n" "$status" "$run" > "$run/result.txt"' EXIT
printf 'step\texit_code\ttimeout_seconds\n' > "$run/exit-status.tsv"

# Paper evidence helpers perform child Git reads. Scope container ownership trust
# to this process tree and this checkout, preserving any caller configuration.
git_config_count=${GIT_CONFIG_COUNT:-0}
[[ "$git_config_count" =~ ^[0-9]+$ ]] || { echo 'invalid GIT_CONFIG_COUNT' >&2; exit 2; }
export "GIT_CONFIG_KEY_${git_config_count}=safe.directory"
export "GIT_CONFIG_VALUE_${git_config_count}=$repo"
export GIT_CONFIG_COUNT=$((git_config_count + 1))
export ACPP_VISIBILITY_MASK=omp OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-4}
export ACPP_APPDB_DIR="$run/appdb"
export PYTHONPYCACHEPREFIX="$run/bytecode"
mkdir -p "$ACPP_APPDB_DIR"

annotate_failure() {
  local message=$1
  message=${message//'%'/'%25'}
  message=${message//$'\r'/'%0D'}
  message=${message//$'\n'/'%0A'}
  printf '::error title=SYCL CPU paper acceptance failed::%s\n' "$message" >&2
}
run_step() {
  local name=$1 seconds=$2 command_status rendered
  shift 2
  printf -v rendered '%q ' "$@"
  printf '%s\n' "$rendered" > "$run/$name.command.txt"
  if timeout --signal=TERM --kill-after=15s "${seconds}s" "$@" 2>&1 | tee "$run/$name.log"; then
    command_status=0
  else
    command_status=$?
  fi
  printf '%s\t%s\t%s\n' "$name" "$command_status" "$seconds" >> "$run/exit-status.tsv"
  if ((command_status)); then
    # Only build/test output is included; never print environment or credentials.
    annotate_failure "step=$name; exit=$command_status; command=$rendered; $(tail -n 20 "$run/$name.log" | tail -c 4096)"
    exit "$command_status"
  fi
}

run_step python-runtime 30 "$python" -c 'import sys, torch, numpy, einops, tqdm; assert torch.version.cuda is None; print("python=" + sys.version); print("torch=" + torch.__version__, "numpy=" + numpy.__version__, "einops=" + einops.__version__, "tqdm=" + tqdm.__version__)'
sha256sum tools/run_sycl_paper_cpu.sh tools/check_gslide_sycl.py tools/check_gslide_source_regressions.py tools/check_gslide_train_adapter.py tools/check_sycl_cpu.sh > "$run/integration-source-sha256.txt"

# G-SLIDE: actual shared kernels + production training fixture + exact native host
# caller with mocked CUDA/layers, then separately characterized source regressions.
run_step gslide 300 "$python" tools/check_gslide_sycl.py --build-dir "$run/gslide"
run_step gslide-source-regressions 90 "$python" tools/check_gslide_source_regressions.py --build-dir "$run/gslide-source-regressions"

# MONGOOSE: real production LSH packing fixture, separate Torch fallback, and the
# unchanged original CUDA compile-only probe (never original CUDA execution).
run_step mongoose 180 tools/check_sycl_cpu.sh --source-dir ports/mongoose/sycl --build-dir "$run/mongoose"
mongoose_build=$(sed -n 's/^build=//p' "$run/mongoose/result.txt")
run_step mongoose-paths 120 "$python" ports/mongoose/sycl/tests/validate.py --binary "$mongoose_build/mongoose_sycl_fixture" --evidence "$run/mongoose/paths"

# Scatterbrain: shared C ABI library, four local kernels, maintained Torch hybrid
# forward/backward and a distinct cold-cache foreign-thread regression.
run_step scatterbrain 300 tools/check_sycl_cpu.sh --source-dir ports/scatterbrain/sycl --build-dir "$run/scatterbrain" --cmake-arg "-DPython3_EXECUTABLE=$python"
# Bind generated source adapters and the shared C ABI object to this run too.
find "$run/gslide" "$run/gslide-source-regressions" -type f \( -name '*.cpp' -o -executable \) -print0 | sort -z | xargs -0 -r sha256sum > "$run/generated-and-binary-sha256.txt"
status=PASS
printf 'SYCL_CPU_PAPER_ACCEPTANCE_PASS evidence=%s\n' "$run"
