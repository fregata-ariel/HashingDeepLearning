#!/usr/bin/env bash
# Only this repository, one GPU, bounded resources and an explicit output bind.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo 'usage: run_gpu_container.sh NEW_ABSOLUTE_OUTPUT_DIRECTORY [components|sycl] [run|build-only] [all|PROFILE]' >&2
  exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=$1
profile=${2:-sycl}
mode=${3:-run}
selection=${4:-all}
[[ $profile == components || $profile == sycl ]] || { echo 'unknown image profile' >&2; exit 2; }
[[ $mode == run || $mode == build-only ]] || { echo 'unknown mode' >&2; exit 2; }
case $selection in
  all|gslide-native|gslide-library|mongoose-original|mongoose-wrapper|mongoose-fallback|mongoose-rng|scatterbrain|magicpig|sycl) ;;
  *) echo 'unknown execution profile' >&2; exit 2 ;;
esac
[[ $selection != sycl || $profile == sycl ]] || { echo 'SYCL image required' >&2; exit 2; }
[[ $selection != gslide-library || $profile == sycl ]] || { echo 'SYCL image required' >&2; exit 2; }
[[ $output == /* && ! -e $output ]] || { echo 'new absolute output directory required' >&2; exit 2; }
output=$(realpath -m -- "$output")
[[ ! -e $output ]] || { echo 'resolved output directory already exists' >&2; exit 2; }
[[ $output != "$repo" && $output != "$repo/"* ]] || { echo 'output must be outside the checkout' >&2; exit 2; }
mkdir -p -- "$output"
image=hashing-gpu:$profile
docker build --target "$profile" --progress plain --tag "$image" --file "$repo/docker/gpu/Dockerfile" "$repo" > "$output/image-build.log" 2>&1
docker image inspect "$image" > "$output/image.json"
if [[ $mode == build-only ]]; then exit 0; fi
args=(--only "$selection")
if [[ $profile == sycl ]]; then args+=(--sycl); fi
# Compiler/JIT caches and all writes are confined to the output and tmpfs.
# No host network, Docker socket, privilege, home or credential bind.
docker run --rm --init --gpus device=0 --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --cpus 4 --memory 16g --pids-limit 512 \
  --shm-size 512m --tmpfs /tmp:rw,exec,nosuid,size=8g \
  --mount "type=bind,src=$repo,dst=/workspace,readonly" \
  --mount "type=bind,src=$output,dst=/evidence" \
  --env XDG_CACHE_HOME=/evidence/cache --env CUPY_CACHE_DIR=/evidence/cache/cupy \
  --env TORCH_EXTENSIONS_DIR=/evidence/cache/torch \
  --env FLASHINFER_WORKSPACE_BASE=/evidence/cache/flashinfer \
  --env ACPP_APPDB_DIR=/evidence/cache/acpp \
  "$image" python tools/check_all_gpu.py --output /evidence "${args[@]}"
