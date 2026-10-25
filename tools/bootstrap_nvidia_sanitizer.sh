#!/usr/bin/env bash
# Local extraction only; caller removes the destination after the checkpoint.
set -euo pipefail
if [[ $# != 1 || -e $1 ]]; then
  echo 'usage: bootstrap_nvidia_sanitizer.sh NEW_WRITABLE_DIRECTORY' >&2
  exit 2
fi
root=$1
mkdir -p -- "$root"
root=$(realpath "$root")
cleanup() {
  status=$?
  if [[ $status != 0 ]]; then rm -rf -- "$root"; fi
}
trap cleanup EXIT
url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-12.9.79-archive.tar.xz
sha=e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf
curl --fail --silent --show-error --location --retry 2 "$url" -o "$root/archive.tar.xz"
printf '%s  %s\n' "$sha" "$root/archive.tar.xz" | sha256sum --check --status
tar --extract --xz --file "$root/archive.tar.xz" --directory "$root" --strip-components=1 --no-same-owner
rm -- "$root/archive.tar.xz"
printf 'url=%s\nsha256=%s\ncomponent_version=12.9.79\n' "$url" "$sha" > "$root/source.txt"
"$root/bin/compute-sanitizer" --version
