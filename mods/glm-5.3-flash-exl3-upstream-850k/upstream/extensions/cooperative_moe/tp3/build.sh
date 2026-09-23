#!/usr/bin/env bash
# Offline artifact build only. Never starts a model, installs dependencies, or pulls images.
set -euo pipefail
output_dir=${1:?Usage: build.sh EMPTY_OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty build output directory.' >&2; exit 1;
}
python3 "$source_dir/manifest.py" verify-sources "$source_dir"
mkdir -p "$output_dir/headers/quant" "$output_dir/source"
cp -a "$source_dir/vendor/exllamav3_ext/." "$output_dir/headers/"
cp "$source_dir/native/cooperative_moe_kernel.cuh" "$output_dir/headers/quant/glm53_coop_kernel.cuh"
cp "$source_dir/native/exl3_moe_coop.cuh" "$output_dir/headers/quant/"
cp -a "$source_dir/native" "$output_dir/source/"
cp "$source_dir/dispatch_policy.json" "$source_dir/runtime.py" "$source_dir/PROVENANCE.json" "$source_dir/LICENSE.MIT" "$source_dir/LICENSE.upstream-AGPL-3.0" "$output_dir/"
# Drop transport metadata from this newly created output tree only.
find "$output_dir" -name '._*' -type f -delete
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
flags=(-std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr
  -gencode arch=compute_121a,code=sm_121a -Xcompiler -fPIC --ptxas-options=-v
  -I "$output_dir/headers")
"$nvcc" --version > "$output_dir/toolchain.txt"
# Exactly two compiler jobs. No GPUs are required for compilation.
"$nvcc" "${flags[@]}" -DGLM53_ROWS_MAX=32 -c "$output_dir/source/native/cooperative_moe.cu" -o "$output_dir/rows32.o" > "$output_dir/build32.log" 2>&1 &
pid32=$!
"$nvcc" "${flags[@]}" -DGLM53_ROWS_MAX=64 -c "$output_dir/source/native/cooperative_moe.cu" -o "$output_dir/rows64.o" > "$output_dir/build64.log" 2>&1 &
pid64=$!
failed=0
wait "$pid32" || failed=1
wait "$pid64" || failed=1
if [ "$failed" != 0 ]; then cat "$output_dir/build32.log" "$output_dir/build64.log" >&2; exit 1; fi
"$nvcc" "${flags[@]}" -shared "$output_dir/source/native/dispatch.cu" "$output_dir/rows32.o" "$output_dir/rows64.o" -o "$output_dir/cooperative_moe.so" > "$output_dir/link.log" 2>&1
python3 "$source_dir/manifest.py" create "$output_dir"
sha256sum "$output_dir/cooperative_moe.so" "$output_dir/runtime.py" "$output_dir/manifest.json"
