#!/usr/bin/env bash
# Build artifacts only; never select a profile or modify a running service.
# The recipe image has nvcc but not git: verify/archive the pin on the host,
# then run this script in the container against the extracted tree.
set -euo pipefail

upstream_checkout=${1:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
output_dir=${2:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Header archive matching the specialized kernel.cuh. Kernel origin is
# turboderp-org/exllamav3@58d4d732 (two-stage cooperative MoE, MIT).
upstream_pin=02aef45cd681b960a00afcd0749a4ab99e6c1bfe

mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty build output directory.' >&2
  exit 1
}

mkdir -- "$output_dir/upstream"
if command -v git >/dev/null 2>&1 && [ -e "$upstream_checkout/.git" ]; then
  test "$(git -C "$upstream_checkout" rev-parse "$upstream_pin^{commit}")" = "$upstream_pin"
  git -C "$upstream_checkout" archive "$upstream_pin" exllamav3/exllamav3_ext |
    tar -x -C "$output_dir/upstream"
else
  test -d "$upstream_checkout/exllamav3/exllamav3_ext"
  if [ -f "$upstream_checkout/.glm53_coop_upstream_pin" ]; then
    test "$(cat "$upstream_checkout/.glm53_coop_upstream_pin")" = "$upstream_pin"
  fi
  cp -a -- "$upstream_checkout/exllamav3" "$output_dir/upstream/exllamav3"
fi

cp -- "$source_dir/native/cooperative_moe.cu" "$output_dir/glm53_coop.cu"
cp -- "$source_dir/native/cooperative_moe_kernel.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/glm53_coop_kernel.cuh"
cp -- "$source_dir/native/exl3_moe_coop.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh"
cp -- "$source_dir/runtime.py" "$output_dir/runtime.py"

# CUDA 13 supports --frandom-seed for deterministic generated symbol names.
# SOURCE_DATE_EPOCH and the fixed /work paths used by the documented container
# build remove other avoidable variance. The ELF hash is intentionally not a
# contract: a different CUDA/host toolchain may produce a different valid
# binary from these same pinned sources.
export SOURCE_DATE_EPOCH=1789312052
export TZ=UTC
export LC_ALL=C
reproducible_seed=0x474c4d3533434f4f
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
compile_flags=(
  -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr
  --frandom-seed "$reproducible_seed"
  -gencode arch=compute_121a,code=sm_121a
  -shared -Xcompiler -fPIC --ptxas-options=-v
)
"$nvcc" "${compile_flags[@]}" \
  -I "$output_dir/upstream/exllamav3/exllamav3_ext" \
  "$output_dir/glm53_coop.cu" -o "$output_dir/glm53-coop.so" \
  > "$output_dir/cooperative_moe-build.log" 2>&1
mv -- "$output_dir/glm53-coop.so" "$output_dir/cooperative_moe.so"

GLM53_COOP_SOURCE_DIR="$source_dir" \
GLM53_COOP_UPSTREAM_ROOT="$output_dir/upstream/exllamav3/exllamav3_ext" \
GLM53_COOP_OUTPUT_DIR="$output_dir" \
GLM53_COOP_NVCC="$nvcc" \
GLM53_COOP_SEED="$reproducible_seed" \
GLM53_COOP_UPSTREAM_PIN="$upstream_pin" \
python3 - <<'PY'
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


source = Path(os.environ["GLM53_COOP_SOURCE_DIR"])
output = Path(os.environ["GLM53_COOP_OUTPUT_DIR"])
nvcc = os.environ["GLM53_COOP_NVCC"]
nvcc_path = shutil.which(nvcc)
if nvcc_path is None:
    raise RuntimeError(f"nvcc disappeared after compilation: {nvcc}")
uname = os.uname()
manifest = {
    "schema": 1,
    "artifact": "glm53-cooperative-moe",
    "target": "sm_121a",
    "upstream": {
        "repository": "https://github.com/turboderp-org/exllamav3.git",
        "commit": os.environ["GLM53_COOP_UPSTREAM_PIN"],
        "tree_sha256": tree_sha(os.environ["GLM53_COOP_UPSTREAM_ROOT"]),
    },
    "inputs": {
        "native/cooperative_moe.cu": sha(source / "native/cooperative_moe.cu"),
        "native/cooperative_moe_kernel.cuh": sha(source / "native/cooperative_moe_kernel.cuh"),
        "native/exl3_moe_coop.cuh": sha(source / "native/exl3_moe_coop.cuh"),
        "runtime.py": sha(source / "runtime.py"),
        "build.sh": sha(source / "build.sh"),
    },
    "build": {
        "source_date_epoch": int(os.environ["SOURCE_DATE_EPOCH"]),
        "random_seed": os.environ["GLM53_COOP_SEED"],
        "nvcc": subprocess.check_output([nvcc, "--version"], text=True).strip(),
        "nvcc_sha256": sha(Path(nvcc_path).resolve()),
        "host_compiler": subprocess.check_output(
            [os.environ.get("CXX", "c++"), "--version"], text=True
        ).splitlines()[0],
        "system": {
            "sysname": uname.sysname,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "flags": [
            "-std=c++17", "-O3", "--use_fast_math", "-lineinfo",
            "--expt-relaxed-constexpr", "--frandom-seed",
            os.environ["GLM53_COOP_SEED"],
            "-gencode", "arch=compute_121a,code=sm_121a", "-shared",
            "-Xcompiler", "-fPIC", "--ptxas-options=-v",
        ],
    },
    # This binds the manifest to its artifact without imposing a repository-
    # wide expected ELF hash. A valid build from another toolchain records its
    # own output digest here.
    "output": {
        "sha256": sha(output / "cooperative_moe.so"),
        "size": (output / "cooperative_moe.so").stat().st_size,
    },
}
(output / "build-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY
sha256sum "$output_dir/cooperative_moe.so" "$output_dir/runtime.py" \
  "$output_dir/build-manifest.json"
