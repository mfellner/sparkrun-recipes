#!/bin/bash
# DFlash2 -> EXL3 quantization + packaging for the dense-EXL3 opt-in.
#
# One command on a GPU node with the recipe image:
#
#   tools/dflash2_exl3_quant.sh              # 6.0 bpw draft pack (the shipping recipe)
#   BITS=5.0 tools/dflash2_exl3_quant.sh     # any other bitrate
#
# What it does:
#   1. stages MiaAI-Lab/exllamav3 pinned at 63b32f0 (MIT, v1.4.2) under
#      $BUILD/src — DFlash2 is quantized uncalibrated (synthetic Hessian, no
#      forwards, no target model);
#   2. converts on one GPU inside the recipe image via
#      tools/dflash2_exl3_convert.py, which keeps k_proj/v_proj BF16 (serving
#      builds the fused context-KV weight from the qkv projection's K/V rows,
#      which must stay a plain tensor) and puts q/o/gate/up/down, the dynconv
#      kernel_projection and fc into the quantization budget;
#   3. packages a servable snapshot via tools/dflash2_exl3_package.py:
#      EXL3 parts for the quantized linears, byte-identical BF16 source
#      tensors for everything else, and the quantization_config block
#      overlay/exl3.py reads (checkpoint-relative model.layers.N prefixes;
#      the serving layer shifts them by the target's layer count).
#
# Env overrides:
#   BITS                    converter bitrate (default 6.0)
#   DRAFT_QUANT_KERNEL_PROJ 0 leaves the dynconv kernel_projection BF16 (default 1)
#   DRAFT_QUANT_FC          0 leaves fc BF16 (default 1)
#   IMG                     container image (default the recipe image)
#   DRAFT_SNAP              BF16 draft snapshot dir (default the HF cache of
#                           incoai/GLM-5.3-Flash-DFlash2 @ dc77ff1c)
#   BUILD                   work dir for the converter source + job state
#                           (default ~/.local/state/dflash2-exl3-build)
#   OUT                     output snapshot dir
#                           (default ~/.local/state/dflash2-exl3-<BITS>bpw)
#   TORCH_CUDA_ARCH_LIST    default 12.1 (GB10)
#
# Changing BITS/knobs gets its own converter work/out dirs under $BUILD (job
# state is never shared). Stage the result as
# $HF_CACHE/hub/models--local--<name>/snapshots/<rev> (+ refs/main) and serve
# it with DFLASH_MODEL=local/<name> DFLASH_REVISION=<rev>.
set -euo pipefail

PIN=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
IMG="${IMG:-glm53-upstream:ca85576-20260918}"
BITS="${BITS:-6.0}"
KP="${DRAFT_QUANT_KERNEL_PROJ:-1}"
FC="${DRAFT_QUANT_FC:-1}"
BUILD="${BUILD:-$HOME/.local/state/dflash2-exl3-build}"
OUT="${OUT:-$HOME/.local/state/dflash2-exl3-${BITS%.*}bpw}"
DRAFT_SNAP="${DRAFT_SNAP:-$HOME/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/dc77ff1c99eeb2df044ee3d4f0094eb033fee410}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TAG="${BITS}bpw-kp${KP}-fc${FC}"

if [ "${1:-}" = "--in-container" ]; then
    # GPU step (inside IMG; /build = $BUILD, $DRAFT_IN = $DRAFT_SNAP ro, /tools ro).
    # torch cu130 wheels ship cusparse.h etc. under nvidia/cu13, not /usr/local/cuda/include
    export CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
    export LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
    cd /build/src
    python3 -c "import marisa_trie" 2>/dev/null || pip install marisa-trie
    # Editable pip install rebuilds the ext in a fresh temp dir every time; instead build
    # once (inplace .so persists on the mounted volume) and import via PYTHONPATH.
    SO=/build/src/exllamav3_ext.cpython-312-aarch64-linux-gnu.so
    if [ ! -e "$SO" ]; then
        echo "== building exllamav3_ext (one-off, ~3 min)"
        pip install --no-deps --no-build-isolation -e . 2>&1 | tail -3
    fi
    export PYTHONPATH=/build/src
    python3 -c "import exllamav3, exllamav3_ext; from exllamav3.version import __version__; print('exllamav3', __version__, exllamav3.__file__)"
    echo "== converting (BITS=$BITS, kernel_projection=$KP, fc=$FC)"
    python3 /tools/dflash2_exl3_convert.py \
        -i "$DRAFT_IN" \
        -w "/build/work-$TAG" \
        -o "/build/out-$TAG" \
        -b "$BITS" \
        --devices 0
    exit 0
fi

# ---- host step
[ -d "$DRAFT_SNAP" ] || { echo "draft snapshot not found: $DRAFT_SNAP (set DRAFT_SNAP)" >&2; exit 1; }
[ -f "$DRAFT_SNAP/config.json" ] || { echo "DRAFT_SNAP must be the snapshot dir holding config.json" >&2; exit 1; }
# HF snapshot entries are relative links into ../../blobs: mount the whole
# repo so they resolve inside the container. A plain directory mounts as-is.
DRAFT_SNAP="$(cd "$DRAFT_SNAP" && pwd)"
if [ "$(basename "$(dirname "$DRAFT_SNAP")")" = "snapshots" ]; then
    DRAFT_MOUNT="$(dirname "$(dirname "$DRAFT_SNAP")")"
    DRAFT_IN="/draft/snapshots/$(basename "$DRAFT_SNAP")"
else
    DRAFT_MOUNT="$DRAFT_SNAP"
    DRAFT_IN=/draft
fi
if [ ! -d "$BUILD/src/exllamav3" ]; then
    echo "== staging MiaAI-Lab/exllamav3 @ $PIN"
    mkdir -p "$BUILD"
    curl -fL "https://github.com/MiaAI-Lab/exllamav3/archive/$PIN.tar.gz" | tar xz -C "$BUILD"
    mv "$BUILD/exllamav3-$PIN" "$BUILD/src"
fi
echo "== GPU conversion: BITS=$BITS kernel_projection=$KP fc=$FC (image $IMG)"
docker run --rm --gpus all --entrypoint bash \
    -v "$BUILD":/build -v "$DRAFT_MOUNT":/draft:ro -v "$SCRIPT_DIR":/tools:ro \
    -e TORCH_CUDA_ARCH_LIST -e BITS -e TAG -e DRAFT_IN="$DRAFT_IN" \
    -e DRAFT_QUANT_KERNEL_PROJ="$KP" -e DRAFT_QUANT_FC="$FC" \
    "$IMG" /tools/dflash2_exl3_quant.sh --in-container
echo "== packaging (CPU)"
mkdir -p "$OUT"
docker run --rm --entrypoint bash \
    -v "$BUILD":/build -v "$DRAFT_MOUNT":/draft:ro -v "$SCRIPT_DIR":/tools:ro -v "$OUT":/out \
    "$IMG" -c "python3 /tools/dflash2_exl3_package.py \
        --src $DRAFT_IN \
        --conv /build/out-$TAG \
        --out /out && chown -R $(id -u):$(id -g) /out"
echo "== done: $OUT"
ls -la "$OUT"
