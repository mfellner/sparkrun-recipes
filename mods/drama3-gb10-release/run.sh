#!/usr/bin/env bash
# SPDX-License-Identifier: MIT; Fish code and checkpoint have separate terms.
set -euo pipefail
cd /opt/drama-release
sha256sum --quiet -c manifest.sha256
/opt/drama-env/bin/python perf/apply_patch.py --verify-only /opt/fish-speech
cd /opt/drama
sha256sum --quiet -c manifest.sha256
: "${DRAMA_BIND_IP:?set an explicit target RFC1918 IPv4}"
export FISH_DRAMA3_COMPACT_HEAD="${FISH_DRAMA3_COMPACT_HEAD-1}"
export DRAMA_MODEL_DIR="${DRAMA_MODEL_DIR:-/cache/huggingface/hub/models--fishaudio--drama-3/snapshots/dd64a85cd82e9521f7202610ede97b367fc83946}"
export DRAMA_OUTPUT_DIR="${DRAMA_OUTPUT_DIR:-/cache/huggingface/drama3-fast/outputs}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/huggingface/drama3/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/cache/huggingface/drama3/cuda}"
mkdir -p "$DRAMA_OUTPUT_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd /cache/huggingface
exec /opt/drama-env/bin/python /opt/drama-release/serve.py --host "$DRAMA_BIND_IP" --port "${DRAMA_PORT:-8010}" "$@"
