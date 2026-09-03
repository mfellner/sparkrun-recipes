#!/usr/bin/env bash
set -euo pipefail

IMAGE="vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOD_HOST_DIR="${MOD_HOST_DIR:-$(cd "$SCRIPT_DIR/../../mods/qwen3.8-flash-next-nvfp4-upstream-ple" && pwd)}"
MOD_CONTAINER_DIR="/workspace/mods/qwen3.8-flash-next-nvfp4-upstream-ple"

docker image inspect "$IMAGE" --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}} architecture={{.Architecture}} size={{.Size}}'
docker run --rm \
  --gpus all \
  --network none \
  --entrypoint /bin/bash \
  -e PLE_QUANT_OVERRIDE=fp8 \
  -v "${MOD_HOST_DIR}:${MOD_CONTAINER_DIR}:ro" \
  "$IMAGE" -lc "
    set -e
    bash ${MOD_CONTAINER_DIR}/run.sh
    sha256sum /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
    python3 -c \"from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import _get_ple_embedding_quant_method as f; print(type(f(None, 'model.language_model.ple')).__name__)\"
  "
