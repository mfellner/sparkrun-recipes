#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
uv run --with pytest --with torch --with Jinja2 --with numpy python -m pytest -q \
    upstream/tests --ignore=upstream/tests/test_apc_per_group_retention.py
docker run --rm --entrypoint bash \
    -v "$PWD/upstream:/upstream:ro" \
    "$IMAGE" -lc 'python3 /upstream/tests/test_apc_per_group_retention.py'
python3 test_suppress_stops_multitoken.py
