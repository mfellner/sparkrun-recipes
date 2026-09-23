#!/usr/bin/env bash
# Offline, no-device qualification. Never runs tests against a serving container.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d"
BASE_IMAGE="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
# Five pristine, hash-checked files required by test_draft_kv_compact.py.
: "${GLM53_VLLM_SRC:?set to pristine vLLM 487ecf187 source checkout (see README)}"
# Deliberately bounded while another workload is loaded. The complete suite can
# exceed this cap: unload separately and approve more RAM before raising it.
MEMORY="${GLM53_TEST_MEMORY:-768m}"
for image in "$IMAGE" "$BASE_IMAGE"; do
    docker image inspect "$image" >/dev/null
done
common=(--rm --network none --memory "$MEMORY" --memory-swap "$MEMORY" --cpus 1
    --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e EXL3_SELFCHECK_GPU=0
    -e TMPDIR=/review/tmp -e GLM53_DISPOSABLE_CPU_TEST=1
    -e PYTHONPATH=/review/mod:/review/mod/upstream:/usr/local/lib/python3.12/dist-packages
    -e GLM53_SCHEDULER_PY_SRC=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
    -e GLM53_VLLM_SRC=/review/pristine
    -v "$PWD:/review/mod:ro" -v "$(realpath "$GLM53_VLLM_SRC"):/review/pristine:ro")
docker run "${common[@]}" "$IMAGE" -c '
    set -euo pipefail
    mkdir -p /review/tmp
    cd /review/mod/upstream
    python3 -S -m pytest -q -rs -p no:cacheprovider -p upstream_fixture_adapter tests \
        --ignore=tests/test_apc_per_group_retention.py \
        --ignore=tests/test_exl3_thin_fast_gpu.py \
        --ignore=tests/test_kda_bf16_large_m_gpu.py
    for t in test_scheduler_decode_floor.py test_scheduler_decode_floor_restart.py \
             test_hybrid_prefix_hit.py test_adaptive_k_patch.py; do
        python3 -S "tests/$t"
    done
    cd /review/mod
    python3 -S test_upstream_fixture_adapter.py
    python3 -S test_suppress_stops_multitoken.py
    python3 -S test_e3_execution_marker.py
    python3 -S test_runtime_patch_composition.py
    GLM53_VLLM_SRC_ROOT=/usr/local/lib/python3.12/dist-packages/vllm \
        GLM53_REQUIRE_VLLM=1 GLM53_REQUIRE_COMPOSITION=1 \
        python3 upstream/tests/test_apc_no_store.py
'
# This standalone gate needs pristine coordinator source, not the public image's
# older preinstalled coordinator patch. Do not silently substitute a skip.
docker run "${common[@]}" "$BASE_IMAGE" -c '
    set -euo pipefail
    mkdir -p /review/tmp
    python3 /review/mod/upstream/tests/test_apc_per_group_retention.py
'
