#!/usr/bin/env bash
# Offline CPU qualification. No GPU devices, network, or serving container.
# GPU kernel checks and EXL3 overlay selfcheck require a separately composed
# GPU runtime; mixed-prefill and tool-concurrency scripts require live serving.
# Missing comma-decimal locales and optional PR230 input are reported as skips.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d"
BASE_IMAGE="vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
: "${GLM53_VLLM_SRC:?set to pristine vLLM 487ecf187 source checkout (see README)}"
# This is expensive and must never run by default or beside the serving model.
# Opt in explicitly and provide a cap; headroom check below includes a reserve.
: "${GLM53_RUN_CPU_SUITE:?refusing: set GLM53_RUN_CPU_SUITE=1 after stopping workloads}"
[[ "$GLM53_RUN_CPU_SUITE" == 1 ]] || { echo 'GLM53_RUN_CPU_SUITE must equal 1' >&2; exit 2; }
: "${GLM53_TEST_MEMORY:?refusing: explicitly set GLM53_TEST_MEMORY (e.g. 12g)}"
MEMORY="$GLM53_TEST_MEMORY"
# Byte conversion is deliberately restricted to integer GiB values.
[[ "$MEMORY" =~ ^([1-9][0-9]*)g$ ]] || { echo 'GLM53_TEST_MEMORY must be integer GiB, e.g. 12g' >&2; exit 2; }
requested_gib=${BASH_REMATCH[1]}
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
required_kib=$(( (requested_gib + 4) * 1024 * 1024 ))
if (( available_kib < required_kib )); then
    echo "refusing: MemAvailable ${available_kib}KiB; require ${required_kib}KiB (${requested_gib}GiB cap + 4GiB host reserve)" >&2
    exit 2
fi
if command -v docker >/dev/null && docker ps --format '{{.Names}} {{.Image}}' | grep -Eqi 'glm-5\.3|glm53'; then
    echo 'refusing: GLM-5.3 serving container is running' >&2
    exit 2
fi
# The boundary-offset cache sweep alone exceeds 4 GiB. RAM and swap are capped.
RSYNC="${GLM53_TEST_RSYNC:-$(command -v rsync || true)}"
POPT="${GLM53_TEST_LIBPOPT:-$(ldconfig -p | awk '/libpopt.so.0 / && !p {p=$NF} END {print p}')}"
: "${RSYNC:?install rsync on the host or set GLM53_TEST_RSYNC}"
: "${POPT:?install libpopt on the host or set GLM53_TEST_LIBPOPT}"
for image in "$IMAGE" "$BASE_IMAGE"; do docker image inspect "$image" >/dev/null; done
common=(--rm --network none --memory "$MEMORY" --memory-swap "$MEMORY" --cpus 1
    --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e EXL3_SELFCHECK_GPU=0 -e CUDA_VISIBLE_DEVICES=
    -e TMPDIR=/review/tmp -e GLM53_DISPOSABLE_CPU_TEST=1
    -e PYTHONPATH=/review/mod:/review/mod/upstream:/usr/local/lib/python3.12/dist-packages
    -e GLM53_SCHEDULER_PY_SRC=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
    -e GLM53_VLLM_SRC=/review/pristine
    -v "$PWD:/review/mod:ro" -v "$(realpath "$GLM53_VLLM_SRC"):/review/pristine:ro"
    -v "$(realpath "$RSYNC"):/usr/bin/rsync:ro"
    -v "$(realpath "$POPT"):/lib/aarch64-linux-gnu/libpopt.so.0:ro")
if [[ -n ${GLM53_PR230_PATCH:-} ]]; then
    common+=(-v "$(realpath "$GLM53_PR230_PATCH"):/review/pr230.py:ro" -e GLM53_PR230_PATCH=/review/pr230.py)
else
    echo 'NOTE: exact PR230 loadclone composition probe will be skipped; set GLM53_PR230_PATCH to the hash-pinned upstream PR artifact.'
fi
status=0
docker run "${common[@]}" "$IMAGE" -c '
    set -uo pipefail
    mkdir -p /review/tmp
    rsync --version >/dev/null || exit 1
    cp /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py /review/tmp/weight_utils-original.py || exit 1
    export GLM53_LOADCLONE_SOURCE=/review/tmp/weight_utils-original.py
    # The recipe installs this overlay at startup; test the resulting disposable
    # image, not the older baked source. A failed application is a hard failure.
    python3 /review/mod/upstream/overlay/patch_indexer_warmup_range.py || exit 1
    cd /review/mod/upstream || exit 1
    failed=0; ran=0; empty=0
    for path in tests/test_*.py; do
        case "$path" in
            tests/test_apc_per_group_retention.py|tests/test_exl3_thin_fast_gpu.py|tests/test_kda_bf16_large_m_gpu.py) continue;;
        esac
        echo "===== $path ====="
        timeout --kill-after=10s 360s python3 -S -m pytest -q -rs -p no:cacheprovider -p upstream_fixture_adapter "$path"
        rc=$?
        if [[ $rc == 5 ]]; then
            case "$path" in
                # Explicit roles: standalone: ABLIT, adaptive-K, dense-FP8,
                # EXL3 build/runtime GPU self-check, scheduler restart fixture;
                # live API: mixed-prefill and tool-concurrency.
                tests/test_ablit.py|tests/test_adaptive_k_patch.py|tests/test_dense_fp8_patch.py|tests/test_exl3_overlay.py|tests/test_mixed_prefill_decode.py|tests/test_scheduler_decode_floor_restart.py|tests/test_tool_concurrency.py)
                    echo "NO PYTEST CASES: $path (validated standalone/live role below)"; empty=$((empty+1));;
                *) echo "UNEXPECTED EMPTY MODULE: $path"; failed=$((failed+1));;
            esac
        elif [[ $rc != 0 ]]; then echo "FAILED MODULE: $path (exit $rc)"; failed=$((failed+1))
        else ran=$((ran+1)); fi
    done
    echo "pytest module summary: $ran passed, $failed failed, $empty no pytest cases; GPU files excluded, retention tested on base image"
    for t in test_scheduler_decode_floor.py test_scheduler_decode_floor_restart.py test_adaptive_k_patch.py test_ablit.py test_dense_fp8_patch.py; do
        echo "===== standalone $t ====="
        timeout --kill-after=10s 180s python3 -S "tests/$t" || failed=$((failed+1))
    done
    cd /review/mod || exit 1
    for t in test_upstream_fixture_adapter.py test_suppress_stops_multitoken.py test_e3_execution_marker.py test_runtime_patch_composition.py; do
        echo "===== standalone $t ====="
        timeout --kill-after=10s 180s python3 -S "$t" || failed=$((failed+1))
    done
    echo "===== standalone APC no-store composition ====="
    GLM53_VLLM_SRC_ROOT=/usr/local/lib/python3.12/dist-packages/vllm GLM53_REQUIRE_VLLM=1 GLM53_REQUIRE_COMPOSITION=1 \
        timeout --kill-after=10s 180s python3 upstream/tests/test_apc_no_store.py || failed=$((failed+1))
    [[ $failed == 0 ]]
' || status=1
# The public image contains an older coordinator overlay; use pristine base.
docker run "${common[@]}" "$BASE_IMAGE" -c '
    set -euo pipefail
    mkdir -p /review/tmp
    timeout --kill-after=10s 180s python3 /review/mod/upstream/tests/test_apc_per_group_retention.py
' || status=1
exit "$status"
