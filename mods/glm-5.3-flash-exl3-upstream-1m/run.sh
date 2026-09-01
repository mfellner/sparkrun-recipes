#!/usr/bin/env bash
set -euo pipefail

# Exact runtime overlay bundle from MiaAI-Lab commit:
# c190db1ae17ba8dff20129ed1f308d10c63cf37d
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$MOD_DIR"
sha256sum -c SHA256SUMS

# Latest upstream preflight, adapted to SparkRun's per-rank execution: every
# rank validates every selected local HCA, so both nodes fail before NCCL if
# the configured RoCEv2 GID is empty.
gid_index="${NCCL_IB_GID_INDEX:-3}"
IFS=',' read -r -a hcas <<< "${NCCL_IB_HCA:-}"
if [ "${#hcas[@]}" -eq 0 ]; then
  echo "FATAL: NCCL_IB_HCA is empty" >&2
  exit 1
fi
for hca in "${hcas[@]}"; do
  path="/sys/class/infiniband/${hca}/ports/1/gids/${gid_index}"
  value="$(cat "$path" 2>/dev/null || true)"
  if [ -z "$(printf '%s' "$value" | tr -d ':0')" ]; then
    echo "FATAL: NCCL_IB_GID_INDEX=${gid_index} is empty on ${hca} (${path})" >&2
    exit 1
  fi
  echo "[OK] ${hca} GID ${gid_index} populated: ${value}"
done

# The pinned derived image carries upstream's compiled E2 fat-expert extension.
# Reinstall the exact vendored Python overlay and opt-in ABLIT payload before
# applying the current patch sequence. ABLIT remains disabled by default.
install -m 0644 exl3.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
install -d -m 0755 /opt/glm53/ablit
install -m 0644 ablit_runtime.py /opt/glm53/ablit_runtime.py
install -m 0644 ablit/* /opt/glm53/ablit/
install -m 0644 upstream/files/chat_template.jinja /opt/glm53/chat_template.jinja

python3 patch_glm_video_placeholders.py
python3 patch_suppress_stops_in_reasoning.py
python3 patch_suppress_stops_multitoken.py
python3 patch_scheduler_decode_floor.py
python3 patch_glm5_drafter_group.py
python3 patch_hybrid_prefix_hit.py
python3 patch_xgrammar_termination.py
python3 patch_kpool_tail_slotmap.py
python3 upstream/overlay/patch_spinwait.py
python3 upstream/overlay/patch_indexer_workspace.py
python3 patch_ablit.py

# Patchers above are idempotent and fail on anchor drift. The exact image build
# runs the full upstream source suite, and the published image is GPU-gated on
# GB10 before distribution. Keep per-rank launch validation compact so importing
# test frameworks does not consume the free-memory margin vLLM admits against.
python3 -c "import torch, exllamav3_ext as e; assert hasattr(e, 'exl3_moe'); assert hasattr(e, 'exl3_fat_gemm'); assert hasattr(e, 'exl3_fat_gemm_scatter')"

echo "[OK] MiaAI-Lab GLM-5.3 EXL3 1M runtime overlays applied fail-closed"
