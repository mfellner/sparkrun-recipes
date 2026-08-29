#!/usr/bin/env bash
set -euo pipefail

# Exact runtime overlay bundle from MiaAI-Lab commit:
# 32db610d9207a42e2688a6994d3bfaf7af96eecb
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

python3 patch_glm_video_placeholders.py
python3 patch_suppress_stops_in_reasoning.py
python3 patch_suppress_stops_multitoken.py
python3 patch_scheduler_decode_floor.py
python3 patch_glm5_drafter_group.py
python3 patch_hybrid_prefix_hit.py
python3 patch_xgrammar_termination.py

python3 upstream/tests/test_warm_restart_stdout.py
python3 upstream/tests/test_start_overrides.py
bash test_postready_termination.sh
EXL3_SELFCHECK_GPU=1 python3 test_exl3_overlay.py
python3 test_suppress_stops.py
python3 test_suppress_stops_multitoken.py
python3 test_scheduler_decode_floor.py
python3 test_hybrid_prefix_hit.py
python3 upstream/tests/test_xgrammar_termination.py

echo "[OK] MiaAI-Lab GLM-5.3 EXL3 1M runtime overlay applied and tested fail-closed"
