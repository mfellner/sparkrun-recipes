#!/usr/bin/env bash
set -euo pipefail

# Exact runtime overlay bundle from MiaAI-Lab commit:
# 1caea9a10b26ae93b88d08e82d1e7abb0dc45a42
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$MOD_DIR"
sha256sum -c SHA256SUMS

# Adapt upstream's preflight to SparkRun's per-rank execution. Every rank
# validates every selected local HCA before NCCL starts.
gid_index="${NCCL_IB_GID_INDEX:-3}"
IFS=',' read -r -a hcas <<< "${NCCL_IB_HCA:-}"
python3 verify_roce_gid.py --gid-index "$gid_index" "${hcas[@]}"

# The pinned public image contains Mia's compiled E3 grouped fat-expert
# extension. Install the exact latest pure-Python overlay and ABLIT payload,
# then apply Mia's runtime patch sequence. Optional adaptive-k, dense FP8, and
# ABLIT paths are installed but remain disabled by the recipe.
install -d -m 0755 /opt/glm53/ablit
install -m 0644 upstream/overlay/exl3.py /opt/glm53/exl3.py
install -m 0644 upstream/overlay/ablit_runtime.py /opt/glm53/ablit_runtime.py
install -m 0644 upstream/ablit/* /opt/glm53/ablit/
install -m 0644 upstream/files/chat_template.jinja /opt/glm53/chat_template.jinja

python3 upstream/overlay/patch_glm_video_placeholders.py
python3 upstream/overlay/patch_suppress_stops_in_reasoning.py
python3 patch_suppress_stops_multitoken.py
python3 upstream/overlay/patch_scheduler_decode_floor.py
python3 upstream/overlay/patch_glm5_drafter_group.py
python3 upstream/overlay/patch_hybrid_prefix_hit.py
python3 upstream/overlay/patch_xgrammar_termination.py
python3 upstream/overlay/patch_kpool_tail_slotmap.py
python3 upstream/overlay/patch_spinwait.py
python3 upstream/overlay/patch_adaptive_k.py
python3 patch_e3_execution_marker.py
python3 upstream/overlay/patch_dense_fp8.py
python3 upstream/overlay/patch_indexer_workspace.py
python3 patch_ablit.py
python3 verify_runtime_patch_state.py

# Keep launch-time validation lightweight because imports consume unified-memory
# headroom. The exact source suite and GPU self-check run separately before the
# live launch; every rank still proves that all compiled E3 symbols are loaded.
python3 -c "import torch, exllamav3_ext as e; assert hasattr(e, 'exl3_moe'); assert hasattr(e, 'exl3_fat_gemm'); assert hasattr(e, 'exl3_fat_gemm_scatter'); assert hasattr(e, 'exl3_fat_moe_gateup'); assert hasattr(e, 'exl3_fat_moe_down'); assert hasattr(e, 'exl3_fat_moe_gather')"

echo "[OK] MiaAI-Lab GLM-5.3 EXL3 850K runtime overlays applied fail-closed"
