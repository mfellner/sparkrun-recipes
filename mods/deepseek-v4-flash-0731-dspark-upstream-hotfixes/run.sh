#!/usr/bin/env bash
set -euo pipefail

# Exact runtime hotfix bundle from MiaAI-Lab commit:
# a462a9e541c684b58c7f380bbd92c7d851557f31
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
MODEL_REVISION="${DSPARK_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
MODEL_DIR="/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/${MODEL_REVISION}"
ENCODING_SOURCE="${DSPARK_ENCODING_FILE:-${MODEL_DIR}/encoding/encoding_dsv4.py}"
ENCODING_DEST="${VLLM_ROOT}/tokenizers/deepseek_v4_encoding.py"

cd "$MOD_DIR"
sha256sum -c SHA256SUMS

verify_status() {
  local output
  output=$("$@" 2>&1) || {
    printf '%s\n' "$output" >&2
    return 1
  }
  printf '%s\n' "$output"
  if grep -Fq 'NOT APPLIED' <<<"$output" || ! grep -Fq 'APPLIED' <<<"$output"; then
    echo "FATAL: hotfix status did not verify APPLIED: $*" >&2
    return 1
  fi
}

if [ ! -d "$VLLM_ROOT" ]; then
  echo "FATAL: vLLM root not found: $VLLM_ROOT" >&2
  exit 1
fi
if [ ! -f "$ENCODING_SOURCE" ]; then
  echo "FATAL: pinned encoder not found: $ENCODING_SOURCE" >&2
  exit 1
fi
echo 'abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6  '"$ENCODING_SOURCE" | sha256sum -c -
install -m 0644 "$ENCODING_SOURCE" "$ENCODING_DEST"

# Preserve upstream low/high/max reasoning semantics in the Anemll wrapper.
python3 - "$VLLM_ROOT/tokenizers/deepseek_v4.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"'''
new = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"'''
if new not in s:
    if s.count(old) != 1:
        raise SystemExit("FATAL: reasoning-effort patch anchor missing or ambiguous")
    p.write_text(s.replace(old, new, 1))
if new not in p.read_text():
    raise SystemExit("FATAL: reasoning-effort patch verification failed")
print("[OK] reasoning-effort mapping verified")
PY

python3 hotfix-encoding-dsv4-issue21.py
verify_status python3 hotfix-encoding-dsv4-issue21.py --status

if [ "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then
  python3 hotfix-dsv4-issue31-v2-thinking-budget-gpu.py
  verify_status python3 hotfix-dsv4-issue31-v2-thinking-budget-gpu.py --status
fi

python3 hotfix-dsv4-issue55-tool-truncation.py
verify_status python3 hotfix-dsv4-issue55-tool-truncation.py --status

if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" != "1" ]; then
  bash hotfix-nvfp4-ds-mla-issue22.sh
  verify_status bash hotfix-nvfp4-ds-mla-issue22.sh --status
fi
if [ "${DSPARK_SKIP_SPIN_WAIT_HOTFIX:-0}" != "1" ]; then
  bash hotfix-gb10-spin-wait.sh
fi

if [ "${DSPARK_SKIP_HOTFIX:-0}" != "1" ]; then
  for hotfix in \
    hotfix-dsv4-mtp-buffer-50312.sh \
    hotfix-dsv4-adaptive-topk-50004.sh \
    hotfix-dsv4-skip-topk-49486.sh \
    hotfix-dsv4-dense-prefill-indexer-48407.sh \
    hotfix-dsv4-skip-empty-c128-48957.sh \
    hotfix-dsv4-flashmla-workspace-50298.sh \
    hotfix-dsv4-grammar-advance.sh
  do
    bash "$hotfix"
    verify_status bash "$hotfix" --status
  done
fi

python3 hotfix-dsv4-issue27-partial-prefill-concurrency.py
verify_status python3 hotfix-dsv4-issue27-partial-prefill-concurrency.py --status
python3 hotfix-dsv4-issue43-decode-fairness-and-diag.py
verify_status python3 hotfix-dsv4-issue43-decode-fairness-and-diag.py --status
python3 hotfix-dsv4-issue26-hybrid-swa-min.py
verify_status python3 hotfix-dsv4-issue26-hybrid-swa-min.py --status

if [ "${DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX:-0}" != "1" ]; then
  python3 hotfix-dsv4-suppress-stops-in-reasoning.py
  verify_status python3 hotfix-dsv4-suppress-stops-in-reasoning.py --status
fi

if [ "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1" ]; then
  python3 hotfix-dsv4-assistant-final-continuation.py
fi

echo "[OK] MiaAI-Lab DSpark hotfix bundle applied and verified fail-closed"
