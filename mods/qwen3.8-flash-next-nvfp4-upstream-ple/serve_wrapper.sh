#!/usr/bin/env bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"

# SparkRun 0.3.6 can downgrade a cluster-only recipe to solo mode when only
# one host is supplied. Enforce this recipe's exact two-rank contract at the
# final execution boundary instead of trusting scheduler metadata alone.
args=("$@")
nnodes=""
node_rank=""
nnodes_count=0
node_rank_count=0
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --nnodes)
      ((i + 1 < ${#args[@]})) || { echo "FATAL: --nnodes requires a value" >&2; exit 64; }
      nnodes="${args[i + 1]}"
      nnodes_count=$((nnodes_count + 1))
      ;;
    --nnodes=*)
      nnodes="${args[i]#--nnodes=}"
      nnodes_count=$((nnodes_count + 1))
      ;;
    --node-rank)
      ((i + 1 < ${#args[@]})) || { echo "FATAL: --node-rank requires a value" >&2; exit 64; }
      node_rank="${args[i + 1]}"
      node_rank_count=$((node_rank_count + 1))
      ;;
    --node-rank=*)
      node_rank="${args[i]#--node-rank=}"
      node_rank_count=$((node_rank_count + 1))
      ;;
  esac
done
if [[ "$nnodes_count" -ne 1 || "$nnodes" != "2" ]]; then
  echo "FATAL: Qwen3.8 Flash Next requires exactly --nnodes 2" >&2
  exit 64
fi
if [[ "$node_rank_count" -ne 1 || ! "$node_rank" =~ ^[01]$ ]]; then
  echo "FATAL: Qwen3.8 Flash Next requires exactly one --node-rank in {0,1}" >&2
  exit 64
fi

terminate_serve() {
  local pid="$1"
  local attempt
  kill -TERM "$pid" 2>/dev/null || true
  for attempt in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  ! kill -0 "$pid" 2>/dev/null
}

if [[ " $* " == *" --headless "* ]]; then
  exec "$VLLM_BIN" serve "$@"
fi

"$VLLM_BIN" serve "$@" &
serve_pid=$!
python3 "$MOD_DIR/postready_gate.py" "$serve_pid" &
gate_pid=$!

# The gate is a required startup phase, not detached telemetry. Wait for it
# first so every non-zero or unexpected gate exit terminates the serving
# process rather than leaving an unaccepted API online.
set +e
wait "$gate_pid"
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
  if ! terminate_serve "$serve_pid"; then
    echo "FATAL: could not terminate vLLM after readiness-gate failure" >&2
    exit 70
  fi
  exit "$gate_rc"
fi

wait "$serve_pid"
serve_rc=$?
set -e
exit "$serve_rc"
