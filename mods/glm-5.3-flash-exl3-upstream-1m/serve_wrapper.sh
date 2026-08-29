#!/usr/bin/env bash
# Receive SparkRun-appended native distributed flags, gate rank 0, exec vLLM.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
rank=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--node-rank" ]; then rank="$arg"; break; fi
  prev="$arg"
done
[ -n "$rank" ] || { echo "FATAL: --node-rank missing from native-distributed command" >&2; exit 88; }
if [ "${SPARKRUN_WRAPPER_TEST:-0}" = 1 ]; then
  printf 'rank=%s\n' "$rank"
  exit 0
fi
if [ "$rank" = 0 ]; then
  rm -f /tmp/glm53-postready.rc /tmp/glm53-postready.ok /tmp/glm53-postready.log
  SPARKRUN_SERVE_PID=$$ nohup setsid bash "$MOD_DIR/postready_gate.sh" \
    "http://127.0.0.1:${GLM53_SERVE_PORT:-8000}" \
    "${GLM53_SERVED_MODEL:-GLM-5.3-Flash-EXL3}" \
    > /tmp/glm53-postready.log 2>&1 &
fi
exec /usr/local/bin/vllm serve "$@"
