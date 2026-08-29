#!/usr/bin/env bash
# Negative control: TERM-resistant serve process must be KILLed before gate exits.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
pid=""
cleanup() { [ -z "$pid" ] || kill -KILL -- "-$pid" 2>/dev/null || true; }
trap cleanup EXIT
setsid bash -c 'trap "" TERM; exec python3 -c "import time; time.sleep(300)"' &
pid=$!
sleep 1
set +e
POSTREADY_GATE_TEST_FAIL=1 SPARKRUN_SERVE_PID="$pid" \
  bash "$MOD_DIR/postready_gate.sh" http://127.0.0.1:9 test >/tmp/glm53-gate-negative.log 2>&1
rc=$?
set -e
[ "$rc" -eq 92 ] || { cat /tmp/glm53-gate-negative.log; echo "expected gate rc 92, got $rc" >&2; exit 1; }
if kill -0 "$pid" 2>/dev/null; then
  cat /tmp/glm53-gate-negative.log
  echo "TERM-resistant serve pid survived gate" >&2
  exit 1
fi
pid=""
echo "postready termination negative control OK"
