#!/usr/bin/env bash
# Negative control: a TERM-resistant serve-group child must be KILLed.
set -euo pipefail
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
pid=""
child_pid=""
cleanup() {
  [ -z "$pid" ] || kill -KILL -- "-$pid" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT
setsid bash -c '
  trap "exit 0" TERM
  bash -c '\''trap "" TERM; while :; do sleep 1; done'\'' &
  child=$!
  printf "%s\n" "$child" > "$1"
  wait "$child"
' _ "$tmp/child.pid" &
pid=$!
for _ in $(seq 1 100); do
  [ -s "$tmp/child.pid" ] && break
  sleep 0.05
done
[ -s "$tmp/child.pid" ] || { echo "TERM-resistant child PID missing" >&2; exit 1; }
child_pid="$(cat "$tmp/child.pid")"
set +e
POSTREADY_GATE_TEST_FAIL=1 SPARKRUN_SERVE_PID="$pid" SPARKRUN_SERVE_PGID="$pid" \
  bash "$MOD_DIR/postready_gate.sh" http://127.0.0.1:9 test >/tmp/glm53-gate-negative.log 2>&1
rc=$?
set -e
[ "$rc" -eq 92 ] || { cat /tmp/glm53-gate-negative.log; echo "expected gate rc 92, got $rc" >&2; exit 1; }
for target in "$pid" "$child_pid"; do
  if kill -0 "$target" 2>/dev/null; then
    cat /tmp/glm53-gate-negative.log
    echo "serve process-group member $target survived gate" >&2
    exit 1
  fi
done
pid=""
echo "postready process-group termination negative control OK"
