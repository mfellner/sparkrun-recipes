#!/usr/bin/env bash
# Receive SparkRun-appended native distributed flags and supervise every rank.
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
vllm_bin="${GLM53_VLLM_BIN:-/usr/local/bin/vllm}"

RC=/tmp/glm53-postready.rc
OK=/tmp/glm53-postready.ok
LOG=/tmp/glm53-postready.log
GATE_PID_FILE="${GLM53_GATE_PID_FILE:-/tmp/glm53-postready-gate.pid}"
gate_bin="${GLM53_POSTREADY_GATE_BIN:-$MOD_DIR/postready_gate.sh}"
rm -f "$RC" "$OK" "$LOG" "$GATE_PID_FILE"
serve_pgid=""
gate_pgid=""
cleanup_active=0

process_group_alive() {
  local pgid="${1:-}"
  [ -n "$pgid" ] || return 1
  ps -eo pgid=,stat= | awk -v pgid="$pgid" '
    $1 == pgid && $2 !~ /^Z/ { alive=1 }
    END { exit(alive ? 0 : 1) }
  '
}

terminate_group() {
  local pgid="${1:-}"
  [ -n "$pgid" ] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    if ! process_group_alive "$pgid"; then
      [ "${GLM53_TEST_FORCE_CLEANUP_FAILURE:-0}" != 1 ]
      return $?
    fi
    sleep 0.1
  done
  kill -KILL -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    if ! process_group_alive "$pgid"; then
      [ "${GLM53_TEST_FORCE_CLEANUP_FAILURE:-0}" != 1 ]
      return $?
    fi
    sleep 0.1
  done
  if ! process_group_alive "$pgid"; then
    [ "${GLM53_TEST_FORCE_CLEANUP_FAILURE:-0}" != 1 ]
    return $?
  fi
  return 1
}

cleanup() {
  local rc="${1:-$?}"
  local failed=0
  [ "$cleanup_active" -eq 0 ] || return
  cleanup_active=1
  trap - EXIT TERM INT
  terminate_group "$gate_pgid" || {
    printf 'FATAL: gate process group %s survived cleanup\n' "${gate_pgid:-unset}" >&2
    failed=1
  }
  terminate_group "$serve_pgid" || {
    printf 'FATAL: serve process group %s survived cleanup\n' "${serve_pgid:-unset}" >&2
    failed=1
  }
  if [ "$failed" -ne 0 ]; then
    printf 'FATAL: failed to terminate supervised process group(s): serve=%s gate=%s\n' \
      "${serve_pgid:-unset}" "${gate_pgid:-unset}" >&2
    rc=127
  fi
  exit "$rc"
}
trap 'cleanup "$?"' EXIT
trap 'cleanup 143' TERM
trap 'cleanup 130' INT

setsid "$vllm_bin" serve "$@" &
serve_pgid=$!
if [ "$rank" -eq 0 ]; then
  SPARKRUN_SERVE_PID="$serve_pgid" SPARKRUN_SERVE_PGID="$serve_pgid" \
    setsid bash "$gate_bin" \
      "http://127.0.0.1:${GLM53_SERVE_PORT:-8000}" \
      "${GLM53_SERVED_MODEL:-GLM-5.3-Flash-EXL3}" \
      > "$LOG" 2>&1 &
  gate_pgid=$!
  printf '%s\n' "$gate_pgid" > "$GATE_PID_FILE"
fi

set +e
if [ "$rank" -eq 0 ]; then
  completed=""
  wait -n -p completed "$serve_pgid" "$gate_pgid"
  child_rc=$?
else
  wait "$serve_pgid"
  child_rc=$?
  completed="$serve_pgid"
fi
set -e
if [ -n "$gate_pgid" ] && [ "$completed" = "$gate_pgid" ]; then
  if [ "$child_rc" -ne 0 ]; then
    exit "$child_rc"
  fi
  terminate_group "$gate_pgid" || exit 127
  gate_pgid=""
  rm -f "$GATE_PID_FILE"
  set +e
  wait "$serve_pgid"
  serve_rc=$?
  set -e
  exit "$serve_rc"
fi

# The serving leader exited, wait -n failed, or an unexpected child completed.
# EXIT cleanup retains both PGIDs and terminates every surviving group member.
exit "$child_rc"
